import time

from kd1_anime.cluster.slurm import SlurmDispatcher, SlurmJob
from kd1_anime.config import settings


def make_job(tmp_path, submitted_at=None):
    return SlurmJob(
        job_id="123",
        scene_id=1,
        script_path=tmp_path / "job.sh",
        log_out=tmp_path / "job.out",
        log_err=tmp_path / "job.err",
        media_dir=tmp_path / "media",
        scene_class_name="Demo",
        submitted_at=submitted_at or time.time(),
    )


def test_cairo_does_not_request_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "cairo")
    monkeypatch.setattr(settings, "SLURM_GPU_TYPE", "A100")
    script = SlurmDispatcher()._build_script(
        1, tmp_path / "scene.py", "Demo", tmp_path / "media", tmp_path / "out", tmp_path / "err"
    )
    assert "--gres=gpu" not in script
    assert "--renderer=cairo" in script


def test_opengl_requires_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "opengl")
    monkeypatch.setattr(settings, "SLURM_GPU_TYPE", "")
    try:
        SlurmDispatcher()._build_script(
            1, tmp_path / "scene.py", "Demo", tmp_path / "media", tmp_path / "out", tmp_path / "err"
        )
    except RuntimeError as exc:
        assert "SLURM_GPU_TYPE" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_opengl_script_forces_write_to_movie(monkeypatch, tmp_path):
    """OpenGL 渲染器必须显式传 --write_to_movie，否则 manim 静默不产出视频。"""
    monkeypatch.setattr(settings, "MANIM_RENDERER", "opengl")
    monkeypatch.setattr(settings, "SLURM_GPU_TYPE", "A100")
    script = SlurmDispatcher()._build_script(
        1, tmp_path / "scene.py", "Demo", tmp_path / "media", tmp_path / "out", tmp_path / "err"
    )
    assert "--renderer=opengl" in script
    assert "--write_to_movie" in script
    assert "--gres=gpu" in script


def test_cairo_script_does_not_pass_write_to_movie(monkeypatch, tmp_path):
    """cairo 渲染器默认写视频，无需 --write_to_movie。"""
    monkeypatch.setattr(settings, "MANIM_RENDERER", "cairo")
    monkeypatch.setattr(settings, "SLURM_GPU_TYPE", "")
    script = SlurmDispatcher()._build_script(
        1, tmp_path / "scene.py", "Demo", tmp_path / "media", tmp_path / "out", tmp_path / "err"
    )
    assert "--write_to_movie" not in script


def test_queue_timeout_cancels_job(monkeypatch, tmp_path):
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path, submitted_at=time.time() - 100)
    cancelled = []
    monkeypatch.setattr(settings, "MONITOR_QUEUE_TIMEOUT", 1)
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: {"123": "PENDING"})
    monkeypatch.setattr(dispatcher, "cancel_job", lambda jid: cancelled.append(jid) or True)
    result = dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1)
    assert result == {"123": False}
    assert cancelled == ["123"]


def test_empty_partition_and_qos_use_scheduler_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "cairo")
    monkeypatch.setattr(settings, "SLURM_PARTITION", "")
    monkeypatch.setattr(settings, "SLURM_QOS", "")
    script = SlurmDispatcher()._build_script(
        1,
        tmp_path / "scene.py",
        "Demo",
        tmp_path / "media",
        tmp_path / "out",
        tmp_path / "err",
    )

    assert "#SBATCH -p" not in script
    assert "#SBATCH --qos=" not in script


def test_cancel_failure_is_preserved_and_not_reported_as_plain_failure(monkeypatch, tmp_path):
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path, submitted_at=time.time() - 100)
    monkeypatch.setattr(settings, "MONITOR_QUEUE_TIMEOUT", 1)
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: {"123": "PENDING"})
    monkeypatch.setattr(dispatcher, "cancel_job", lambda jid: False)

    result = dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1)

    assert result == {"123": False}
    assert job.status == "CANCEL_FAILED"
    assert job.cancelled is False
    assert "禁止自动重提" in job.failure_reason


def test_known_running_state_resets_unknown_streak(monkeypatch, tmp_path):
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path)
    statuses = iter(
        [
            {"123": "UNKNOWN"},
            {"123": "RUNNING"},
            {"123": "UNKNOWN"},
            {"123": "COMPLETED"},
        ]
    )
    cancelled = []
    monkeypatch.setattr(settings, "MONITOR_MAX_UNKNOWN", 2)
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: next(statuses))
    monkeypatch.setattr(dispatcher, "cancel_job", lambda jid: cancelled.append(jid) or True)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    result = dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1)

    assert result == {"123": True}
    assert cancelled == []


def test_submit_uses_parsable_output(monkeypatch, tmp_path):
    from types import SimpleNamespace

    dispatcher = SlurmDispatcher()
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n")
    commands = []
    monkeypatch.setattr("kd1_anime.cluster.slurm.shutil.which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="456;cluster\n", stderr="")

    monkeypatch.setattr("kd1_anime.cluster.slurm.subprocess.run", fake_run)

    assert dispatcher.submit(script) == "456"
    assert commands == [["/usr/bin/sbatch", "--parsable", str(script)]]


def test_submit_timeout_is_not_retried(monkeypatch, tmp_path):
    import subprocess

    import pytest

    dispatcher = SlurmDispatcher()
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n")
    calls = []
    monkeypatch.setattr("kd1_anime.cluster.slurm.shutil.which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, timeout=30)

    monkeypatch.setattr("kd1_anime.cluster.slurm.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="避免重复提交"):
        dispatcher.submit(script)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# GONE (作业已从调度器消失) 与 UNKNOWN (集群查询失败) 的解耦
# ---------------------------------------------------------------------------
def test_poll_all_statuses_gone_vs_unknown(monkeypatch):
    from types import SimpleNamespace

    dispatcher = SlurmDispatcher()
    monkeypatch.setattr("kd1_anime.cluster.slurm.shutil.which", lambda name: f"/usr/bin/{name}")

    # squeue/sacct 都查询成功但都无记录 → GONE
    def fake_run(command, **kwargs):
        assert command[0].endswith(("squeue", "sacct"))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kd1_anime.cluster.slurm.subprocess.run", fake_run)
    assert dispatcher.poll_all_statuses(["111"]) == {"111": "GONE"}

    # squeue 无记录但 sacct 返回终态 → 终态
    def fake_run2(command, **kwargs):
        if command[0].endswith("sacct"):
            return SimpleNamespace(returncode=0, stdout="222|FAILED\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kd1_anime.cluster.slurm.subprocess.run", fake_run2)
    assert dispatcher.poll_all_statuses(["222"]) == {"222": "FAILED"}

    # 集群查询失败 → UNKNOWN
    def fake_run3(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="error")

    monkeypatch.setattr("kd1_anime.cluster.slurm.subprocess.run", fake_run3)
    assert dispatcher.poll_all_statuses(["333"]) == {"333": "UNKNOWN"}


def test_gone_with_video_is_completed(monkeypatch, tmp_path):
    """作业已消失但视频已产出 → 判定为渲染成功 (不再误杀)。"""
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path)
    job.media_dir.mkdir(parents=True)
    (job.media_dir / "Demo.mp4").write_bytes(b"fake")
    monkeypatch.setattr(settings, "MONITOR_MAX_UNKNOWN", 2)
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: {"123": "GONE"})
    monkeypatch.setattr(dispatcher, "cancel_job", lambda jid: True)

    result = dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1)

    assert result == {"123": True}
    assert job.status == "GONE" or job.status == "COMPLETED"
    assert job.cancelled is False


def test_gone_with_error_log_is_failed(monkeypatch, tmp_path):
    """作业已消失但 .err 有回溯 → 判定为失败, 可走自动修复而非永久判死。"""
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path)
    job.log_err.write_text("Traceback (most recent call last):\n  boom\n")
    monkeypatch.setattr(settings, "MONITOR_MAX_UNKNOWN", 2)
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: {"123": "GONE"})
    monkeypatch.setattr(dispatcher, "cancel_job", lambda jid: True)

    result = dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1)

    assert result == {"123": False}
    assert job.status == "FAILED"
    assert "依据日志判定为失败" in job.failure_reason


def test_gone_without_artifacts_waits_for_streak(monkeypatch, tmp_path):
    """作业消失且无任何产物 → 不立即判死, 计数达到阈值后按失败交给修复。"""
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path)
    statuses = iter([{"123": "GONE"}, {"123": "GONE"}])
    cancelled = []
    monkeypatch.setattr(settings, "MONITOR_MAX_UNKNOWN", 2)
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: next(statuses))
    monkeypatch.setattr(dispatcher, "cancel_job", lambda jid: cancelled.append(jid) or True)

    result = dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1)

    assert result == {"123": False}
    assert job.status == "FAILED"
    assert cancelled == []  # 作业已消失, 无需 scancel


def test_cancel_job_invalid_id_is_benign(monkeypatch):
    """scancel 报 Invalid job id → 作业已不在调度器, 视为取消成功 (无重复作业风险)。"""
    from types import SimpleNamespace

    dispatcher = SlurmDispatcher()
    monkeypatch.setattr("kd1_anime.cluster.slurm.shutil.which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=1, stdout="", stderr="slurm job 123: Invalid job id specified\n"
        )

    monkeypatch.setattr("kd1_anime.cluster.slurm.subprocess.run", fake_run)
    assert dispatcher.cancel_job("123") is True


def test_gone_with_nested_video_is_completed(monkeypatch, tmp_path):
    """manim 成品在嵌套路径 <media_dir>/videos/<file>/<quality>/<Scene>.mp4
    下也要能识别 (HPC 实测场景: 源文件 scene_3.py → videos/scene_3/1080p30/)."""
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path)
    nested = job.media_dir / "videos" / "scene_3" / "1080p30"
    nested.mkdir(parents=True)
    (nested / "Demo.mp4").write_bytes(b"fake")
    monkeypatch.setattr(settings, "MONITOR_MAX_UNKNOWN", 2)
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: {"123": "GONE"})
    monkeypatch.setattr(dispatcher, "cancel_job", lambda jid: True)

    result = dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1)

    assert result == {"123": True}


def test_gone_with_stale_video_is_not_completed(monkeypatch, tmp_path):
    """提交前就存在的旧 mp4 (上一次修复尝试残留) 不能当作本次成功产物。"""
    import os

    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path, submitted_at=time.time())
    nested = job.media_dir / "videos" / "scene_3" / "1080p30"
    nested.mkdir(parents=True)
    vid = nested / "Demo.mp4"
    vid.write_bytes(b"fake")
    old = time.time() - 3600
    os.utime(vid, (old, old))
    monkeypatch.setattr(settings, "MONITOR_MAX_UNKNOWN", 1)
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: {"123": "GONE"})
    monkeypatch.setattr(dispatcher, "cancel_job", lambda jid: True)

    result = dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1)

    assert result == {"123": False}
    assert job.status == "FAILED"


def test_poll_all_statuses_squeue_invalid_id_falls_back_to_user_query(monkeypatch):
    """squeue -j 对已消失的 job id 非零退出时, 改用 squeue -u 兜底判 GONE。"""
    from types import SimpleNamespace

    dispatcher = SlurmDispatcher()
    monkeypatch.setattr("kd1_anime.cluster.slurm.shutil.which", lambda name: f"/usr/bin/{name}")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0].endswith("squeue") and "-j" in command:
            # 已消失的 job id 在部分集群上会让 squeue -j 非零退出
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="slurm_load_jobs error: Invalid job id specified",
            )
        if command[0].endswith("squeue") and "-u" in command:
            # 按用户名查询成功, 该作业不在队列 → 已消失
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kd1_anime.cluster.slurm.subprocess.run", fake_run)

    assert dispatcher.poll_all_statuses(["444"]) == {"444": "GONE"}
    assert any("-u" in c for c in calls)


def test_poll_all_statuses_both_squeue_fail_is_unknown(monkeypatch):
    """squeue -j 与 -u 都失败 (集群不可查) → UNKNOWN, 不能误判 GONE。"""
    from types import SimpleNamespace

    dispatcher = SlurmDispatcher()
    monkeypatch.setattr("kd1_anime.cluster.slurm.shutil.which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="connect refused")

    monkeypatch.setattr("kd1_anime.cluster.slurm.subprocess.run", fake_run)
    assert dispatcher.poll_all_statuses(["555"]) == {"555": "UNKNOWN"}
