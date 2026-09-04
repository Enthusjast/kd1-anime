import time

import pytest

from kd1_anime.cluster.resource_estimator import RenderResourceProfile
from kd1_anime.cluster.slurm import JobMonitor, SlurmDispatcher, SlurmJob, SlurmMonitorCoordinator
from kd1_anime.config import settings
from kd1_anime.rendering import VideoMetadata


def valid_metadata():
    return VideoMetadata(
        size_bytes=4,
        duration_seconds=1,
        width=settings.MANIM_PIXEL_WIDTH,
        height=settings.MANIM_PIXEL_HEIGHT,
        frame_rate=settings.MANIM_FRAME_RATE,
    )


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


def test_script_uses_per_scene_resource_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "opengl")
    resources = RenderResourceProfile(
        cpus_per_task=8,
        mem_gb=32,
        time_limit="02:00:00",
        gpu_type="A100",
        gpu_count=2,
    )

    script = SlurmDispatcher()._build_script(
        1,
        tmp_path / "scene.py",
        "Demo",
        tmp_path / "media",
        tmp_path / "out",
        tmp_path / "err",
        resource_profile=resources,
    )

    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=32G" in script
    assert "#SBATCH -t 02:00:00" in script
    assert "#SBATCH --gres=gpu:A100:2" in script


def test_opengl_container_receives_pyopengl_platform(monkeypatch, tmp_path):
    """Apptainer cleanenv 下仍要显式传递 EGL/GLX 平台。"""
    image = tmp_path / "manim.sif"
    image.write_bytes(b"image")
    monkeypatch.setattr(settings, "MANIM_RENDERER", "opengl")
    monkeypatch.setattr(settings, "MANIM_OPENGL_PLATFORM", "egl")
    monkeypatch.setattr(settings, "SLURM_GPU_TYPE", "A100")
    monkeypatch.setattr(settings, "SLURM_CONTAINER_IMAGE", image)

    script = SlurmDispatcher()._build_script(
        1, tmp_path / "scene.py", "Demo", tmp_path / "media", tmp_path / "out", tmp_path / "err"
    )

    assert "--cleanenv" in script
    assert "export MANIM_RENDERER=opengl" in script
    assert "--env MANIM_RENDERER=opengl" in script
    assert "--env PYOPENGL_PLATFORM=egl" in script


def test_cairo_script_does_not_pass_write_to_movie(monkeypatch, tmp_path):
    """cairo 渲染器默认写视频，无需 --write_to_movie。"""
    monkeypatch.setattr(settings, "MANIM_RENDERER", "cairo")
    monkeypatch.setattr(settings, "SLURM_GPU_TYPE", "")
    script = SlurmDispatcher()._build_script(
        1, tmp_path / "scene.py", "Demo", tmp_path / "media", tmp_path / "out", tmp_path / "err"
    )
    assert "--write_to_movie" not in script


def test_script_runs_same_renderer_smoke_before_formal_render(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "cairo")
    monkeypatch.setattr(settings, "SMOKE_RENDER_ENABLED", True)
    monkeypatch.setattr(settings, "SMOKE_RENDER_MODE", "video")
    monkeypatch.setattr(settings, "SMOKE_RENDER_QUALITY", "l")
    monkeypatch.setattr(settings, "SMOKE_RENDER_TIMEOUT", 42)

    script = SlurmDispatcher()._build_script(
        1,
        tmp_path / "scene.py",
        "Demo",
        tmp_path / "media",
        tmp_path / "out",
        tmp_path / "err",
        run_root=tmp_path,
    )

    assert script.index("[Smoke]") < script.index("manim render --renderer=cairo -qh")
    assert "timeout 42s" in script
    assert "smoke_scene_1.json" in script
    assert "未生成有效最终 MP4" in script
    assert "ffprobe -v error" in script
    assert "partial_movie_files" in script
    assert '"$smoke_video"' in script


def test_script_runs_import_only_and_short_video_stages(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "cairo")
    monkeypatch.setattr(settings, "SMOKE_RENDER_ENABLED", True)
    monkeypatch.setattr(settings, "SMOKE_RENDER_MODE", "both")
    monkeypatch.setattr(settings, "SMOKE_RENDER_SHORT_ANIMATIONS", 3)

    script = SlurmDispatcher()._build_script(
        1,
        tmp_path / "scene.py",
        "Demo",
        tmp_path / "media",
        tmp_path / "out",
        tmp_path / "err",
        run_root=tmp_path,
    )

    assert "importlib.util" in script
    assert "import-only 检查通过" in script
    assert "--from_animation_number 0,3" in script


def test_script_can_run_fast_frame_canary_without_video_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "cairo")
    monkeypatch.setattr(settings, "SMOKE_RENDER_ENABLED", True)
    monkeypatch.setattr(settings, "SMOKE_RENDER_MODE", "frame")

    script = SlurmDispatcher()._build_script(
        1,
        tmp_path / "scene.py",
        "Demo",
        tmp_path / "media",
        tmp_path / "out",
        tmp_path / "err",
        run_root=tmp_path,
    )

    assert "--format png" in script
    assert "--save_last_frame" in script
    assert "-name 'Demo_*.png'" in script
    assert "未生成有效最后一帧 PNG" in script
    assert "smoke_video=$(find" not in script
    assert "ffprobe -v error" not in script


def test_script_can_disable_smoke_render(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "cairo")
    monkeypatch.setattr(settings, "SMOKE_RENDER_ENABLED", False)

    script = SlurmDispatcher()._build_script(
        1,
        tmp_path / "scene.py",
        "Demo",
        tmp_path / "media",
        tmp_path / "out",
        tmp_path / "err",
        run_root=tmp_path,
    )

    assert "[Smoke]" not in script
    assert '"status": "disabled"' in script


def test_script_pins_resolution_and_frame_rate(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "cairo")
    monkeypatch.setattr(settings, "MANIM_PIXEL_WIDTH", 1280)
    monkeypatch.setattr(settings, "MANIM_PIXEL_HEIGHT", 720)
    monkeypatch.setattr(settings, "MANIM_FRAME_RATE", 30)

    script = SlurmDispatcher()._build_script(
        1, tmp_path / "scene.py", "Demo", tmp_path / "media", tmp_path / "out", tmp_path / "err"
    )

    assert "--resolution 1280,720" in script
    assert "--fps 30" in script


def test_container_can_disable_network(monkeypatch, tmp_path):
    image = tmp_path / "manim.sif"
    image.write_bytes(b"image")
    monkeypatch.setattr(settings, "MANIM_RENDERER", "cairo")
    monkeypatch.setattr(settings, "SLURM_CONTAINER_IMAGE", image)
    monkeypatch.setattr(settings, "SLURM_CONTAINER_DISABLE_NETWORK", True)

    script = SlurmDispatcher()._build_script(
        1, tmp_path / "scene.py", "Demo", tmp_path / "media", tmp_path / "out", tmp_path / "err"
    )

    assert "apptainer exec" in script
    assert "--net --network none" in script


def test_script_rejects_multiline_directive_paths(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="单行"):
        SlurmDispatcher()._build_script(
            1,
            tmp_path / "scene.py",
            "Demo",
            tmp_path / "media",
            tmp_path / "out\n#SBATCH --account=attacker",
            tmp_path / "err",
        )


def test_each_submission_gets_an_isolated_media_directory(monkeypatch, tmp_path):
    dispatcher = SlurmDispatcher()
    scene = tmp_path / "scenes" / "scene_1.py"
    scene.parent.mkdir()
    scene.write_text("from manim import *\n", encoding="utf-8")
    job_ids = iter(["101", "102"])
    monkeypatch.setattr(dispatcher, "submit", lambda script: next(job_ids))

    first = dispatcher.submit_scene(
        1,
        scene,
        "Demo",
        scenes_dir=tmp_path / "scenes",
        logs_dir=tmp_path / "logs",
        videos_dir=tmp_path / "videos",
    )
    second = dispatcher.submit_scene(
        1,
        scene,
        "Demo",
        scenes_dir=tmp_path / "scenes",
        logs_dir=tmp_path / "logs",
        videos_dir=tmp_path / "videos",
    )

    assert first.media_dir != second.media_dir
    assert first.media_dir.parent == second.media_dir.parent == tmp_path / "videos" / "scene_1"


def test_attempt_media_dir_does_not_break_container_run_bind(monkeypatch, tmp_path):
    """带 attempt 子目录时, 容器仍必须绑定整个 run 根目录。"""
    image = tmp_path / "manim.sif"
    image.write_bytes(b"image")
    run_root = tmp_path / "runs" / "20260811-120000-abcdef12"
    scene = run_root / "scenes" / "scene_1.py"
    scene.parent.mkdir(parents=True)
    scene.write_text("from manim import *\n", encoding="utf-8")
    monkeypatch.setattr(settings, "MANIM_RENDERER", "cairo")
    monkeypatch.setattr(settings, "SLURM_CONTAINER_IMAGE", image)

    script_path, _, _, media_dir = SlurmDispatcher().generate_script(
        1,
        scene,
        "Demo",
        scenes_dir=scene.parent,
        logs_dir=run_root / "logs",
        videos_dir=run_root / "videos",
        attempt_token="abcdef123456",
    )

    script = script_path.read_text(encoding="utf-8")
    assert media_dir == run_root / "videos" / "scene_1" / "attempt_abcdef123456"
    assert f"--bind {run_root}:{run_root}" in script
    assert f"#SBATCH -J kd1-{run_root.name}-s1" in script


def test_render_script_records_compute_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "cairo")
    run_root = tmp_path / "20260811-120000-abcdef12"
    scene = run_root / "scenes" / "scene_1.py"
    scene.parent.mkdir(parents=True)
    scene.write_text("from manim import *\n", encoding="utf-8")

    script_path, _, _, _ = SlurmDispatcher().generate_script(
        1,
        scene,
        "Demo",
        scenes_dir=scene.parent,
        logs_dir=run_root / "logs",
        videos_dir=run_root / "videos",
    )

    script = script_path.read_text(encoding="utf-8")
    assert "environment_scene_1.json" in script
    assert "importlib.metadata" in script


def test_submission_uses_captured_render_profile_not_mutated_settings(monkeypatch, tmp_path):
    from kd1_anime.rendering import RenderProfile

    dispatcher = SlurmDispatcher()
    scene = tmp_path / "scenes" / "scene_1.py"
    scene.parent.mkdir()
    scene.write_text("from manim import *\n", encoding="utf-8")
    profile = RenderProfile(
        renderer="cairo",
        quality="m",
        pixel_width=1280,
        pixel_height=720,
        frame_rate=24,
        opengl_platform="egl",
    )
    monkeypatch.setattr(settings, "MANIM_QUALITY", "h")
    monkeypatch.setattr(settings, "MANIM_PIXEL_WIDTH", 1920)
    monkeypatch.setattr(settings, "MANIM_PIXEL_HEIGHT", 1080)
    monkeypatch.setattr(settings, "MANIM_FRAME_RATE", 60)
    monkeypatch.setattr(dispatcher, "submit", lambda script: "123")

    job = dispatcher.submit_scene(
        1,
        scene,
        "Demo",
        scenes_dir=tmp_path / "scenes",
        logs_dir=tmp_path / "logs",
        videos_dir=tmp_path / "videos",
        render_profile=profile,
    )
    script = job.script_path.read_text(encoding="utf-8")

    assert "-qm" in script
    assert "--resolution 1280,720" in script
    assert "--fps 24" in script
    assert job.render_profile == profile


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


def test_explicit_timeout_is_not_overridden_by_legacy_timeout(monkeypatch, tmp_path):
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path, submitted_at=time.time() - 5)
    monkeypatch.setattr(settings, "MONITOR_TIMEOUT", 1)
    monkeypatch.setattr(dispatcher, "cancel_job", lambda jid: pytest.fail("should not cancel"))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    calls = {"count": 0}

    def fake_poll(ids):
        calls["count"] += 1
        return {"123": "PENDING" if calls["count"] == 1 else "COMPLETED"}

    monkeypatch.setattr(dispatcher, "poll_all_statuses", fake_poll)
    monkeypatch.setattr(dispatcher, "validate_completed_job", lambda current: True)
    assert dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1, timeout=10) == {"123": True}


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
    monkeypatch.setattr(dispatcher, "validate_completed_job", lambda job: True)
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


def test_normalize_state_handles_sacct_suffix():
    assert SlurmDispatcher._normalize_state("CANCELLED+ by 0") == "CANCELLED"


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
    monkeypatch.setattr(
        "kd1_anime.cluster.slurm.verify_video", lambda path, profile: valid_metadata()
    )

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


def test_gone_with_normal_text_is_not_marked_failed(monkeypatch, tmp_path):
    """普通课程输出中的 error/failed 单词不能代替真实失败证据。"""
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path)
    job.log_out.write_text(
        "Lesson: error handling is important; the failed example is explained.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dispatcher, "validate_completed_job", lambda current: False)

    assert dispatcher._classify_gone(job) is None


def test_gone_without_artifacts_waits_for_streak(monkeypatch, tmp_path):
    """作业消失且无任何产物 → 不立即判死, 计数达到阈值后按失败交给修复。"""
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path)
    statuses = iter([{"123": "GONE"}, {"123": "GONE"}])
    cancelled = []
    monkeypatch.setattr(settings, "MONITOR_MAX_UNKNOWN", 2)
    monkeypatch.setattr(settings, "MONITOR_ARTIFACT_GRACE", 0)
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: next(statuses))
    monkeypatch.setattr(dispatcher, "cancel_job", lambda jid: cancelled.append(jid) or True)

    result = dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1)

    assert result == {"123": False}
    assert job.status == "FAILED"
    assert cancelled == []  # 作业已消失, 无需 scancel


def test_gone_waits_for_artifact_grace_before_failing(monkeypatch, tmp_path):
    """sacct 尚未出现终态时，GONE 也必须等待共享文件系统产物同步。"""
    import kd1_anime.cluster.slurm as slurm_mod

    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path, submitted_at=100.0)
    clock = [100.0]
    calls = {"validate": 0}
    statuses = iter([{"123": "GONE"}, {"123": "GONE"}])
    monkeypatch.setattr(slurm_mod.time, "time", lambda: clock[0])
    monkeypatch.setattr(settings, "MONITOR_ARTIFACT_GRACE", 60)
    monkeypatch.setattr(settings, "MONITOR_MAX_UNKNOWN", 1)
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: next(statuses))

    def validate(current):
        calls["validate"] += 1
        if calls["validate"] == 1:
            current.failure_reason = "最终 MP4 尚未同步"
            return False
        current.output_path = current.media_dir / "Demo.mp4"
        current.output_path.parent.mkdir(parents=True, exist_ok=True)
        current.output_path.write_bytes(b"video")
        current.output_metadata = valid_metadata()
        return True

    monkeypatch.setattr(dispatcher, "validate_completed_job", validate)
    monkeypatch.setattr(slurm_mod.time, "sleep", lambda _: clock.__setitem__(0, clock[0] + 30))

    assert dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1) == {"123": True}
    assert calls["validate"] == 2


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
    monkeypatch.setattr(
        "kd1_anime.cluster.slurm.verify_video", lambda path, profile: valid_metadata()
    )

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
    monkeypatch.setattr(settings, "MONITOR_ARTIFACT_GRACE", 0)
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


def test_poll_all_statuses_sacct_failure_is_unknown_not_gone(monkeypatch):
    """squeue 可访问但 sacct 失败时，不能把仍可能运行的作业判为 GONE。"""
    from types import SimpleNamespace

    dispatcher = SlurmDispatcher()
    monkeypatch.setattr("kd1_anime.cluster.slurm.shutil.which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        if command[0].endswith("sacct"):
            return SimpleNamespace(returncode=1, stdout="", stderr="accounting unavailable")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kd1_anime.cluster.slurm.subprocess.run", fake_run)

    assert dispatcher.poll_all_statuses(["777"]) == {"777": "UNKNOWN"}
    assert "sacct" in dispatcher.last_status_diagnostic


def test_monitor_query_exception_is_graced_and_recovers(monkeypatch, tmp_path):
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path)
    cancelled = []

    def fail_poll(_ids):
        raise OSError("slurmctld unavailable")

    monkeypatch.setattr(dispatcher, "poll_all_statuses", fail_poll)
    monkeypatch.setattr(dispatcher, "cancel_job", lambda job_id: cancelled.append(job_id) or True)

    monitor = JobMonitor(dispatcher, poll_interval=1)
    monitor.add_job(job)

    # 一次控制面查询异常只记为 UNKNOWN，不能误杀远端作业。
    assert monitor.poll_once() is False
    assert monitor.pending == {"123": job}
    assert monitor.results == {}
    assert cancelled == []
    assert job.status == "UNKNOWN"

    statuses = iter([{"123": "RUNNING"}, {"123": "COMPLETED"}])
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: next(statuses))
    monkeypatch.setattr(dispatcher, "validate_completed_job", lambda current: True)

    assert monitor.poll_once() is False
    assert job.status == "RUNNING"
    assert monitor.poll_once() is True
    assert monitor.pending == {}
    assert monitor.results == {"123": True}
    assert cancelled == []


def test_isolated_attempt_video_ignores_wall_clock_skew(monkeypatch, tmp_path):
    """当前提交专用目录中的视频不应因节点时钟偏差被拒绝。"""
    import os

    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path, submitted_at=time.time())
    job.media_dir = tmp_path / "media" / "attempt_abcdef123456"
    job.media_dir.mkdir(parents=True)
    video = job.media_dir / "Demo.mp4"
    video.write_bytes(b"fake")
    old = time.time() - 3600
    os.utime(video, (old, old))
    monkeypatch.setattr(dispatcher, "validate_completed_job", lambda current: True)

    assert dispatcher._find_final_video(job) == video


def test_squeue_output_captures_actual_start_time():
    dispatcher = SlurmDispatcher()
    seen = {}
    starts = {}

    dispatcher._merge_squeue_output(
        "123|RUNNING|2026-08-06T17:43:30\n124|PENDING|N/A\n",
        seen,
        starts,
    )

    assert seen == {"123": "RUNNING", "124": "PENDING"}
    assert starts["123"] > 0
    assert "124" not in starts


def test_preempted_back_to_pending_resets_run_timeout(monkeypatch, tmp_path):
    """作业被抢占退回 PENDING 后, 之前累计的运行时长应重置, 不误触发 RUN_TIMEOUT。"""
    import kd1_anime.cluster.slurm as slurm_mod

    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path, submitted_at=1000.0)
    clock = [1000.0]
    statuses = iter(["RUNNING", "PENDING", "RUNNING", "COMPLETED"])
    monkeypatch.setattr(slurm_mod.time, "time", lambda: clock[0])
    monkeypatch.setattr(settings, "MONITOR_RUN_TIMEOUT", 50)
    monkeypatch.setattr(settings, "MONITOR_QUEUE_TIMEOUT", 100_000)
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: {"123": next(statuses)})
    monkeypatch.setattr(dispatcher, "cancel_job", lambda jid: True)
    monkeypatch.setattr(dispatcher, "validate_completed_job", lambda job: True)

    def fake_sleep(_seconds):
        clock[0] += 40  # 每次轮询间隔推进 40s

    monkeypatch.setattr(slurm_mod.time, "sleep", fake_sleep)

    result = dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1)

    # 若不重置 running_since, 第二次 RUNNING 时已累计 80s > 50s 会触发 RUN_TIMEOUT
    assert result == {"123": True}
    assert job.status == "COMPLETED"


def test_requeued_job_starts_a_new_queue_timeout_window(monkeypatch, tmp_path):
    """重新排队后的等待时间不应继续消耗第一次提交的排队预算。"""
    import kd1_anime.cluster.slurm as slurm_mod

    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path, submitted_at=1000.0)
    clock = [1000.0]
    statuses = iter(["RUNNING", "PENDING", "RUNNING", "COMPLETED"])
    monkeypatch.setattr(slurm_mod.time, "time", lambda: clock[0])
    monkeypatch.setattr(settings, "MONITOR_RUN_TIMEOUT", 50)
    monkeypatch.setattr(settings, "MONITOR_QUEUE_TIMEOUT", 20)
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: {"123": next(statuses)})
    monkeypatch.setattr(dispatcher, "cancel_job", lambda jid: pytest.fail("不应取消"))
    monkeypatch.setattr(dispatcher, "validate_completed_job", lambda current: True)
    monkeypatch.setattr(slurm_mod.time, "sleep", lambda _: clock.__setitem__(0, clock[0] + 40))

    assert dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1) == {"123": True}


def test_completed_without_final_video_is_failed(monkeypatch, tmp_path):
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path)
    monkeypatch.setattr(settings, "MONITOR_ARTIFACT_GRACE", 0)
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: {"123": "COMPLETED"})

    result = dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1)

    assert result == {"123": False}
    assert job.status == "FAILED"
    assert "最终 MP4" in job.failure_reason


def test_completed_with_invalid_video_is_failed(monkeypatch, tmp_path):
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path, submitted_at=time.time() - 1)
    monkeypatch.setattr(settings, "MONITOR_ARTIFACT_GRACE", 0)
    job.media_dir.mkdir(parents=True)
    (job.media_dir / "Demo.mp4").write_bytes(b"corrupt")
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: {"123": "COMPLETED"})
    monkeypatch.setattr(
        "kd1_anime.cluster.slurm.verify_video",
        lambda path, profile: (_ for _ in ()).throw(ValueError("corrupt mp4")),
    )

    result = dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1)

    assert result == {"123": False}
    assert job.status == "FAILED"
    assert "视频验证失败" in job.failure_reason


def test_completed_job_records_compute_environment(monkeypatch, tmp_path):
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path)
    job.script_path.parent.mkdir(parents=True, exist_ok=True)
    marker = job.script_path.parent.parent / "artifacts" / "environment_scene_1.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        '{"python":"3.12.1","manim":"0.20.1","renderer":"cairo"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(dispatcher, "_find_final_video", lambda current: tmp_path / "Demo.mp4")
    (tmp_path / "Demo.mp4").write_bytes(b"video")
    monkeypatch.setattr(
        "kd1_anime.cluster.slurm.verify_video", lambda path, profile: valid_metadata()
    )

    assert dispatcher.validate_completed_job(job) is True
    assert job.environment_fingerprint["manim"] == "0.20.1"
    assert job.environment_fingerprint["renderer"] == "cairo"


def test_monitor_coordinator_batches_scene_jobs(monkeypatch, tmp_path):
    dispatcher = SlurmDispatcher()
    jobs = [make_job(tmp_path / f"scene_{scene_id}") for scene_id in (1, 2)]
    jobs[1].job_id = "124"
    calls = []
    monkeypatch.setattr(
        dispatcher,
        "poll_all_statuses",
        lambda job_ids: calls.append(list(job_ids)) or {job_id: "COMPLETED" for job_id in job_ids},
    )
    monkeypatch.setattr(dispatcher, "validate_completed_job", lambda job: True)

    coordinator = SlurmMonitorCoordinator(dispatcher, poll_interval=1)
    try:
        for job in jobs:
            coordinator.register(job)
        assert coordinator.wait("123") is True
        assert coordinator.wait("124") is True
    finally:
        coordinator.close()

    assert calls == [["123", "124"]]


def test_monitor_coordinator_cancels_queued_jobs_when_stopped(monkeypatch, tmp_path):
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path)
    cancelled = []
    monkeypatch.setattr(dispatcher, "cancel_job", lambda job_id: cancelled.append(job_id) or True)

    coordinator = SlurmMonitorCoordinator(dispatcher, poll_interval=60)
    coordinator.register(job)
    coordinator.cancel_pending(reason="test stop")

    assert coordinator.wait(job.job_id) is False
    assert cancelled == [job.job_id]
    coordinator.close()


def test_completed_waits_for_delayed_artifact(monkeypatch, tmp_path):
    """Slurm 已完成但 NFS 上的 MP4 延迟出现时，下一轮应识别为成功。"""
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path)
    statuses = iter([{"123": "COMPLETED"}, {"123": "COMPLETED"}])
    calls = {"count": 0}
    monkeypatch.setattr(settings, "MONITOR_ARTIFACT_GRACE", 60)
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: next(statuses))
    monkeypatch.setattr(time, "sleep", lambda _: None)

    def validate(current):
        calls["count"] += 1
        if calls["count"] == 1:
            current.failure_reason = "Slurm 状态为 COMPLETED，但未找到本次作业的最终 MP4"
            return False
        current.output_path = current.media_dir / "Demo.mp4"
        current.output_path.parent.mkdir(parents=True, exist_ok=True)
        current.output_path.write_bytes(b"video")
        current.output_metadata = valid_metadata()
        return True

    monkeypatch.setattr(dispatcher, "validate_completed_job", validate)

    assert dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1) == {"123": True}
    assert calls["count"] == 2


def test_unknown_status_requires_time_grace_before_cancel(monkeypatch, tmp_path):
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path)
    clock = [100.0]
    monkeypatch.setattr("kd1_anime.cluster.slurm.time.time", lambda: clock[0])
    monkeypatch.setattr(settings, "MONITOR_MAX_UNKNOWN", 2)
    monkeypatch.setattr(settings, "MONITOR_UNKNOWN_TIMEOUT", 100)
    statuses = iter([{"123": "UNKNOWN"}, {"123": "UNKNOWN"}, {"123": "UNKNOWN"}])
    cancelled = []
    monkeypatch.setattr(dispatcher, "poll_all_statuses", lambda ids: next(statuses))
    monkeypatch.setattr(dispatcher, "cancel_job", lambda jid: cancelled.append(jid) or True)
    monkeypatch.setattr(
        "kd1_anime.cluster.slurm.time.sleep", lambda _: clock.__setitem__(0, clock[0] + 60)
    )

    assert dispatcher.wait_for_all_jobs({"123": job}, poll_interval=1) == {"123": False}
    assert cancelled == ["123"]


def test_scene_id_and_scene_class_are_validated(tmp_path):
    dispatcher = SlurmDispatcher()
    with pytest.raises(ValueError):
        dispatcher.generate_script(0, tmp_path / "scene.py", "Demo", scenes_dir=tmp_path / "scenes")
    with pytest.raises(ValueError):
        dispatcher.generate_script(
            1, tmp_path / "scene.py", "Demo;touch", scenes_dir=tmp_path / "scenes"
        )


def test_error_log_falls_back_to_stdout_when_stderr_is_empty(tmp_path):
    dispatcher = SlurmDispatcher()
    job = make_job(tmp_path)
    job.log_out.write_text(
        "Traceback (most recent call last):\nValueError: boom\n", encoding="utf-8"
    )

    error_log = dispatcher.get_error_log(job=job)

    assert error_log is not None
    assert "ValueError: boom" in error_log
