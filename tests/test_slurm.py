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
