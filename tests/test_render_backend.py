"""统一渲染后端的确定性测试，不启动真实 Manim 或 Slurm。"""

from __future__ import annotations

from pathlib import Path

from kd1_anime.cluster.render_backend import LocalRenderBackend, create_render_backend
from kd1_anime.config import settings
from kd1_anime.rendering import RenderProfile


class FakeProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.pid = 4321
        self.waited = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waited = True
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


def test_backend_factory_rejects_unknown_backend():
    assert create_render_backend("slurm").name == "slurm"
    assert create_render_backend("local").name == "local"

    try:
        create_render_backend("other")
    except ValueError as exc:
        assert "不支持" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("unknown backend was accepted")


def test_local_backend_starts_foreground_process_with_private_attempt_dirs(
    monkeypatch,
    tmp_path: Path,
):
    source = tmp_path / "scenes" / "scene_1.py"
    source.parent.mkdir()
    source.write_text("from manim import *\nclass Demo(Scene): pass\n", encoding="utf-8")
    logs = tmp_path / "logs"
    videos = tmp_path / "videos"
    process = FakeProcess()
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return process

    monkeypatch.setattr("kd1_anime.cluster.render_backend.subprocess.Popen", fake_popen)
    monkeypatch.setattr("kd1_anime.cluster.render_backend.shutil.which", lambda name: None)
    monkeypatch.setattr(settings, "LOCAL_RENDER_TIMEOUT", 60)
    monkeypatch.setattr(settings, "LOCAL_RENDER_MEMORY_MB", 512)

    job = LocalRenderBackend().submit_scene(
        1,
        source,
        "Demo",
        scenes_dir=source.parent,
        logs_dir=logs,
        videos_dir=videos,
        code_sha256="a" * 64,
        render_profile=RenderProfile.current(),
    )

    assert job.backend == "local"
    assert job.job_id.startswith("local-")
    assert job.media_dir.name.startswith("attempt_")
    assert captured.get("shell") is not True
    assert captured["start_new_session"] is True
    assert captured["cwd"] == source.parent.resolve()
    command = captured["args"] if "args" in captured else captured["command"]
    assert "--media_dir" in command
    assert str(job.media_dir) in command


def test_local_backend_poll_and_cancel_do_not_use_slurm(monkeypatch, tmp_path: Path):
    source = tmp_path / "scene.py"
    source.write_text("from manim import *\nclass Demo(Scene): pass\n", encoding="utf-8")
    process = FakeProcess()
    monkeypatch.setattr(
        "kd1_anime.cluster.render_backend.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr("kd1_anime.cluster.render_backend.shutil.which", lambda name: None)
    backend = LocalRenderBackend()
    job = backend.submit_scene(
        1,
        source,
        "Demo",
        scenes_dir=tmp_path,
        logs_dir=tmp_path / "logs",
        videos_dir=tmp_path / "videos",
        render_profile=RenderProfile.current(),
    )

    assert backend.poll_all_statuses([job.job_id]) == {job.job_id: "RUNNING"}
    process.returncode = 0
    assert backend.poll_all_statuses([job.job_id]) == {job.job_id: "COMPLETED"}
    assert backend.cancel_job(job.job_id) is True
