import time
from pathlib import Path

from typer.testing import CliRunner

from kd1_anime.cli import app
from kd1_anime.cluster.slurm import SlurmDispatcher, SlurmJob
from kd1_anime.config import settings
from kd1_anime.run_store import RunRepository


def test_render_copies_source_into_private_run_directory(monkeypatch, tmp_path):
    import kd1_anime.orchestrator as orchestrator_module

    source = tmp_path / "external" / "scene.py"
    source.parent.mkdir()
    source.write_text(
        "from manim import *\nclass Demo(Scene):\n    def construct(self):\n        self.wait()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path / "workspace")
    monkeypatch.setattr(
        orchestrator_module.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )
    captured: dict[str, Path] = {}

    def fake_submit(self, scene_id, python_file, scene_class_name, **kwargs):
        captured["python_file"] = Path(python_file)
        return SlurmJob(
            job_id="123",
            scene_id=scene_id,
            script_path=kwargs["scenes_dir"] / "render_1.sh",
            log_out=kwargs["logs_dir"] / "out",
            log_err=kwargs["logs_dir"] / "err",
            media_dir=kwargs["videos_dir"] / "scene_1",
            scene_class_name=scene_class_name,
            submitted_at=time.time(),
        )

    monkeypatch.setattr(SlurmDispatcher, "submit_scene", fake_submit)

    result = CliRunner().invoke(app, ["render", str(source)])

    assert result.exit_code == 0, result.output
    copied = captured["python_file"]
    assert copied != source
    assert copied.parent.name == "scenes"
    assert copied.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert copied.stat().st_mode & 0o777 == 0o600
    manifest = RunRepository(settings.WORKSPACE_DIR).load(copied.parent.parent.name)
    assert manifest.state == "MONITORING"
    assert manifest.status == "running"
    assert manifest.auto_fix is False
    assert manifest.scenes[1].slurm_job.job_id == "123"
    assert f"Run ID: {manifest.run_id}" in result.output
    assert f"kd1-anime resume {manifest.run_id}" in result.output
