import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_pipeline_scene import SOURCE, inputs

from kd1_anime.config import settings
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.smoke import SmokePipeline
from kd1_anime.run_store import sha256_text


def test_smoke_runs_import_frame_and_video_with_pinned_source(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_SMOKE_RENDER_MODE", "both")
    monkeypatch.setattr(settings, "ADAPTIVE_SMOKE_RENDER", False)
    snapshot = inputs(tmp_path)
    snapshot.scene.code_file = ObjectStore(tmp_path).source(1, SOURCE)
    snapshot.scene.code_sha256 = sha256_text(SOURCE)
    snapshot.scene.class_name = "Pause"
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "--media_dir" in command:
            directory = Path(command[command.index("--media_dir") + 1])
            directory.mkdir(parents=True, exist_ok=True)
            (
                directory / ("Pause.png" if "--save_last_frame" in command else "Pause.mp4")
            ).write_bytes(b"isolated media")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kd1_anime.pipeline.smoke._run_limited_process", run)
    report = SmokePipeline().run(snapshot, SimpleNamespace(checkpoint=lambda: None))
    raw = json.loads((tmp_path / report.path).read_text())
    assert raw["code_sha256"] == snapshot.scene.code_sha256
    assert raw["execution_sha256"] == snapshot.execution_spec.digest()
    assert raw["mode"] == "both" and len(calls) == 3
    assert "--save_last_frame" in calls[1][0] and "--from_animation_number" in calls[2][0]
    assert all(call[1]["memory_mb"] == settings.LOCAL_SMOKE_RENDER_MEMORY_MB for call in calls)
    assert snapshot.scene.local_smoke_status == "pending"


def test_smoke_rejects_tampered_source_before_process(tmp_path, monkeypatch):
    snapshot = inputs(tmp_path)
    snapshot.scene.code_file = ObjectStore(tmp_path).source(1, SOURCE)
    snapshot.scene.code_sha256 = "a" * 64
    snapshot.scene.class_name = "Pause"
    calls = []
    monkeypatch.setattr(
        "kd1_anime.pipeline.smoke._run_limited_process", lambda *a, **k: calls.append(a)
    )
    with pytest.raises(ValueError):
        SmokePipeline().run(
            replace(snapshot, dry_run=True), SimpleNamespace(checkpoint=lambda: None)
        )
    assert not calls
