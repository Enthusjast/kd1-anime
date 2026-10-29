"""Historical runs remain inspectable but never authorize mutating operations."""

import json

import pytest
from test_application import application_boundary as application_boundary
from typer.testing import CliRunner

from kd1_anime.cli import app
from kd1_anime.eval.evaluator import Evaluator
from kd1_anime.event_store import EventStore, EventStoreError
from kd1_anime.run_store import RunRepository
from kd1_anime.stats import write_run_report


def snapshot(root):
    return {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }, {path.relative_to(root) for path in root.rglob("*") if path.is_dir()}


def forbidden(*args, **kwargs):
    pytest.fail("legacy run must be rejected before external or mutating work")


@pytest.fixture(params=range(4, 11))
def legacy_run(application_boundary, request):
    boundary = application_boundary
    boundary.app.plan_only("Show a one-second pause")
    root = next((boundary.workspace / "runs").iterdir())
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = request.param
    manifest.pop("agent")
    manifest["updated_at"] = "2020-01-01T00:00:00Z"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    # Listing and inspection remain supported, without upgrading the old manifest.
    assert RunRepository(boundary.workspace).load(root.name).schema_version == request.param
    return boundary, root


def test_clean_skips_legacy_before_lock_or_job_access(legacy_run, monkeypatch):
    boundary, root = legacy_run
    before = snapshot(boundary.workspace)
    monkeypatch.setattr("kd1_anime.cli.lock_run", forbidden)
    monkeypatch.setattr("kd1_anime.cli._cancel_jobs_before_clean", forbidden)
    monkeypatch.setattr("kd1_anime.cli.shutil.rmtree", forbidden)
    result = CliRunner().invoke(app, ["clean", "--older-than", "1d", "--yes", "--include-running"])
    assert result.exit_code == 0, result.output
    assert root.name in result.output and "只读" in result.output
    assert snapshot(boundary.workspace) == before


def test_legacy_queries_do_not_lock_or_modify_run(legacy_run, monkeypatch):
    boundary, root = legacy_run
    lock = root / ".run.lock"
    lock.unlink(missing_ok=True)
    before = snapshot(boundary.workspace)
    monkeypatch.setattr("kd1_anime.event_store.EventStore.locked", forbidden)
    view = boundary.app.run_view(root.name)
    assert view.run_id == root.name and view.schema_version in range(4, 11)
    assert boundary.app.approval_requests(root.name) == ()
    assert snapshot(boundary.workspace) == before


def test_legacy_events_are_replayed_without_lock_or_mutation(legacy_run, monkeypatch):
    boundary, root = legacy_run
    path = root / "events.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "manifest_revision": 0,
                "scene_id": None,
                "scene_revision": None,
                "event_id": "old",
                "timestamp": "2026-10-27T12:00:00Z",
                "run_id": root.name,
                "event": "checkpoint",
                "data": {"message": "old"},
            }
        )
        + "\n"
    )
    (root / ".run.lock").unlink(missing_ok=True)
    before = snapshot(boundary.workspace)
    manifest = RunRepository(boundary.workspace).load(root.name)
    store = EventStore(root, root.name)
    monkeypatch.setattr(store, "locked", forbidden)
    assert store.replay()[0].event == "checkpoint"
    assert store.watermark() == 0
    with pytest.raises(EventStoreError):
        store.append(name="diagnostic", durability="committed", manifest_revision=0)
    with pytest.raises(EventStoreError):
        store.reconcile(manifest)
    assert snapshot(boundary.workspace) == before


def test_historical_report_writer_rejects_before_file_changes(legacy_run):
    boundary, root = legacy_run
    manifest = RunRepository(boundary.workspace).load(root.name)
    before = snapshot(boundary.workspace)
    with pytest.raises(ValueError, match="只读"):
        write_run_report(manifest, root)
    assert snapshot(boundary.workspace) == before


@pytest.mark.parametrize(
    "target", ["run", "scene", "recent", "compare_baseline", "compare_current"]
)
def test_evaluate_cli_rejects_legacy_before_probe_or_evaluator(legacy_run, monkeypatch, target):
    boundary, root = legacy_run
    if target.startswith("compare_"):
        boundary.app.plan_only("Show a one-second pause")
        current = next(path for path in (boundary.workspace / "runs").iterdir() if path != root)
        args = (
            ["evaluate", current.name, "--compare", root.name]
            if target == "compare_baseline"
            else ["evaluate", root.name, "--compare", current.name]
        )
    elif target == "recent":
        args = ["evaluate"]
    elif target == "scene":
        args = ["evaluate", root.name, "--scene-id", "1"]
    else:
        args = ["evaluate", root.name]
    args += ["--visual"]
    before = snapshot(boundary.workspace)
    monkeypatch.setattr("kd1_anime.cli._ensure_visual_llm_api_available", forbidden)
    monkeypatch.setattr("kd1_anime.eval.Evaluator", forbidden)
    monkeypatch.setattr("kd1_anime.cli.lock_run", forbidden)
    monkeypatch.setattr("kd1_anime.run_store.lock_run", forbidden)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    assert "只读" in result.output
    assert snapshot(boundary.workspace) == before


@pytest.mark.parametrize("entry", ["run", "directory", "scene", "batch", "compare"])
def test_evaluator_rejects_legacy_before_frames_or_requests(legacy_run, monkeypatch, entry):
    boundary, root = legacy_run
    if entry in {"batch", "compare"}:
        boundary.app.plan_only("Show a one-second pause")
        current = next(path for path in (boundary.workspace / "runs").iterdir() if path != root)
    before = snapshot(boundary.workspace)
    evaluator = Evaluator(enable_visual_eval=False)
    monkeypatch.setattr(evaluator, "evaluate_scene_video", forbidden)
    monkeypatch.setattr(evaluator.code_evaluator, "evaluate", forbidden)
    monkeypatch.setattr("kd1_anime.run_store.lock_run", forbidden)
    with pytest.raises(ValueError, match="只读"):
        if entry == "run":
            evaluator.evaluate_run(root.name)
        elif entry == "directory":
            evaluator.evaluate_run(root.name, root)
        elif entry == "scene":
            evaluator.evaluate_run_scene(root.name, 1)
        elif entry == "batch":
            evaluator.evaluate_batch([current.name, root.name])
        else:
            evaluator.compare_runs(current.name, root.name)
    assert snapshot(boundary.workspace) == before
