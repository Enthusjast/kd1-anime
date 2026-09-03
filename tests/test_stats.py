from kd1_anime.run_store import RunManifest, StoredSceneState, write_manifest
from kd1_anime.stats import collect_stats, summarize_manifest


def test_summarize_manifest_counts_scene_outcomes(tmp_path):
    manifest = RunManifest(
        run_id="20260728-120000-1234abcd",
        status="failed",
        state="ERROR",
        user_prompt="prompt",
        output_path=str(tmp_path / "output.mp4"),
        scenes={
            1: StoredSceneState(
                plan={
                    "scene_id": 1,
                    "title": "demo",
                    "duration_seconds": 1,
                    "purpose": "test",
                    "math_concept": "test",
                    "visual_design": "fixed",
                    "camera_movement": "fixed",
                    "visual_flow": ["show"],
                    "key_moments": ["hold"],
                    "computation": "x=x",
                },
                plan_ready=True,
                plan_reviewed=True,
                reviewed=True,
                rendered=True,
            )
        },
    )
    report = summarize_manifest(manifest)
    assert report["scenes"]["rendered"] == 1
    assert report["scenes"]["reviewed"] == 1


def test_collect_stats_reads_event_stage_durations(tmp_path):
    root = tmp_path / "runs" / "20260728-120000-1234abcd"
    root.mkdir(parents=True)
    manifest = RunManifest(
        run_id="20260728-120000-1234abcd",
        status="completed",
        state="DONE",
        user_prompt="prompt",
        output_path=str(root / "output.mp4"),
    )
    write_manifest(root / "manifest.json", manifest)
    (root / "events.jsonl").write_text(
        '{"event":"stage_start","timestamp":"2026-07-28T12:00:00+00:00","data":{"stage":"coding"}}\n',
        encoding="utf-8",
    )

    report = collect_stats(tmp_path)
    assert report["runs"][0]["run_id"] == manifest.run_id
    assert "coding" in report["runs"][0]["stage_durations_seconds"]
