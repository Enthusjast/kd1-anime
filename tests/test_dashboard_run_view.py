from datetime import datetime, timezone

from kd1_anime.dashboard import SceneDashboard
from kd1_anime.queries import RunView, SceneView


def test_dashboard_reconstructs_terminal_dry_run_from_committed_view():
    now = datetime.now(timezone.utc)
    view = RunView(
        run_id="20261028-120000-1234abcd",
        schema_version=11,
        revision=9,
        high_watermark=12,
        state="DONE",
        status="dry_run_complete",
        run_phase="finished",
        goal="math",
        created_at=now,
        updated_at=now,
        scenes=(
            SceneView(
                scene_id=1,
                revision=2,
                input_sha256="a" * 64,
                title="Pause",
                phase="reviewed",
                reviewed=True,
                rendered=False,
                failed=False,
                visual_status="skipped",
                static_verification="passed",
            ),
        ),
    )
    dashboard = SceneDashboard()
    dashboard.on_view(view)
    assert dashboard.run_id == view.run_id and dashboard.total == 1
    assert dashboard.scenes[1].title == "Pause"
    assert dashboard.scenes[1].state == "completed" and "审查" in dashboard.scenes[1].done
    assert dashboard.scenes[1].static_verification == "passed"
