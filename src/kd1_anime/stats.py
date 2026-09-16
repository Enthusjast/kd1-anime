"""从运行清单和事件日志计算离线流水线统计。"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from kd1_anime.run_store import RunManifest, RunRepository, atomic_write_json


def _read_events(root: Path) -> list[dict[str, Any]]:
    path = root / "events.jsonl"
    if not path.is_file() or path.is_symlink():
        return []
    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and isinstance(item.get("event"), str):
                events.append(item)
    except OSError:
        return []
    return events


def _stage_durations(events: list[dict[str, Any]], updated_at: datetime) -> dict[str, float]:
    starts: list[tuple[str, datetime]] = []
    for event in events:
        if event.get("event") != "stage_start":
            continue
        stage = event.get("data", {}).get("stage") if isinstance(event.get("data"), dict) else ""
        timestamp = event.get("timestamp")
        if not stage or not timestamp:
            continue
        try:
            parsed = datetime.fromisoformat(str(timestamp))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            continue
        starts.append((str(stage), parsed))
    durations: dict[str, float] = {}
    for index, (stage, started) in enumerate(starts):
        ended = starts[index + 1][1] if index + 1 < len(starts) else updated_at
        seconds = max(0.0, (ended - started).total_seconds())
        durations[stage] = round(durations.get(stage, 0.0) + seconds, 3)
    return durations


def summarize_manifest(manifest: RunManifest, *, root: Path | None = None) -> dict[str, Any]:
    """汇总单次运行，不执行网络请求、Slurm 查询或代码。"""

    scenes = list(manifest.scenes.values())
    failures = Counter(
        scene.failure_category or "unknown" for scene in scenes if scene.failed or scene.give_up
    )
    events = _read_events(root) if root is not None else []
    event_counts = Counter(str(event["event"]) for event in events)
    review_artifacts = (
        len(list((root / "artifacts").glob("code_review_scene_*.json"))) if root else 0
    )
    plan_review_artifacts = (
        len(list((root / "artifacts").glob("plan_review_scene_*.json"))) if root else 0
    )
    scene_details = {
        str(scene_id): {
            "phase": scene.phase,
            "failure_category": scene.failure_category,
            "review_round": scene.review_round,
            "plan_review_round": scene.plan_review_round,
            "fix_attempts": scene.fix_attempts,
            "candidate_count": len(getattr(scene, "candidates", [])),
            "smoke_status": scene.local_smoke_status,
            "static_verification": scene.static_verification.model_dump(mode="json"),
            "execution_verification": scene.execution_verification.model_dump(mode="json"),
            "visual_verification": scene.visual_verification.model_dump(mode="json"),
            "capability_status": getattr(scene, "capability_status", "pending"),
            "resource_profile": (
                scene.resource_profile.model_dump(mode="json")
                if getattr(scene, "resource_profile", None) is not None
                else None
            ),
        }
        for scene_id, scene in sorted(manifest.scenes.items())
    }
    return {
        "run_id": manifest.run_id,
        "status": manifest.status,
        "state": manifest.state,
        "backend": getattr(manifest, "backend", "slurm"),
        "created_at": manifest.created_at.isoformat(),
        "updated_at": manifest.updated_at.isoformat(),
        "scene_count": len(scenes),
        "scenes": {
            "plan_ready": sum(bool(scene.plan_ready) for scene in scenes),
            "plan_reviewed": sum(bool(scene.plan_reviewed) for scene in scenes),
            "reviewed": sum(bool(scene.reviewed) for scene in scenes),
            "rendered": sum(bool(scene.rendered) for scene in scenes),
            "failed": sum(bool(scene.failed) for scene in scenes),
            "give_up": sum(bool(scene.give_up) for scene in scenes),
            "safe_fallback": sum(bool(scene.safe_fallback_used) for scene in scenes),
            "visual_passed": sum(scene.visual_status == "passed" for scene in scenes),
            "static_verified": sum(
                scene.static_verification.status == "passed" for scene in scenes
            ),
            "execution_verified": sum(
                scene.execution_verification.status == "passed" for scene in scenes
            ),
            "visual_verified": sum(
                scene.visual_verification.status in {"passed", "warning", "unknown"}
                for scene in scenes
            ),
        },
        "review_attempts": review_artifacts,
        "plan_review_attempts": plan_review_artifacts,
        "fix_attempts": sum(int(scene.fix_attempts) for scene in scenes),
        "failure_categories": dict(sorted(failures.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "stage_durations_seconds": _stage_durations(events, manifest.updated_at),
        "scene_details": scene_details,
    }


def write_run_report(manifest: RunManifest, root: Path) -> Path:
    """把离线统计写入当前 run，供用户和后续调参直接查看。"""

    report_path = root / "run_report.json"
    report = summarize_manifest(manifest, root=root)
    atomic_write_json(report_path, report)
    return report_path


def collect_stats(
    workspace_dir: Path,
    run_id: str | None = None,
    *,
    limit: int = 100,
) -> dict[str, Any]:
    """收集最近运行或指定运行的统计。"""

    repository = RunRepository(workspace_dir)
    manifests = [repository.load(run_id)] if run_id else repository.list()[:limit]
    items = [
        summarize_manifest(manifest, root=repository.run_root(manifest.run_id))
        for manifest in manifests
    ]
    return {
        "run_id": run_id,
        "runs": items,
        "read_errors": list(repository.list_errors),
    }


__all__ = ["collect_stats", "summarize_manifest", "write_run_report"]
