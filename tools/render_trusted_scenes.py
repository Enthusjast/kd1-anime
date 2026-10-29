"""Explicit trusted-fixture integration through the complete local run lifecycle.

This entry point accepts only fixed task identifiers. It never calls a model or
submits Slurm work. Run it separately from ordinary unit tests.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from kd1_anime.application import Application
from kd1_anime.candidate_acceptor import CandidateAcceptor
from kd1_anime.config import settings
from kd1_anime.domain.program import SceneProgram
from kd1_anime.eval.benchmark import fixed_tasks, task_plan
from kd1_anime.rendering import RenderProfile, sha256_file
from kd1_anime.run_store import RunRepository, atomic_write_json
from kd1_anime.services.program_compiler import compile_program


def main() -> None:
    tasks = {task.task_id: task for task in fixed_tasks() if task.supported}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", choices=sorted(tasks))
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    selected = args.task or ["geometry_circle", "graph_sine", "formula_quadratic", "unicode_tex"]
    root = Path(tempfile.mkdtemp(prefix="kd1-trusted-render-"))
    root.chmod(0o700)
    settings.WORKSPACE_DIR = root
    settings.RENDER_BACKEND = "local"
    settings.MANIM_RENDERER = "cairo"
    settings.MANIM_QUALITY = "l"
    settings.MANIM_PIXEL_WIDTH = 320
    settings.MANIM_PIXEL_HEIGHT = 180
    settings.MANIM_FRAME_RATE = 15
    settings.ENABLE_AUTO_EVAL = False
    settings.ENABLE_VISUAL_EVAL = False
    settings.RAG_ENABLED = False
    settings.LOCAL_RENDER_TIMEOUT = args.timeout
    settings.MONITOR_POLL_INTERVAL = 1
    settings.LLM_API_KEY = ""
    settings.VISUAL_LLM_API_KEY = ""
    profile = RenderProfile.current()
    rows = []
    for task_id in selected:
        task = tasks[task_id]
        plan = task_plan(task)
        program = SceneProgram.model_validate(task.program)
        accepted = CandidateAcceptor().accept_candidate(
            compile_program(program, plan),
            plan,
            renderer="cairo",
            manim_version=profile.manim_version,
        )
        started = time.monotonic()
        job, video, run_id = Application().submit_existing_scene(
            accepted.code, accepted.class_name, wait=True, backend="local"
        )
        if video is None:
            raise RuntimeError("Trusted render did not publish a final video")
        manifest = RunRepository(root).load_for_resume(run_id)
        if manifest.request_budget.reservations:
            raise RuntimeError("Trusted integration unexpectedly dispatched a model request")
        artifact = manifest.scenes[1].artifact
        if artifact is None or not manifest.execution_spec.matches_environment(
            artifact.environment_fingerprint
        ):
            raise RuntimeError("Trusted fixture lacks a complete execution identity")
        rows.append(
            {
                "task_id": task_id,
                "run_id": run_id,
                "manifest": str(root / "runs" / run_id / "manifest.json"),
                "job_id": job.job_id,
                "code_sha256": accepted.code_sha256,
                "video": str(video),
                "video_sha256": sha256_file(video),
                "metadata": artifact.metadata.model_dump(mode="json"),
                "execution_sha256": artifact.execution_sha256,
                "environment": artifact.environment_fingerprint,
                "elapsed_seconds": time.monotonic() - started,
                "model_requests": 0,
            }
        )
        atomic_write_json(
            root / "report.json",
            {"scope": "trusted_fixture_render", "tasks": rows, "semantic_quality_evaluated": False},
        )
        print(json.dumps({"task_id": task_id, "status": "passed"}), flush=True)
    print(json.dumps({"report": str(root / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
