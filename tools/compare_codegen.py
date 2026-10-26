"""Compare bound evidence, or explicitly execute the fixed model benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kd1_anime.config import settings
from kd1_anime.eval.benchmark import compare_modes, fixed_tasks, offline_baseline
from kd1_anime.orchestrator import Orchestrator
from kd1_anime.run_store import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--bindings", type=Path, help="JSON: mode -> task ID -> run directory")
    action.add_argument(
        "--run-models",
        action="store_true",
        help="Actually call configured models and render; consumes request budgets",
    )
    parser.add_argument("--task", action="append", choices=[task.task_id for task in fixed_tasks()])
    parser.add_argument("--backend", choices=["local", "slurm"], default="local")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bindings:
        bindings = json.loads(args.bindings.read_text())
        report = compare_modes(
            {
                mode: {task: Path(root) for task, root in rows.items()}
                for mode, rows in bindings.items()
            }
        )
    elif args.run_models:
        selected = set(args.task or [task.task_id for task in fixed_tasks()])
        bindings = {}
        for mode in ("llm", "compiler_first"):
            settings.SCENE_CODEGEN_MODE = mode
            bindings[mode] = {}
            for task in fixed_tasks():
                if task.task_id not in selected:
                    continue
                workflow = Orchestrator()
                try:
                    workflow.run(task.prompt, dry_run=args.dry_run, backend=args.backend)
                except Exception as exc:
                    # The run manifest contains the redacted failure evidence.
                    print(
                        json.dumps(
                            {"task": task.task_id, "mode": mode, "outcome": type(exc).__name__}
                        )
                    )
                if (
                    workflow._ctx is not None
                    and (workflow._ctx.paths.root / "manifest.json").is_file()
                ):
                    bindings[mode][task.task_id] = workflow._ctx.paths.root
                if args.output:
                    atomic_write_json(
                        args.output.with_suffix(".bindings.json"),
                        {
                            mode: {task: str(root) for task, root in rows.items()}
                            for mode, rows in bindings.items()
                        },
                    )
        report = compare_modes(bindings)
    else:
        report = offline_baseline()
    if args.output:
        atomic_write_json(args.output, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
