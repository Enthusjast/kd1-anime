"""Independent fixed tasks and an offline compiler-contract baseline.

Run ``python -m kd1_anime.eval.benchmark`` without contacting models or rendering.
Actual pipeline comparisons use ``summarize_runs`` with task-to-run bindings.
"""

from __future__ import annotations

import ast
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from kd1_anime.candidate_acceptor import CandidateAcceptor
from kd1_anime.domain.planning import ScenePlan, VisualElementState
from kd1_anime.domain.program import SceneProgram
from kd1_anime.services.program_compiler import compile_program


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    task_id: str
    category: str
    prompt: str
    supported: bool
    program: dict | None
    failure_scenario: str | None = None


def fixed_tasks() -> tuple[BenchmarkTask, ...]:
    def program(objects, events=None):
        return {
            "scene_id": 1,
            "class_name": "BenchmarkScene",
            "duration_seconds": 4,
            "objects": objects,
            "events": events
            if events is not None
            else [
                {
                    "event_id": "reveal",
                    "operation": "create",
                    "start_seconds": 0,
                    "end_seconds": 2,
                    "source_ids": [item["element_id"] for item in objects],
                }
            ],
        }

    def obj(kind, identifier="subject", **params):
        return {
            "kind": kind,
            "element_id": identifier,
            "variable_name": identifier,
            "position": [0, 0, 0],
            **params,
        }

    tasks = []
    formulas = [
        ("quadratic", r"x^2+2x+1=(x+1)^2"),
        ("triangle", r"a^2+b^2=c^2"),
        ("limit", r"\lim_{x\to0}\frac{\sin x}{x}=1"),
        ("series", r"e^x=\sum_{n=0}^{\infty}\frac{x^n}{n!}"),
    ]
    for name, formula in formulas:
        tasks.append(
            BenchmarkTask(
                "formula_" + name,
                "formula",
                "逐字展示公式 " + formula,
                True,
                program([obj("MathTex", text=formula)]),
            )
        )
    for name, expression, domain in [
        ("quadratic", "x**2", [-2, 2]),
        ("sine", "sin(x)", [-3, 3]),
        ("exponential", "exp(x)", [-1, 1]),
        ("reciprocal", "1/x", [1, 4]),
    ]:
        axes = obj("Axes", "axes", x_range=[-4, 4, 1], y_range=[-2, 5, 1], x_length=6, y_length=4)
        tasks.append(
            BenchmarkTask(
                "graph_" + name,
                "graph",
                f"在给定定义域绘制 {expression}",
                True,
                program(
                    [
                        axes,
                        obj("Graph", axes_element_id="axes", expression=expression, x_range=domain),
                    ]
                ),
            )
        )
    for name, shape in [
        ("circle", obj("Circle", radius=1.75)),
        ("triangle", obj("Polygon", vertices=[[0, 0, 0], [3, 0, 0], [0, 2, 0]])),
        ("segment", obj("Line", start=[-2, 1, 0], end=[2, -1, 0])),
        ("rectangle", obj("Rectangle", width=3, height=1.5)),
    ]:
        tasks.append(
            BenchmarkTask(
                "geometry_" + name, "geometry", "按显式尺寸展示 " + name, True, program([shape])
            )
        )
    tasks.extend(
        [
            BenchmarkTask(
                "continuity_export",
                "continuity",
                "导出半径为 2 的圆供后续场景使用",
                True,
                program([obj("Circle", radius=2, exported=True)]),
            ),
            BenchmarkTask(
                "continuity_replace",
                "continuity",
                "用新公式替换旧公式并导出新公式",
                True,
                program(
                    [
                        obj("MathTex", "old", text="x+x"),
                        obj("MathTex", "subject", text="2x", exported=True),
                    ],
                    [
                        {
                            "event_id": "old_in",
                            "operation": "write",
                            "start_seconds": 0,
                            "end_seconds": 1,
                            "source_ids": ["old"],
                        },
                        {
                            "event_id": "replace",
                            "operation": "replacement_transform",
                            "start_seconds": 1,
                            "end_seconds": 3,
                            "source_ids": ["old"],
                            "target_ids": ["subject"],
                        },
                    ],
                ),
            ),
            BenchmarkTask(
                "unicode_text",
                "unicode_tex",
                "显示中文：面积与边长",
                True,
                program([obj("Text", text="面积与边长")]),
            ),
            BenchmarkTask(
                "unicode_tex",
                "unicode_tex",
                "展示含中文标注的面积公式",
                True,
                program([obj("Tex", text=r"面积 $A=\pi r^2$")]),
            ),
            BenchmarkTask("complex_surface", "complex", "展示三维曲面及其法向量", False, None),
            BenchmarkTask("complex_updater", "complex", "使用连续 updater 驱动切线", False, None),
            BenchmarkTask("complex_camera", "complex", "镜头环绕并推近几何体", False, None),
            BenchmarkTask(
                "complex_overlap",
                "complex",
                "两个重叠时段的独立动画",
                False,
                program(
                    [obj("Circle", radius=1)],
                    [
                        {
                            "event_id": "a",
                            "operation": "create",
                            "start_seconds": 0,
                            "end_seconds": 2,
                            "source_ids": ["subject"],
                        },
                        {
                            "event_id": "b",
                            "operation": "wait",
                            "start_seconds": 1,
                            "end_seconds": 3,
                        },
                    ],
                ),
            ),
            BenchmarkTask(
                "recovery_render",
                "recovery",
                "展示并导出半径为 2 的圆；注入首轮渲染失败，修复后保持尺寸与导出",
                True,
                program([obj("Circle", radius=2, exported=True)]),
                "render_failure_once",
            ),
            BenchmarkTask(
                "recovery_budget",
                "recovery",
                "展示公式 x^2；注入请求预算耗尽，停止新增请求并保留有效视频",
                True,
                program([obj("MathTex", text="x^2")]),
                "budget_exhausted",
            ),
            BenchmarkTask(
                "recovery_submission",
                "recovery",
                "展示边长为 2 的正方形；注入提交确认丢失，禁止重复提交",
                True,
                program([obj("Square", side_length=2)]),
                "submission_unknown",
            ),
            BenchmarkTask(
                "recovery_cancel",
                "recovery",
                "展示从 (0,0) 到 (3,2) 的线段；注入取消失败，保留作业身份并禁止重提",
                True,
                program([obj("Line", start=[0, 0, 0], end=[3, 2, 0])]),
                "cancel_failed",
            ),
        ]
    )
    return tuple(tasks)


def task_plan(task: BenchmarkTask) -> ScenePlan:
    exported = [
        VisualElementState(
            element_id=item["element_id"], variable_name=item["variable_name"], kind=item["kind"]
        )
        for item in (task.program or {}).get("objects", [])
        if item.get("exported")
    ]
    return ScenePlan(
        scene_id=1,
        title=task.task_id,
        duration_seconds=4,
        purpose=task.prompt,
        math_concept=task.prompt,
        visual_design="explicit parameters",
        camera_movement="fixed",
        visual_flow=["reveal"],
        key_moments=["ending"],
        computation=task.prompt,
        new_elements=exported,
    )


def offline_baseline() -> dict:
    rows = []
    for task in fixed_tasks():
        started = time.monotonic()
        error = ""
        accepted = False
        try:
            if task.program is None:
                raise ValueError("Capability uses the free-code path")
            plan = task_plan(task)
            code = compile_program(SceneProgram.model_validate(task.program), plan)
            CandidateAcceptor().accept_candidate(code, plan, manim_version="0.20.1")
            accepted = True
        except ValueError as exc:
            error = str(exc)
        rows.append(
            {
                "task_id": task.task_id,
                "category": task.category,
                "expected_supported": task.supported,
                "compiled": accepted,
                "contract_check_passed": accepted == task.supported,
                "failure_scenario": task.failure_scenario,
                "recovery_verified": None,
                "compile_seconds": time.monotonic() - started,
                "model_requests": 0,
                "fix_rounds": 0,
                "error": error,
            }
        )
    return {
        "schema_version": 1,
        "scope": "offline_compiler_contract",
        "catalog_version": "0.20.1",
        "rendered": False,
        "tasks": rows,
    }


def recovery_evidence(task: BenchmarkTask, manifest) -> bool | None:
    """Return unknown unless the requested fault has observable manifest evidence."""
    scenes = list(manifest.scenes.values())
    if task.failure_scenario == "render_failure_once":
        repaired = [scene for scene in scenes if scene.fix_attempts]
        return all(scene.rendered and scene.reviewed for scene in repaired) if repaired else None
    if task.failure_scenario == "budget_exhausted":
        from collections import Counter

        budget = manifest.request_budget
        counts = Counter(item.scene_id for item in budget.reservations if item.scene_id is not None)
        exhausted = len(budget.reservations) == budget.run_limit or any(
            count == budget.scene_limit for count in counts.values()
        )
        return (
            not budget.integrity_errors()
            and manifest.status in {"completed", "failed", "dry_run_complete"}
            if exhausted
            else None
        )
    if task.failure_scenario in {"submission_unknown", "cancel_failed"}:
        targets = [
            scene
            for scene in scenes
            if (
                task.failure_scenario == "submission_unknown"
                and scene.submission is not None
                and scene.submission.status == "unknown"
            )
            or (
                task.failure_scenario == "cancel_failed"
                and scene.slurm_job is not None
                and scene.slurm_job.status == "CANCEL_FAILED"
            )
        ]
        return (
            all(
                len(scene.submission_attempts) == 1 and (scene.failed or scene.give_up)
                for scene in targets
            )
            if targets
            else None
        )
    return None


def contract_evidence(task: BenchmarkTask, manifest, root: Path) -> bool | None:
    """Recognize an exact independently specified program; other code is unknown."""
    if task.program is None or len(manifest.scenes) != 1:
        return None
    from kd1_anime.run_store import restore_run_path, sha256_text

    scene = next(iter(manifest.scenes.values()))
    if not scene.code_file:
        return None
    try:
        code = restore_run_path(root, scene.code_file).read_text(encoding="utf-8")
        if sha256_text(code) != scene.code_sha256:
            return None
        program = SceneProgram.model_validate(
            {**task.program, "scene_id": scene.plan.scene_id, "class_name": scene.class_name}
        )
        expected_plan = task_plan(task).model_copy(update={"scene_id": scene.plan.scene_id})
        expected = compile_program(program, expected_plan)
        matches = ast.dump(ast.parse(code), include_attributes=False) == ast.dump(
            ast.parse(expected), include_attributes=False
        )
    except (OSError, ValueError, SyntaxError):
        return None
    return True if matches and scene.static_verification.status == "passed" else None


def summarize_runs(bindings: Mapping[str, Path]) -> list[dict]:
    """Compare actual run reports; missing runtime/quality evidence stays unknown."""
    tasks = {task.task_id: task for task in fixed_tasks()}
    rows = []
    for task_id, root in bindings.items():
        if task_id not in tasks:
            raise ValueError(f"Unknown benchmark task: {task_id}")
        from kd1_anime.run_store import RunManifest
        from kd1_anime.stats import summarize_manifest

        manifest = RunManifest.model_validate_json((root / "manifest.json").read_text())
        report = summarize_manifest(manifest, root=root)
        rows.append(
            {
                "task_id": task_id,
                "codegen_mode": manifest.scene_codegen_mode,
                "success": manifest.status == "completed",
                "static_contract_accepted": all(
                    scene.static_verification.status == "passed"
                    for scene in manifest.scenes.values()
                )
                if manifest.scenes
                else None,
                "contract_preserved": contract_evidence(tasks[task_id], manifest, root),
                "recovery_verified": recovery_evidence(tasks[task_id], manifest),
                "model_requests": report["model_requests"],
                "fix_rounds": report["fix_attempts"],
                "elapsed_seconds": (manifest.updated_at - manifest.created_at).total_seconds(),
                "visual_verified": report["scenes"]["visual_verified"],
            }
        )
    return rows


def compare_modes(bindings: Mapping[str, Mapping[str, Path]]) -> dict:
    """Aggregate explicit task/run bindings; unevaluated contracts remain unknown."""
    groups = {}
    for mode, runs in bindings.items():
        if mode not in {"llm", "compiler_first"}:
            raise ValueError(f"Unknown codegen mode: {mode}")
        rows = summarize_runs(runs)
        if any(row["codegen_mode"] != mode for row in rows):
            raise ValueError("Run mode does not match comparison group")
        contracts = [
            row["contract_preserved"] for row in rows if row["contract_preserved"] is not None
        ]
        groups[mode] = {
            "tasks": rows,
            "coverage": len(rows) / len(fixed_tasks()),
            "success_rate": sum(row["success"] for row in rows) / len(rows) if rows else None,
            "contract_preservation_rate": sum(contracts) / len(contracts) if contracts else None,
            "contract_measured_tasks": len(contracts),
            "contract_unknown_tasks": len(rows) - len(contracts),
            "model_requests": sum(row["model_requests"] for row in rows)
            if rows and all(row["model_requests"] is not None for row in rows)
            else None,
            "fix_rounds": sum(row["fix_rounds"] for row in rows),
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
        }
    return {"scope": "bound_runtime_runs", "groups": groups}


if __name__ == "__main__":
    print(json.dumps(offline_baseline(), ensure_ascii=False, indent=2))
