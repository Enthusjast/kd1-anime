"""Immutable visual worker contracts with local media and endpoint doubles only."""

import base64
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_application import ArtifactMerger
from test_pipeline_render import final_inputs, rendered
from test_pipeline_render import stage as stage

from kd1_anime.domain.runtime import RunConfig, VisualEvalProfile
from kd1_anime.eval.metrics import EvalMetric, EvalResult, QualityScore
from kd1_anime.eval.visual_eval import (
    FrameSample,
    VisualAnalysisResult,
    VisualEvaluator,
    VisualIssue,
)
from kd1_anime.pipeline.finalization import FinalizationPipeline
from kd1_anime.pipeline.visual import VisualPipeline, verify_report
from kd1_anime.rendering import sha256_file
from kd1_anime.request_budget import RequestBudgetExceeded


def configured(inputs, **values):
    config = json.loads(inputs.config.values_json)
    config.update({"ENABLE_VISUAL_EVAL": True, "GENERATION_MODE": "strict"})
    config.update(values)
    return replace(inputs, config=RunConfig(values_json=json.dumps(config)))


def analysis(score=5, *, target=None, category="layout", boundary_ids=()):
    dimension = {"score": score, "comprehensive_evaluation": "local evaluation"}
    issues = (
        []
        if target is None
        else [
            VisualIssue(
                severity="major",
                category=category,
                repair_target=target,
                frame_ids=["F01"],
                boundary_ids=list(boundary_ids),
                evidence="Visible issue",
                recommendation="Repair only this issue",
            )
        ]
    )
    return VisualAnalysisResult("Local result", *(dimension.copy() for _ in range(5)), issues)


def sample(path, *, scene_id=None, role="middle", frame_id="F01", boundary_id=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    # A complete 4x4 RGB PNG, generated with stdlib zlib rather than optional Pillow.
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGMIqFgA"
            "RwzEcQBEUhaBF3VqlAAAAABJRU5ErkJggg=="
        )
    )
    return FrameSample(
        frame_id=frame_id,
        path=path,
        image_sha256=sha256_file(path),
        scene_id=scene_id,
        role=role,
        boundary_id=boundary_id,
    )


def visual_stage(results, *, request_context=None, fixer_factory=None):
    results = iter(results)
    calls = SimpleNamespace(evaluators=[], evaluated=[], fixes=[])
    control = SimpleNamespace(check=lambda: None)
    request_context = request_context or (lambda scene_id, stage: (scene_id, stage))

    class Visual:
        def __init__(self):
            self.agent = SimpleNamespace()

        def evaluate_video_frames(self, samples, description, **kwargs):
            calls.evaluated.append((self.agent, samples, kwargs))
            result = next(results)
            if isinstance(result, Exception):
                raise result
            return result

    class Evaluator:
        def __init__(self, **kwargs):
            self.visual_evaluator = Visual() if kwargs["enable_visual_eval"] else None
            calls.evaluators.append((self, kwargs))

        def evaluate_scene_video(self, video, *, output_dir, **kwargs):
            frames = [sample(output_dir / "scene.png")]
            return self.visual_evaluator.evaluate_video_frames(frames, ""), frames

        def extract_video_samples(self, video, output_dir, **kwargs):
            return [sample(output_dir / "final.png")]

        def extract_boundary_samples(self, rendered, output_dir, **kwargs):
            left, right = rendered[:2]
            return [
                sample(
                    output_dir / "end.png",
                    scene_id=left[0],
                    role="boundary_end",
                    boundary_id=f"{left[0]}->{right[0]}",
                ),
                sample(
                    output_dir / "start.png",
                    scene_id=right[0],
                    role="boundary_start",
                    frame_id="F02",
                    boundary_id=f"{left[0]}->{right[0]}",
                ),
            ]

        def evaluate_code(self, code, **kwargs):
            return EvalResult(run_id="local", scores=[QualityScore(EvalMetric.CODE_STYLE, 1)])

        def _evaluate_efficiency(self, root, *, manifest):
            assert manifest.scenes
            return []

    class Fixer:
        def fix(self, original, feedback, **kwargs):
            calls.fixes.append((self, original, feedback, kwargs))
            return original.replace("self.wait(1)", "self.wait(2)")

    return VisualPipeline(
        Evaluator, fixer_factory or Fixer, request_context=request_context, control=control
    ), calls


def evaluated_scene(stage, result=None, **settings):
    inputs = configured(rendered(stage), **settings)
    pipeline, calls = visual_stage([result or analysis()])
    output = pipeline.scene(inputs, stage[2])
    return replace(inputs, scene=output.scene), output, pipeline, calls


def second_scene(inputs):
    scene = inputs.scene.model_copy(deep=True)
    scene.plan = scene.plan.model_copy(update={"scene_id": 2})
    scene.artifact = scene.artifact.model_copy(update={"scene_id": 2})
    scene.slurm_job = scene.slurm_job.model_copy(update={"scene_id": 2})
    return scene


def test_scene_reports_bind_every_identity_and_use_independent_agents(stage):
    inputs, output, pipeline, calls = evaluated_scene(stage)
    assert output.scene.visual_status == "passed" and output.repair_route == "none"
    assert inputs.scene.visual_verification.code_sha256 == inputs.scene.code_sha256
    report = verify_report(inputs, inputs.scene, output.report)
    assert report["binding"]["video_sha256"] == inputs.scene.artifact.video_sha256
    assert report["binding"]["inherited_elements_sha256"]
    assert report["frames"][0]["path"].startswith("artifacts/frames/")
    agent = calls.evaluated[0][0]
    assert agent.request_context == (1, "visual_evaluating")
    assert agent.control is pipeline.control
    replay = pipeline.scene(inputs, stage[2])
    assert replay.report == output.report and len(calls.evaluators) == 1


@pytest.mark.parametrize("mode", ["python", "json", "json_text"])
def test_visual_outputs_roundtrip_exact_submission_identities(stage, mode):
    from kd1_anime.pipeline.visual import BoundaryVisualOutput, VisualSceneOutput

    inputs = rendered(stage)
    scene = inputs.scene.model_copy(update={"submission_attempts": [inputs.scene.submission]})
    output = VisualSceneOutput(scene=scene)
    boundary = BoundaryVisualOutput(scenes=(scene,), state_ledger=inputs.state_ledger)
    for value in (output, boundary):
        if mode == "json_text":
            restored = type(value).model_validate_json(value.model_dump_json())
        else:
            restored = type(value).model_validate(value.model_dump(mode=mode))
        assert restored == value
        restored_scene = (
            restored.scene if isinstance(restored, VisualSceneOutput) else restored.scenes[0]
        )
        assert restored_scene.submission.token == stage[1].token
        assert restored_scene.submission_attempts[0].token == stage[1].token


def test_visual_output_executor_revalidates_identity_and_privacy_before_commit(stage):
    from kd1_anime.agent_runtime.contracts import EvaluateVisual
    from kd1_anime.agent_runtime.policy import PolicyDecision, ToolPolicySpec
    from kd1_anime.agent_runtime.tools import (
        ExecutionAdapters,
        ToolDefinition,
        ToolExecutor,
        ToolRegistry,
        ToolResult,
    )
    from kd1_anime.commits import ScenePatch
    from kd1_anime.domain.dependencies import WorkIdentity
    from kd1_anime.pipeline.visual import VisualSceneOutput
    from kd1_anime.services.results import StageResult

    inputs = rendered(stage)
    identity = WorkIdentity(inputs.scene.revision, inputs.scene.code_sha256)
    action = EvaluateVisual(
        run_id=inputs.run_id,
        resource_id=f"run:{inputs.run_id}/scene:1",
        input_sha256=identity.input_sha256,
        work_identity=identity,
        scene_id=1,
        video={
            "path": inputs.scene.artifact.video_path,
            "sha256": inputs.scene.artifact.video_sha256,
            "kind": "video",
        },
    )
    raw = VisualSceneOutput.model_construct(
        scene=inputs.scene, feedback="<think>private reasoning</think>token=credential"
    )
    committed = []
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            policy=ToolPolicySpec(
                action="evaluate_visual", permission="exec", side_effect="commit"
            ),
            input_model=EvaluateVisual,
            output_model=VisualSceneOutput,
            execute=lambda action, context: ToolResult(
                StageResult(identity, raw),
                ScenePatch(scene_id=1, identity=identity, replacement=inputs.scene),
            ),
        )
    )
    executor = ToolExecutor(
        registry,
        ExecutionAdapters(
            latest_intent=lambda action, session: None,
            persist_intent=lambda intent: None,
            persist_invocation=lambda invocation: None,
            consume_grant=lambda grant, intent: None,
            commit=lambda intent, invocation, result: (committed.append(result), 1)[1],
            emit_committed=lambda intent, invocation, revision: None,
        ),
    )
    outcome = executor.execute(
        action, "session-visual", PolicyDecision(effect="allow", reason="test")
    )
    assert outcome.status == "completed" and len(committed) == 1
    validated = committed[0].stage.value
    assert validated.scene.submission.token == stage[1].token
    assert validated.feedback == "<redacted>token=<redacted>"
    assert validated is not raw


def test_visual_roundtrip_known_secret_submission_token_fails_closed(stage):
    from pydantic import ValidationError

    from kd1_anime.agent_runtime.contracts import sanitize_record
    from kd1_anime.pipeline.visual import VisualSceneOutput

    inputs = rendered(stage)
    safe = sanitize_record(
        {"scene": inputs.scene.model_dump(mode="json")}, (inputs.scene.submission.token,)
    )
    with pytest.raises(ValidationError):
        VisualSceneOutput.model_validate(safe)


def test_visual_endpoint_profile_never_inherits_main_endpoint(stage):
    inputs = configured(rendered(stage), LLM_BASE_URL="https://main.invalid")
    with inputs.config.activate():
        evaluator = VisualEvaluator()
        assert evaluator.profile.env_prefix == "VISUAL_LLM"
        assert evaluator.profile.silent_stream is False
        assert evaluator.profile.base_url == inputs.config.resolve().VISUAL_LLM_BASE_URL


@pytest.mark.parametrize("change", ["frame", "report", "video", "code", "inherited"])
def test_stale_visual_evidence_fails_closed_before_new_endpoint_dispatch(stage, change):
    inputs, output, pipeline, calls = evaluated_scene(stage)
    if change == "frame":
        report = verify_report(inputs, inputs.scene, output.report)
        (inputs.root / report["frames"][0]["path"]).write_bytes(b"tampered")
    elif change == "report":
        (inputs.root / output.report.path).write_text("{}")
    elif change == "video":
        (inputs.root / inputs.scene.artifact.video_path).write_bytes(b"tampered")
    elif change == "code":
        (inputs.root / inputs.scene.code_file).write_text("tampered")
    else:
        inputs = replace(
            inputs, scene=inputs.scene.model_copy(update={"inherited_elements_code": "new"})
        )
    with pytest.raises(ValueError):
        pipeline.scene(inputs, stage[2])
    assert len(calls.evaluators) == 1


def test_unknown_retains_video_and_null_score_and_redacts_secrets(stage, monkeypatch):
    from kd1_anime.config import settings

    monkeypatch.setattr(settings, "VISUAL_LLM_API_KEY", "secret-visual-key")
    inputs, output, _, _ = evaluated_scene(stage, RuntimeError("secret-visual-key unavailable"))
    assert output.scene.rendered and output.scene.artifact is not None
    assert output.scene.visual_status == "unknown" and output.scene.visual_score is None
    assert output.scene.visual_verification.status == "unknown"
    assert "secret-visual-key" not in (inputs.root / output.report.path).read_text()
    final = final_inputs(inputs)
    merged = FinalizationPipeline(ArtifactMerger()).merge(final, stage[2])
    assert (inputs.root / merged.video.path).is_file()


@pytest.mark.parametrize(
    "target,category,expected",
    [
        ("coder", "layout", "coder"),
        ("planner", "mathematics", "planner"),
        ("continuity", "consistency", "continuity"),
        ("unknown", "other", "none"),
    ],
)
def test_routes_and_reserves_one_finite_repair_without_destroying_video(
    stage, target, category, expected
):
    inputs, output, pipeline, calls = evaluated_scene(
        stage, analysis(2, target=target, category=category)
    )
    assert output.repair_route == expected
    assert output.scene.rendered and output.scene.artifact is not None
    assert output.visual_fix_attempts == (0 if expected == "none" else 1)
    if expected != "none":
        replay = pipeline.scene(inputs, stage[2])
        assert replay.repair_route == expected and replay.visual_fix_attempts == 1
        assert len(calls.evaluators) == 1


@pytest.mark.parametrize("relaxed,limit", [(True, 2), (False, 0)])
def test_relaxed_and_exhausted_gates_are_diagnostic_only(stage, relaxed, limit):
    _, output, _, calls = evaluated_scene(
        stage,
        analysis(2, target="coder"),
        GENERATION_MODE="relaxed" if relaxed else "strict",
        MAX_VISUAL_FIX_ATTEMPTS=limit,
    )
    assert output.repair_route == "none" and output.scene.visual_status == "warning"
    assert output.visual_fix_attempts == 0 and not calls.fixes


def test_visual_proposal_is_bound_inspected_and_never_installed(stage):
    inputs, output, pipeline, calls = evaluated_scene(stage, analysis(2, target="coder"))
    original = (inputs.root / inputs.scene.code_file).read_text()
    proposal = pipeline.propose_fix(inputs, stage[2])
    assert proposal.accepted and proposal.repair_route == "review_code"
    assert proposal.base_source.sha256 == inputs.scene.code_sha256
    assert proposal.candidate_source.sha256 != proposal.base_source.sha256
    assert proposal.visual_report == output.report
    assert (inputs.root / inputs.scene.code_file).read_text() == original
    assert calls.fixes[0][0].request_context == (1, "visual_fixing")
    assert calls.fixes[0][0].control is pipeline.control
    assert inputs.scene.rendered


def test_budget_rejection_propagates_without_turning_valid_video_into_failure(stage):
    inputs = configured(rendered(stage))

    def denied(*args):
        raise RequestBudgetExceeded("no dispatch budget")

    pipeline, calls = visual_stage([analysis()], request_context=denied)
    with pytest.raises(RequestBudgetExceeded):
        pipeline.scene(inputs, stage[2])
    assert inputs.scene.rendered and not calls.evaluated


def test_failed_repair_restores_readonly_verified_best_candidate(stage):
    inputs, output, pipeline, calls = evaluated_scene(stage)
    failed = inputs.scene.model_copy(
        update={
            "rendered": False,
            "failed": True,
            "artifact": None,
            "slurm_job": None,
            "visual_fix_attempts": 2,
        }
    )
    restored = pipeline.scene(replace(inputs, scene=failed), stage[2])
    assert restored.restored and restored.scene.rendered and not restored.scene.failed
    assert restored.scene.visual_status == "warning"
    assert restored.scene.artifact == output.scene.artifact
    assert restored.scene.visual_fix_attempts == 2 and len(calls.evaluators) == 1


def test_best_rollback_rejects_missing_candidate_video(stage):
    inputs, _, pipeline, _ = evaluated_scene(stage)
    (inputs.root / inputs.scene.visual_best_candidate.artifact.video_path).unlink()
    failed = inputs.scene.model_copy(update={"rendered": False, "failed": True, "artifact": None})
    with pytest.raises(ValueError):
        pipeline.scene(replace(inputs, scene=failed), stage[2])


def test_boundary_routes_to_referenced_opening_scene_and_persists_attempt(stage):
    inputs, _, _, _ = evaluated_scene(stage)
    final = replace(final_inputs(inputs), scenes=(inputs.scene, second_scene(inputs)))
    pipeline, calls = visual_stage(
        [analysis(2, target="continuity", category="consistency", boundary_ids=("1->2",))]
    )
    output = pipeline.boundary(final, stage[2])
    assert output.scene_id == 2 and output.repair_route == "continuity"
    assert output.visual_fix_attempts == 1 and output.scenes[1].visual_fix_attempts == 1
    assert final.scenes[1].visual_fix_attempts == 0
    report = json.loads((inputs.root / output.report.path).read_text())
    assert len(report["scene_bindings"]) == 2 and report["deterministic_checks"]
    assert report["attempts"]["2"] == 0 and len(calls.evaluators) == 1


def test_boundary_respects_relaxed_diagnostic_policy(stage):
    inputs, _, _, _ = evaluated_scene(stage, GENERATION_MODE="relaxed")
    final = replace(final_inputs(inputs), scenes=(inputs.scene, second_scene(inputs)))
    pipeline, _ = visual_stage([analysis(2, target="coder")])
    output = pipeline.boundary(final, stage[2])
    assert output.repair_route == "none" and output.visual_fix_attempts == 0


def test_final_visual_reports_without_unattributable_rewrite_and_reuses_receipt(stage):
    inputs, _, _, _ = evaluated_scene(stage)
    visual, calls = visual_stage([analysis(1, target="planner", category="mathematics")])
    pipeline = FinalizationPipeline(ArtifactMerger())
    pipeline.visual = visual
    final = final_inputs(inputs)
    output = pipeline.evaluate(final, pipeline.merge(final, stage[2]), stage[2])
    assert output.visual_status == "rejected" and output.scenes_to_improve == ()
    assert output.visual_report is not None
    pipeline.evaluate(final, output, stage[2])
    assert len(calls.evaluated) == 1
    report = json.loads((inputs.root / output.visual_report.path).read_text())
    (inputs.root / report["frames"][0]["path"]).write_bytes(b"stale")
    with pytest.raises(ValueError):
        pipeline.evaluate(final, output, stage[2])


def test_auto_evaluation_has_bounded_rounds_and_uses_snapshot_not_manifest(stage):
    inputs = configured(
        rendered(stage), ENABLE_VISUAL_EVAL=False, ENABLE_AUTO_EVAL=True, MAX_EVAL_ROUNDS=1
    )
    visual, calls = visual_stage([])
    pipeline = FinalizationPipeline(ArtifactMerger())
    pipeline.visual = visual
    final = final_inputs(inputs)
    output = pipeline.evaluate(final, pipeline.merge(final, stage[2]), stage[2])
    assert output.scenes_to_improve == (1,) and output.eval_round == 1
    assert all(not kwargs["enable_visual_eval"] for _, kwargs in calls.evaluators)
    final = replace(final, eval_round=1)
    evaluated = pipeline.evaluate(final, pipeline.merge(final, stage[2]), stage[2])
    assert evaluated.scenes_to_improve == () and evaluated.eval_round == 1


def test_partial_merge_policy_excludes_only_explicitly_incomplete_scenes(stage):
    inputs = rendered(stage)
    final = final_inputs(inputs)
    incomplete = second_scene(inputs).model_copy(update={"rendered": False, "artifact": None})
    final = replace(final, scenes=(inputs.scene, incomplete))
    with pytest.raises(ValueError, match="Partial output"):
        FinalizationPipeline(ArtifactMerger()).merge(final, stage[2])
    output = FinalizationPipeline(ArtifactMerger()).merge(
        replace(final, allow_partial=True), stage[2]
    )
    assert (inputs.root / output.video.path).is_file()


def next_render(inputs):
    from kd1_anime.objects import ObjectStore
    from kd1_anime.run_store import sha256_text

    scene = inputs.scene.model_copy(deep=True)
    code = (inputs.root / scene.code_file).read_text().replace("self.wait(1)", "self.wait(2)")
    scene.code_file = ObjectStore(inputs.root).source(scene.plan.scene_id, code)
    scene.code_sha256 = sha256_text(code)
    relative = (Path(scene.slurm_job.media_dir) / "candidate" / "Pause.mp4").as_posix()
    path = inputs.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"new candidate video")
    metadata = scene.artifact.metadata.model_copy(update={"size_bytes": path.stat().st_size})
    video_hash = sha256_file(path)
    scene.slurm_job = scene.slurm_job.model_copy(
        update={
            "job_id": "2",
            "code_sha256": scene.code_sha256,
            "output_path": relative,
            "output_sha256": video_hash,
            "output_metadata": metadata,
        }
    )
    scene.artifact = scene.artifact.model_copy(
        update={
            "job_id": "2",
            "code_sha256": scene.code_sha256,
            "video_path": relative,
            "video_sha256": video_hash,
            "metadata": metadata,
        }
    )
    scene.visual_status = "pending"
    return replace(inputs, scene=scene)


def test_exhausted_visual_repair_restores_better_rank_and_routes_continuity_rebuild(stage):
    inputs = configured(rendered(stage), MAX_VISUAL_FIX_ATTEMPTS=1)
    pipeline, calls = visual_stage([analysis(3), analysis(2, target="coder")])
    first = pipeline.scene(inputs, stage[2])
    assert first.repair_route == "coder" and first.visual_fix_attempts == 1
    newer = next_render(replace(inputs, scene=first.scene))
    output = pipeline.scene(newer, stage[2])
    assert output.restored and output.repair_route == "rebuild"
    assert output.scene.code_sha256 == inputs.scene.code_sha256
    assert output.scene.artifact == inputs.scene.artifact
    assert output.scene.visual_status == "warning" and output.visual_fix_attempts == 1
    assert len(calls.evaluated) == 2


def test_visual_ranking_prefers_no_major_issue_over_higher_subjective_score(stage):
    inputs = configured(rendered(stage), MAX_VISUAL_FIX_ATTEMPTS=1)
    pipeline, _ = visual_stage([analysis(4, target="coder"), analysis(2)])
    first = pipeline.scene(inputs, stage[2])
    newer = next_render(replace(inputs, scene=first.scene))
    output = pipeline.scene(newer, stage[2])
    assert not output.restored and output.scene.visual_best_candidate.score == 2
    assert output.scene.visual_best_candidate.has_major_issue is False
    assert output.scene.artifact == newer.scene.artifact


def test_boundary_coder_proposal_binds_separate_boundary_receipt(stage):
    inputs, _, _, _ = evaluated_scene(stage)
    final = replace(final_inputs(inputs), scenes=(inputs.scene, second_scene(inputs)))
    pipeline, _ = visual_stage([analysis(2, target="coder")])
    boundary = pipeline.boundary(final, stage[2])
    selected = replace(inputs, scene=boundary.scenes[1])
    proposal = pipeline.propose_fix(
        selected, stage[2], boundary_report=boundary.report, feedback=boundary.feedback
    )
    assert proposal.visual_report == boundary.report and proposal.visual_fix_attempts == 1
    assert proposal.accepted and proposal.repair_route == "review_code"


def test_production_visual_dispatch_explicitly_disables_stream(stage):
    inputs = configured(rendered(stage))
    calls = []

    def completion(*args, **kwargs):
        calls.append(kwargs)
        schema = args[2]
        result = analysis().to_dict()
        return schema.model_validate(
            {
                "overall_analysis": result["overall_analysis"],
                "evaluation": result["evaluation"],
                "issues": [],
            }
        )

    with inputs.config.activate():
        evaluator = VisualEvaluator()
        evaluator._agent = SimpleNamespace(call_llm_json=completion)
        evaluator.evaluate_video_frames([sample(inputs.root / "frames" / "one.png")], "pause")
    assert calls[0]["stream"] is False


def test_explicit_persisted_visual_profile_wins_over_config_recomputation(stage):
    inputs = configured(rendered(stage), ENABLE_VISUAL_EVAL=False, VISUAL_EVAL_THRESHOLD=1)
    pinned = VisualEvalProfile(
        enabled=True,
        model="persisted-vision-model",
        threshold=4.5,
        repair_enabled=False,
        frame_count=2,
    )
    inputs = replace(inputs, visual_eval_profile=pinned)
    pipeline, calls = visual_stage([analysis(3)])
    output = pipeline.scene(inputs, stage[2])
    assert output.scene.visual_status == "warning" and output.repair_route == "none"
    assert calls.evaluators[0][1]["visual_eval_model"] == "persisted-vision-model"
    report = json.loads((inputs.root / output.report.path).read_text())
    assert report["visual_profile_sha256"] == pinned.digest()


def test_stale_final_report_hash_is_rejected_before_new_provider_dispatch(stage):
    inputs, _, _, _ = evaluated_scene(stage)
    visual, calls = visual_stage([analysis()])
    pipeline = FinalizationPipeline(ArtifactMerger())
    pipeline.visual = visual
    final = final_inputs(inputs)
    output = pipeline.evaluate(final, pipeline.merge(final, stage[2]), stage[2])
    (inputs.root / output.visual_report.path).write_text("{}")
    with pytest.raises(ValueError):
        pipeline.evaluate(final, output, stage[2])
    assert len(calls.evaluated) == 1


def test_report_or_frame_changes_during_fix_are_rejected_before_proposal_publication(stage):
    inputs, _, pipeline, _ = evaluated_scene(stage, analysis(2, target="coder"))

    class TamperingFixer:
        def fix(self, original, feedback, **kwargs):
            report = json.loads((inputs.root / inputs.scene.visual_report_file).read_text())
            (inputs.root / report["frames"][0]["path"]).write_bytes(b"tampered during model call")
            return original.replace("self.wait(1)", "self.wait(2)")

    pipeline.fixer_factory = TamperingFixer
    with pytest.raises(ValueError):
        pipeline.propose_fix(inputs, stage[2])
    assert inputs.scene.rendered
