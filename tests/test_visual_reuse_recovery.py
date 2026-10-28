"""Visual rollback retains exact historical typed authority, not current aliases."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_pipeline_incremental import SOURCE, context
from test_pipeline_incremental import incremental as incremental
from test_pipeline_render import rendered
from test_pipeline_render import stage as stage
from test_pipeline_visual import analysis, configured, visual_stage

from kd1_anime.agent_runtime.contracts import ObjectRef
from kd1_anime.domain.runtime import VisualEvalProfile
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.incremental import IncrementalPipeline, validate_reused_artifact
from kd1_anime.pipeline.visual import VisualSceneOutput, store_report, verify_report
from kd1_anime.rendering import sha256_file
from kd1_anime.run_store import sha256_text
from kd1_anime.services.dependencies import scene_work_digest
from kd1_anime.verification import ExecutionVerification, StaticVerification


def _with_work_digest(inputs, scene):
    scene.input_sha256 = scene_work_digest(scene, inputs.context_sha256)
    return replace(inputs, scene=scene)


@pytest.fixture
def history(incremental):
    """Create genuine A/B reuse receipts, then fail a third source attempt."""
    current = replace(incremental.current, context_sha256="d" * 64)
    current = _with_work_digest(current, current.scene.model_copy(deep=True))
    first_inputs = replace(incremental, current=current, current_scenes=(current.scene,))
    first = IncrementalPipeline().reuse(first_inputs, context())
    visual_inputs = replace(
        current,
        scene=first.scene,
        reuse_provenance=first.provenance,
        visual_eval_profile=VisualEvalProfile(enabled=True, threshold=4, max_fix_attempts=1),
    )
    pipeline, calls = visual_stage([analysis(3, target="coder")])
    evaluated = pipeline.scene(visual_inputs, context())

    # A later code attempt obtains a different, fully validated reuse receipt B.
    later_source = SOURCE.replace("self.wait()", "self.wait(2)")
    later_hash = sha256_text(later_source)
    base = incremental.base.model_copy(deep=True)
    previous = base.scenes[1]
    video = incremental.base_root / "videos/scene_1/attempt_later/Demo.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"later verified video bytes")
    relative = video.relative_to(incremental.base_root).as_posix()
    metadata = previous.artifact.metadata.model_copy(update={"size_bytes": video.stat().st_size})
    previous.code_file = ObjectStore(incremental.base_root).source(1, later_source)
    previous.code_sha256 = later_hash
    previous.artifact = previous.artifact.model_copy(
        update={
            "code_sha256": later_hash,
            "video_path": relative,
            "video_sha256": sha256_file(video),
            "metadata": metadata,
        }
    )
    previous.slurm_job = previous.slurm_job.model_copy(
        update={
            "code_sha256": later_hash,
            "media_dir": video.parent.relative_to(incremental.base_root).as_posix(),
            "output_path": relative,
            "output_sha256": previous.artifact.video_sha256,
            "output_metadata": metadata,
        }
    )
    previous.static_verification = StaticVerification(status="passed", code_sha256=later_hash)
    previous.execution_verification = ExecutionVerification(
        status="passed",
        scope="formal_video",
        code_sha256=later_hash,
        artifact_sha256=previous.artifact.video_sha256,
        duration_seconds=metadata.duration_seconds,
    )
    later_scene = current.scene.model_copy(
        deep=True,
        update={
            "revision": current.scene.revision + 3,
            "code_file": ObjectStore(current.root).source(1, later_source),
            "code_sha256": later_hash,
            "static_verification": StaticVerification(status="passed", code_sha256=later_hash),
        },
    )
    later = _with_work_digest(current, later_scene)
    second = IncrementalPipeline().reuse(
        replace(first_inputs, base=base, current=later, current_scenes=(later.scene,)),
        context(),
    )
    failed = second.scene.model_copy(
        deep=True,
        update={
            "revision": second.scene.revision + 2,
            "visual_best_candidate": evaluated.scene.visual_best_candidate,
            "visual_fix_attempts": 1,
            "rendered": False,
            "failed": True,
            "reviewed": False,
            "artifact": None,
            "slurm_job": None,
        },
    )
    failed_source = later_source.replace("self.wait(2)", "self.wait(3)")
    failed.code_file = ObjectStore(current.root).source(1, failed_source)
    failed.code_sha256 = sha256_text(failed_source)
    failed.static_verification = StaticVerification(status="failed", code_sha256=failed.code_sha256)
    failed.execution_verification = ExecutionVerification()
    failed_inputs = _with_work_digest(
        replace(visual_inputs, reuse_provenance=second.provenance),
        failed,
    )
    return SimpleNamespace(
        inputs=failed_inputs,
        first=first,
        second=second,
        evaluated=evaluated,
        pipeline=pipeline,
        calls=calls,
    )


@pytest.mark.parametrize("entrypoint", ["scene", "reject_proposal"])
def test_best_restore_recovers_old_receipt_across_new_attempts(history, entrypoint, monkeypatch):
    inputs = history.inputs
    before = inputs.scene.model_dump_json()
    inspected = []
    inspect = history.pipeline.acceptor.inspect_candidate

    def fresh_inspection(code, plan, **kwargs):
        inspected.append((sha256_text(code), kwargs))
        return inspect(code, plan, **kwargs)

    monkeypatch.setattr(history.pipeline.acceptor, "inspect_candidate", fresh_inspection)
    output = getattr(history.pipeline, entrypoint)(inputs, context())
    assert output.restored and output.repair_route == "rebuild"
    assert output.reuse_provenance == history.first.provenance
    assert output.reuse_provenance != history.second.provenance == inputs.reuse_provenance
    assert output.scene.artifact == history.first.scene.artifact
    assert output.scene.code_sha256 == history.first.scene.code_sha256
    assert output.scene.revision == inputs.scene.revision > history.first.scene.revision
    assert output.scene.input_sha256 == scene_work_digest(output.scene, inputs.context_sha256)
    assert output.scene.input_sha256 == history.first.scene.input_sha256
    assert output.scene.input_sha256 != inputs.scene.input_sha256
    assert output.scene.static_verification.code_sha256 == output.scene.code_sha256
    assert output.scene.execution_verification.artifact_sha256 == output.scene.artifact.video_sha256
    assert inspected[0][0] == history.first.scene.code_sha256
    assert inspected[0][1]["allow_repair"] is False
    assert len(inspected) == len(history.calls.evaluated) == 1
    assert inputs.scene.model_dump_json() == before
    validate_reused_artifact(
        inputs.root,
        inputs.run_id,
        output.scene,
        inputs.render_profile,
        provenance=output.reuse_provenance,
    )
    with pytest.raises(ValueError):
        validate_reused_artifact(
            inputs.root,
            inputs.run_id,
            output.scene,
            inputs.render_profile,
            provenance=history.second.provenance,
        )
    assert VisualSceneOutput.model_validate_json(output.model_dump_json()) == output


def test_reused_visual_report_persists_exact_typed_reference(history):
    inputs = history.inputs
    report = verify_report(inputs, history.evaluated.scene, history.evaluated.report)
    assert ObjectRef.model_validate(report["reuse_provenance"]) == history.first.provenance


@pytest.mark.parametrize("corruption", ["missing", "null", "stale", "kind", "hash", "alias"])
def test_best_restore_rejects_missing_stale_or_malformed_stored_reference(history, corruption):
    inputs = history.inputs
    candidate = inputs.scene.visual_best_candidate
    report = verify_report(inputs, history.evaluated.scene, history.evaluated.report)
    if corruption == "missing":
        del report["reuse_provenance"]
    elif corruption == "null":
        report["reuse_provenance"] = None
    elif corruption == "stale":
        report["reuse_provenance"] = history.second.provenance.model_dump(mode="json")
    elif corruption == "kind":
        report["reuse_provenance"]["kind"] = "video"
    elif corruption == "hash":
        report["reuse_provenance"]["sha256"] = "0" * 64
    else:
        alias = inputs.root / "reuse-provenance.json"
        alias.write_bytes((inputs.root / history.first.provenance.path).read_bytes())
        report["reuse_provenance"]["path"] = alias.name
    reference = store_report(inputs, report)
    candidate.report_file, candidate.report_sha256 = reference.path, reference.sha256
    with pytest.raises(ValueError):
        history.pipeline.scene(inputs, context())


def test_best_restore_does_not_require_current_receipt_map_entry(history):
    output = history.pipeline.scene(replace(history.inputs, reuse_provenance=None), context())
    assert output.restored and output.reuse_provenance == history.first.provenance


def test_best_restore_rejects_fresh_inspection_failure(history, monkeypatch):
    monkeypatch.setattr(
        history.pipeline.acceptor,
        "inspect_candidate",
        lambda *args, **kwargs: SimpleNamespace(accepted=False),
    )
    with pytest.raises(ValueError, match="readonly candidate inspection"):
        history.pipeline.scene(history.inputs, context())


@pytest.mark.parametrize(
    "corruption", ["report", "provenance", "missing_provenance", "frame", "video", "source"]
)
def test_best_restore_revalidates_all_immutable_evidence(history, corruption):
    inputs = history.inputs
    candidate = inputs.scene.visual_best_candidate
    if corruption == "report":
        path = inputs.root / candidate.report_file
    elif corruption in {"provenance", "missing_provenance"}:
        path = inputs.root / history.first.provenance.path
    elif corruption == "frame":
        report = verify_report(inputs, history.evaluated.scene, history.evaluated.report)
        path = inputs.root / report["frames"][0]["path"]
    elif corruption == "video":
        path = inputs.root / candidate.artifact.video_path
    else:
        path = inputs.root / candidate.code_file
    if corruption == "missing_provenance":
        path.unlink()
    else:
        path.write_bytes(b"tampered evidence")
    with pytest.raises(ValueError):
        history.pipeline.scene(inputs, context())


@pytest.mark.parametrize("changed", ["context", "plan", "inherited", "technical"])
def test_best_restore_does_not_rebind_historical_authority_to_changed_semantics(history, changed):
    inputs = history.inputs
    if changed == "context":
        inputs = replace(inputs, context_sha256="e" * 64)
    elif changed == "plan":
        inputs.scene.plan = inputs.scene.plan.model_copy(update={"title": "Different lesson"})
    elif changed == "inherited":
        inputs.scene.inherited_elements_code = "# different inherited context"
    else:
        from kd1_anime.domain.technical import TechnicalSpec

        inputs.scene.technical_spec = TechnicalSpec(scene_id=1)
        inputs.scene.technical_spec_sha256 = sha256_text(
            inputs.scene.technical_spec.model_dump_json()
        )
    with pytest.raises(ValueError):
        history.pipeline.scene(inputs, context())


def test_normal_rendered_visual_output_drops_stale_reuse_authority(stage):
    inputs = configured(rendered(stage))
    stale = ObjectRef(
        path="artifacts/objects/" + "0" * 64 + ".json", sha256="0" * 64, kind="report"
    )
    inputs = replace(inputs, reuse_provenance=stale)
    pipeline, _ = visual_stage([analysis()])
    output = pipeline.scene(inputs, stage[2])
    report = json.loads((inputs.root / output.report.path).read_text())
    assert output.reuse_provenance is None and report["reuse_provenance"] is None
    failed = output.scene.model_copy(update={"rendered": False, "failed": True, "artifact": None})
    restored = pipeline.scene(replace(inputs, scene=failed), stage[2])
    assert restored.restored and restored.reuse_provenance is None
