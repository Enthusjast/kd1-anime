import json

import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary
from test_pipeline_visual import analysis, visual_stage

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.config import settings
from kd1_anime.rendering import sha256_file
from kd1_anime.run_store import RunRepository


@pytest.mark.parametrize("unknown", [False, True])
def test_public_generate_commits_scene_boundary_and_final_visual_evidence(
    agent_boundary, monkeypatch, unknown
):
    boundary, dependencies = agent_boundary
    monkeypatch.setattr(settings, "ENABLE_VISUAL_EVAL", True)
    monkeypatch.setattr(settings, "VISUAL_LLM_MODEL", "independent-vision")
    outcomes = (
        [ConnectionError("visual endpoint unavailable")] * 3 if unknown else [analysis(4)] * 3
    )
    visual, calls = visual_stage(outcomes)
    dependencies.evaluator_factory = visual.evaluator_factory
    runtime = AgentRuntime(dependencies=dependencies)
    output = runtime.generate("Show a pause", approve_plan=True)
    manifest = RunRepository(boundary.workspace).load(runtime.manifest.run_id)
    assert manifest.status == "completed" and output.is_file()
    scene = manifest.scenes[1]
    assert scene.rendered and scene.artifact is not None
    assert scene.visual_status == ("unknown" if unknown else "passed")
    assert scene.visual_score is None if unknown else scene.visual_score == 4
    assert scene.visual_verification.artifact_sha256 == scene.artifact.video_sha256
    report = json.loads((runtime.root / scene.visual_report_file).read_text())
    assert report["binding"]["code_sha256"] == scene.code_sha256
    assert report["binding"]["video_sha256"] == scene.artifact.video_sha256
    assert sha256_file(runtime.root / scene.visual_report_file) == scene.visual_report_sha256
    assert all(
        sha256_file(runtime.root / frame["path"]) == frame["sha256"] for frame in report["frames"]
    )
    assert manifest.agent.final_visual_report is not None
    final = json.loads((runtime.root / manifest.agent.final_visual_report.path).read_text())
    assert final["video_sha256"] == manifest.final_video_sha256
    assert final["result"] is None if unknown else final["result"]["overall_score"] == 4
    actions = [item.action for item in manifest.agent.invocations]
    assert (
        actions.index("evaluate_visual")
        < actions.index("evaluate_boundary")
        < actions.index("merge_video")
    )
    before = len(calls.evaluated)
    assert AgentRuntime(dependencies=dependencies).resume(manifest.run_id) == output
    assert len(calls.evaluated) == before and len(boundary.backend.sources) == 1
