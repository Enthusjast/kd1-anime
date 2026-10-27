from types import SimpleNamespace

import pytest

from kd1_anime.domain.planning import LessonSpec, ScenePlan, TeachingGraph
from kd1_anime.domain.review import ReviewResult
from kd1_anime.domain.runtime import ExecutionSpec, RunConfig
from kd1_anime.pipeline.contracts import SceneInputs
from kd1_anime.pipeline.scene import ScenePipeline
from kd1_anime.rendering import RenderProfile
from kd1_anime.request_budget import RequestContext
from kd1_anime.run_store import StoredSceneState

SOURCE = (
    "from manim import *\nclass Pause(Scene):\n    def construct(self):\n        self.wait(1)\n"
)


def inputs(tmp_path):
    plan = ScenePlan(
        scene_id=1,
        title="Pause",
        duration_seconds=1,
        purpose="Wait",
        math_concept="Time",
        visual_design="Empty",
        camera_movement="Fixed",
        visual_flow=["Wait"],
        key_moments=["One second"],
        computation="None",
    )
    profile = RenderProfile(
        renderer="cairo",
        quality="l",
        pixel_width=854,
        pixel_height=480,
        frame_rate=15,
        manim_version="0.20.1",
    )
    return SceneInputs(
        "20260101-000000-1234abcd",
        tmp_path,
        RunConfig.capture(),
        profile,
        ExecutionSpec.capture(profile),
        StoredSceneState(plan=plan),
        (plan,),
        LessonSpec(),
        TeachingGraph(),
        None,
    )


def pipeline(reject=False):
    class Coder:
        def generate_code(self, plan, **kwargs):
            return SOURCE

    class Reviewer:
        def review(self, code, plan, **kwargs):
            return ReviewResult(
                is_valid=not reject, feedback="The requested scene is missing" if reject else ""
            )

    return ScenePipeline(
        Coder,
        Reviewer,
        request_context=lambda scene, stage: RequestContext(lambda *a, **k: None, lambda *a: None),
        control=SimpleNamespace(checkpoint=lambda: None),
    )


def test_source_and_reports_are_immutable_and_hash_bound(tmp_path):
    stage = pipeline()
    scene = inputs(tmp_path)
    generated = stage.generate(scene)
    assert (tmp_path / generated.source.path).read_text() == SOURCE
    scene.scene.code_file = generated.source.path
    scene.scene.code_sha256 = generated.source.sha256
    scene.scene.class_name = generated.class_name
    inspected = stage.inspect(scene)
    assert inspected.accepted and inspected.verification.code_sha256 == generated.source.sha256
    assert (tmp_path / inspected.report.path).is_file()
    reviewed = stage.review(scene)
    assert reviewed.accepted and reviewed.code_sha256 == generated.source.sha256
    assert (tmp_path / reviewed.report.path).is_file()


def test_rejected_review_is_evidence_not_automatic_fix(tmp_path):
    stage = pipeline(reject=True)
    scene = inputs(tmp_path)
    output = stage.generate(scene)
    scene.scene.code_file = output.source.path
    scene.scene.code_sha256 = output.source.sha256
    scene.scene.class_name = output.class_name
    assert not stage.review(scene).accepted


def test_tampered_source_is_rejected_before_review(tmp_path):
    stage = pipeline()
    scene = inputs(tmp_path)
    output = stage.generate(scene)
    scene.scene.code_file = output.source.path
    scene.scene.code_sha256 = output.source.sha256
    scene.scene.class_name = output.class_name
    (tmp_path / output.source.path).write_text("tampered")
    with pytest.raises(ValueError, match="identity"):
        stage.review(scene)
