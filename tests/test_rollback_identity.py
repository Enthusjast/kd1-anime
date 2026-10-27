from types import SimpleNamespace

from kd1_anime.domain.candidates import CodeCandidate
from kd1_anime.domain.planning import LessonSpec, ScenePlan, TeachingGraph, VisualElementState
from kd1_anime.domain.review import ReviewResult
from kd1_anime.domain.runtime import ExecutionSpec, RunConfig
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.contracts import SceneInputs
from kd1_anime.pipeline.scene import ScenePipeline
from kd1_anime.rendering import RenderProfile
from kd1_anime.run_store import StoredSceneState, sha256_text


def test_rollback_reviews_exact_historical_bytes_without_repairing_execution_identity(tmp_path):
    plan = ScenePlan(
        scene_id=1,
        title="Circle",
        duration_seconds=1,
        purpose="Preserve an optional object",
        math_concept="Circle",
        visual_design="Dark",
        camera_movement="Fixed",
        visual_flow=["Show circle"],
        key_moments=["Pause"],
        computation="radius=1",
        new_elements=[
            VisualElementState(element_id="circle", variable_name="circle", required=False)
        ],
    )
    current = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait(1)\n"
    historical = (
        "from manim import *\nclass Demo(Scene):\n    def construct(self):\n"
        "        circle = Circle()\n"
        "        self.play(Create(circle))\n"
        "        self.play(FadeOut(circle))\n"
        "        # KD1_CONTINUITY_EXPORT_BEGIN\n"
        "        # element_id: circle\n"
        "        circle = Circle()\n"
        "        # KD1_CONTINUITY_EXPORT_END\n"
    )
    objects = ObjectStore(tmp_path)
    current_path = objects.source(1, current)
    candidate_path = objects.source(1, historical)
    candidate = CodeCandidate(
        code_file=candidate_path,
        code_sha256=sha256_text(historical),
        class_name="Demo",
        verification="smoke",
        inherited_elements_sha256=sha256_text(""),
    )
    scene = StoredSceneState(
        plan=plan,
        code_file=current_path,
        code_sha256=sha256_text(current),
        class_name="Demo",
        candidates=[candidate],
    )
    profile = RenderProfile.current()
    inputs = SceneInputs(
        "20261027-120000-1234abcd",
        tmp_path,
        RunConfig.capture(),
        profile,
        ExecutionSpec.capture(profile),
        scene,
        (plan,),
        LessonSpec(),
        TeachingGraph(),
        None,
    )
    reviewed = []

    class Reviewer:
        def review(self, code, plan, **kwargs):
            reviewed.append(code)
            return ReviewResult(is_valid=True)

    pipeline = ScenePipeline(
        lambda: None,
        Reviewer,
        request_context=lambda scene_id, stage: None,
        control=SimpleNamespace(check=lambda: None),
    )
    result = pipeline.rollback(inputs)
    assert result.accepted
    assert result.source.sha256 == candidate.code_sha256 == sha256_text(historical)
    assert (tmp_path / result.source.path).read_text() == historical
    assert reviewed == [historical]
    assert scene.code_file == current_path and scene.candidates[0].verification == "smoke"
