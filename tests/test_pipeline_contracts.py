"""Domain and service regressions formerly coupled to private workflow methods."""

from dataclasses import replace

import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary
from test_execution_contracts import plan
from test_runtime_reliability import completed_run

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.agents.api_linter import lint_manim_api, repair_manim_api_compatibility
from kd1_anime.agents.failure_router import classify_failure
from kd1_anime.agents.validator import validate_manim_code
from kd1_anime.application import Application
from kd1_anime.candidate_acceptor import CandidateAcceptor
from kd1_anime.config import settings
from kd1_anime.domain.planning import LessonSpec, TeachingGraph, VisualElementState
from kd1_anime.domain.runtime import ExecutionSpec, RunConfig
from kd1_anime.domain.state import RunPaths
from kd1_anime.eval.visual_eval import VisualAnalysisResult, VisualIssue
from kd1_anime.exceptions import ValidationError
from kd1_anime.objects import ObjectStore
from kd1_anime.pipeline.contracts import SceneInputs
from kd1_anime.rendering import RenderProfile
from kd1_anime.run_store import RunRepository, StoredSceneState, sha256_text
from kd1_anime.services.continuity_state import prepare_inherited_inputs
from kd1_anime.services.generation import GenerationService
from kd1_anime.services.policies import visual_repair_target

EXPORT_SOURCE = """from manim import *
class Demo(Scene):
    def construct(self):
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: kept
        kept = Circle()
        # element_id: other
        other = Square()
        # KD1_CONTINUITY_EXPORT_END
        self.add(kept, other)
"""


def inherited_inputs(tmp_path, monkeypatch, *, mode="minimal", independent=False):
    monkeypatch.setattr(settings, "CONTINUITY_CONTEXT_MODE", mode)
    kept = VisualElementState(element_id="kept", variable_name="kept")
    other = VisualElementState(element_id="other", variable_name="other")
    previous = StoredSceneState(
        plan=plan().model_copy(update={"new_elements": [kept, other]}),
        code_file=ObjectStore(tmp_path).source(1, EXPORT_SOURCE),
        code_sha256=sha256_text(EXPORT_SOURCE),
        class_name="Demo",
        reviewed=True,
    )
    current = StoredSceneState(
        plan=plan(2, dependencies_complete=independent).model_copy(
            update={"inherited_elements": [] if independent else [kept]}
        )
    )
    profile = RenderProfile.current()
    inputs = SceneInputs(
        run_id="20261029-120000-1234abcd",
        root=tmp_path,
        config=RunConfig.capture(),
        render_profile=profile,
        execution_spec=ExecutionSpec.capture(profile),
        scene=current,
        plans=(previous.plan, current.plan),
        lesson_spec=LessonSpec(),
        teaching_graph=TeachingGraph(),
        continuity_bible=None,
    )
    return inputs, previous


@pytest.mark.parametrize("mode", ["minimal", "full"])
def test_continuity_context_contains_only_requested_exports_unless_full(
    tmp_path, monkeypatch, mode
):
    inputs, previous = inherited_inputs(tmp_path, monkeypatch, mode=mode)
    before = previous.model_dump_json()
    result = prepare_inherited_inputs(inputs, (previous,))
    assert "kept = Circle()" in result.scene.inherited_elements_code
    assert ("other = Square()" in result.scene.inherited_elements_code) is (mode == "full")
    assert inputs.scene.inherited_elements_code == ""
    assert previous.model_dump_json() == before


def test_explicitly_independent_scene_does_not_inspect_or_inherit_unrelated_source(
    tmp_path, monkeypatch
):
    inputs, previous = inherited_inputs(tmp_path, monkeypatch, independent=True)
    (tmp_path / previous.code_file).write_text("unrelated source was removed")
    result = prepare_inherited_inputs(inputs, (previous,))
    assert result.scene.inherited_elements_code == ""
    assert result.element_manifest.entries == []
    assert result.state_ledger.elements == []


def test_inheritance_cannot_resurrect_removed_elements(tmp_path, monkeypatch):
    inputs, previous = inherited_inputs(tmp_path, monkeypatch)
    removed = StoredSceneState(
        plan=plan(2).model_copy(
            update={
                "elements_to_remove": [VisualElementState(element_id="kept", variable_name="kept")]
            }
        )
    )
    current = inputs.scene.model_copy(
        update={"plan": inputs.scene.plan.model_copy(update={"scene_id": 3})}
    )
    inputs = replace(inputs, scene=current, plans=(previous.plan, removed.plan, current.plan))
    with pytest.raises(ValueError, match="已被先行场景移除"):
        prepare_inherited_inputs(inputs, (previous, removed))


def test_generation_validates_continuity_before_accepting_source():
    scene_plan = plan().model_copy(
        update={
            "new_elements": [
                VisualElementState(element_id="kept", variable_name="kept"),
                VisualElementState(element_id="other", variable_name="other"),
            ]
        }
    )
    invalid = EXPORT_SOURCE.replace("# element_id: other", "# element_id: kept")
    calls = []

    class Coder:
        def generate_code(self, scene_plan, feedback="", **kwargs):
            calls.append(feedback)
            return invalid if len(calls) == 1 else EXPORT_SOURCE

    generated = GenerationService(
        agent=Coder(),
        validator=validate_manim_code,
        candidate_acceptor=CandidateAcceptor(),
        generation_mode="strict",
        limits={"CODE_VALIDATION_ATTEMPTS": 3},
    ).generate(scene_plan)
    assert generated.code == EXPORT_SOURCE and generated.class_name == "Demo"
    assert generated.attempts == len(calls) == 2
    assert "连续性" in calls[1]


@pytest.mark.parametrize("mode", ["strict", "relaxed"])
def test_generation_obeys_validation_attempt_limit_without_hidden_fallback(mode):
    calls = []
    reservations = []

    class InvalidCoder:
        def generate_code(self, *args, **kwargs):
            calls.append(1)
            return "not valid Python !!!"

    def reserve():
        reservations.append(1)
        return len(reservations)

    with pytest.raises(ValidationError):
        GenerationService(
            agent=InvalidCoder(),
            validator=validate_manim_code,
            candidate_acceptor=CandidateAcceptor(),
            generation_mode=mode,
            limits={"CODE_VALIDATION_ATTEMPTS": 2},
        ).generate(plan(), reserve_attempt=reserve)
    assert len(calls) == len(reservations) == 2


def test_current_manifest_rejects_stale_technical_contract_without_mutation(agent_boundary):
    _, manifest = completed_run(agent_boundary)
    manifest.scenes[1].technical_contract_stale = True
    before = manifest.model_dump_json()
    with pytest.raises(ValueError, match=r"TechnicalSpec|技术合同|stale"):
        manifest.validate_for_resume()
    assert manifest.model_dump_json() == before


def test_public_run_directories_and_records_are_private_and_redacted(agent_boundary):
    root, manifest = completed_run(agent_boundary)
    assert root.stat().st_mode & 0o777 == 0o700
    assert (root / "manifest.json").stat().st_mode & 0o777 == 0o600
    assert (root / manifest.scenes[1].code_file).stat().st_mode & 0o777 == 0o600
    for path in (*root.rglob("*.json"), *root.rglob("*.jsonl")):
        assert path.stat().st_mode & 0o777 == 0o600
        assert "application-test-key" not in path.read_text()


def test_failure_router_prioritizes_infrastructure_over_code_rewrite():
    route = classify_failure(
        "FileNotFoundError: No such file or directory: 'xelatex'",
        phase="render",
    )
    assert route.category == "infrastructure"
    assert route.handler == "infra_retry"
    assert classify_failure("render boom", phase="render").category == "render"


def test_failure_router_separates_plan_math_from_runtime_api_errors():
    assert classify_failure("数学断言不等价", phase="review").handler == "plan_review"
    assert (
        classify_failure("AttributeError: OpenGLCamera has no frame", phase="render").handler
        == "code_patch"
    )
    assert (
        classify_failure(
            "ValueError: operands could not be broadcast together with shapes (0,) (3,)\n"
            "Flash.create_lines",
            phase="render",
        ).category
        == "lifecycle"
    )


def test_api_linter_rejects_deprecated_manim_api():
    result = lint_manim_api(
        "from manim import *\nclass Demo(Scene):\n"
        "    def construct(self): self.play(ShowCreation(Circle()))"
    )
    assert result.is_valid is False
    assert any("ShowCreation" in error for error in result.errors)


def test_api_linter_warns_about_unbounded_graph_and_updater():
    result = lint_manim_api(
        "from manim import *\nclass Demo(Scene):\n"
        "    def construct(self):\n"
        "        axes = Axes()\n"
        "        graph = axes.plot(lambda x: x**2)\n"
        "        dot = always_redraw(lambda: Dot())\n"
    )
    assert result.is_valid is True
    assert any("x_range" in warning for warning in result.warnings)
    assert any("clear_updaters" in warning for warning in result.warnings)


def test_api_repair_replaces_grow_arrow_for_arrow3d():
    code = (
        "from manim import *\n"
        "class Demo(ThreeDScene):\n"
        "    def construct(self):\n"
        "        arrow = Arrow3D(ORIGIN, RIGHT)\n"
        "        self.play(GrowArrow(arrow))\n"
    )

    repaired, repairs = repair_manim_api_compatibility(code)

    assert "self.play(Create(arrow))" in repaired
    assert repairs == ("将 Arrow3D 的 GrowArrow 替换为 Create: arrow",)


def test_api_repair_leaves_grow_arrow_for_arrow_unchanged():
    code = (
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        arrow = Arrow(ORIGIN, RIGHT)\n"
        "        self.play(GrowArrow(arrow))\n"
    )

    repaired, repairs = repair_manim_api_compatibility(code)

    assert repaired == code
    assert repairs == ()


def test_visual_math_issue_routes_back_to_planner():
    result = VisualAnalysisResult(
        overall_analysis="公式不正确",
        mathematical_accuracy={"score": 2, "comprehensive_evaluation": "公式错误"},
        visual_relevance={"score": 4, "comprehensive_evaluation": "相关"},
        visual_quality={"score": 4, "comprehensive_evaluation": "清晰"},
        visual_consistency={"score": 4, "comprehensive_evaluation": "一致"},
        element_layout={"score": 4, "comprehensive_evaluation": "整齐"},
        issues=[
            VisualIssue(
                category="mathematics",
                severity="major",
                frame_ids=["F01"],
                evidence="画面等式前后不等价",
                recommendation="回到计划修正数学断言",
            )
        ],
    )

    assert visual_repair_target(result) == "planner"


def test_visual_boundary_issue_uses_boundary_start_scene_for_repair():
    result = VisualAnalysisResult(
        overall_analysis="边界突变",
        mathematical_accuracy={"score": 4, "comprehensive_evaluation": "正确"},
        visual_relevance={"score": 4, "comprehensive_evaluation": "相关"},
        visual_quality={"score": 4, "comprehensive_evaluation": "清晰"},
        visual_consistency={"score": 2, "comprehensive_evaluation": "突变"},
        element_layout={"score": 4, "comprehensive_evaluation": "整齐"},
        issues=[
            VisualIssue(
                category="consistency",
                severity="major",
                repair_target="continuity",
                frame_ids=["F01", "F02"],
                boundary_ids=["B02"],
                evidence="Scene 2 开头丢失对象",
                recommendation="恢复边界交接",
            )
        ],
    )

    assert visual_repair_target(result) == "continuity"


def test_run_paths_are_unique(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    first = RunPaths.create()
    second = RunPaths.create()
    assert first.root != second.root


def test_code_generation_does_not_treat_unmarked_internal_objects_as_exports():
    from kd1_anime.agents.validator import validate_manim_code
    from kd1_anime.candidate_acceptor import CandidateAcceptor
    from kd1_anime.services.generation import GenerationService

    scene_plan = plan().model_copy(
        update={
            "new_elements": [
                VisualElementState(
                    element_id="temporary_formula",
                    variable_name="temporary_formula",
                    required=False,
                )
            ]
        }
    )
    code = """from manim import *
class Demo(Scene):
    def construct(self):
        temporary_formula = Circle()
        self.play(Create(temporary_formula))
"""

    class FakeCoder:
        calls = 0

        def generate_code(self, scene_plan, feedback="", **kwargs):
            self.calls += 1
            return code

    coder = FakeCoder()
    generated = GenerationService(
        agent=coder,
        validator=validate_manim_code,
        candidate_acceptor=CandidateAcceptor(),
        generation_mode="strict",
    ).generate(scene_plan)

    assert generated.code == code
    assert generated.class_name == "Demo"
    assert coder.calls == 1


def test_resume_rejects_tampered_source_before_render_submission(agent_boundary):
    boundary, dependencies = agent_boundary
    app = Application(runtime_factory=lambda: AgentRuntime(dependencies=dependencies))
    interrupted = []

    def stop_before_submission(manifest):
        if "submit_render" in manifest.agent.scene_next_actions.values() and not interrupted:
            interrupted.append(True)
            raise OSError("interrupted before dispatch")

    dependencies.after_commit = stop_before_submission
    with pytest.raises(OSError, match="before dispatch"):
        app.generate("Show a one-second pause", approve_plan=True)
    dependencies.after_commit = None
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load_for_resume(root.name)
    source = root / manifest.scenes[1].code_file
    source.write_text("tampered source")
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="source hash mismatch"):
        app.resume(root.name)
    assert not boundary.backend.sources
    assert source.read_text() == "tampered source"
    assert (root / "manifest.json").read_bytes() == before
