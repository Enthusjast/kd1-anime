from pathlib import Path

import pytest

from kd1_anime.agents.planner import ScenePlan, VisualElementState
from kd1_anime.agents.technical_planner import TechnicalObject, TechnicalSpec
from kd1_anime.candidate_acceptor import CandidateAcceptor, CandidateRejected


def make_plan() -> ScenePlan:
    return ScenePlan(
        scene_id=1,
        title="测试",
        duration_seconds=1,
        purpose="验证候选入口",
        math_concept="等待",
        visual_design="简洁",
        camera_movement="固定",
        visual_flow=["等待"],
        key_moments=["等待"],
        computation="无",
    )


CODE = "from manim import *\nclass Demo(Scene):\n    def construct(self):\n        self.wait()\n"


def test_candidate_acceptor_returns_hash_and_can_write_atomically(tmp_path: Path):
    destination = tmp_path / "scene.py"
    accepted = CandidateAcceptor().accept(CODE, make_plan(), destination=destination)

    assert accepted.class_name == "Demo"
    assert accepted.code_sha256
    assert destination.read_text(encoding="utf-8") == CODE
    assert accepted.exported_elements == ()


def test_candidate_acceptor_rejects_unsafe_code():
    with pytest.raises(CandidateRejected, match="AST/安全校验"):
        CandidateAcceptor().inspect("import os\nos.system('rm -rf /')", make_plan())


def test_candidate_acceptor_records_and_returns_deterministic_lifecycle_repairs():
    technical = TechnicalSpec(
        scene_id=1,
        objects=[
            TechnicalObject(
                element_id="formula",
                variable_name="formula",
                constructor="MathTex",
                exported=True,
            )
        ],
        animations=[
            {
                "event_id": "show_formula",
                "start_seconds": 0,
                "end_seconds": 1,
                "semantic_action": "introduce",
                "target_element_ids": ["formula"],
                "create_element_ids": ["formula"],
            }
        ],
        export_element_ids=["formula"],
    )
    code = """from manim import *
class Demo(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        tex_template.add_to_preamble(r"\\usepackage{ctex}")
        config.tex_template = tex_template
        # KD1_CONTINUITY_EXPORT_BEGIN
        # element_id: formula
        formula = MathTex(r"x", tex_template=tex_template)
        # KD1_CONTINUITY_EXPORT_END
        self.play(FadeIn(formula))
"""

    plan = make_plan().model_copy(
        update={
            "new_elements": [
                VisualElementState(element_id="formula", variable_name="formula", required=True)
            ]
        }
    )
    accepted = CandidateAcceptor().inspect(code, plan, technical_spec=technical)

    assert accepted.repairs == ("为第 11 行 self.play() 补齐事件标记: show_formula",)
    assert "# KD1_ANIMATION_EVENT: show_formula" in accepted.code
