from pathlib import Path

import pytest

from kd1_anime.agents.planner import ScenePlan
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
