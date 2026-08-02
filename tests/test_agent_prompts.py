from kd1_anime.agents.auto_fixer import AUTO_FIXER_SYSTEM_PROMPT, AutoFixerAgent
from kd1_anime.agents.coder import CODER_SYSTEM_PROMPT


def test_coder_requires_xelatex_xdv_and_ctex_template():
    assert 'tex_compiler="xelatex"' in CODER_SYSTEM_PROMPT
    assert 'output_format=".xdv"' in CODER_SYSTEM_PROMPT
    assert r"\usepackage{ctex}" in CODER_SYSTEM_PROMPT
    assert "config.tex_template" in CODER_SYSTEM_PROMPT
    assert "tex_template=tex_template" in CODER_SYSTEM_PROMPT


def test_coder_prompt_has_scene_skeleton_and_spatial_rules():
    assert "代码骨架模板" in CODER_SYSTEM_PROMPT
    assert "class Scene1(Scene)" in CODER_SYSTEM_PROMPT
    assert "空间布局约束" in CODER_SYSTEM_PROMPT
    assert "next_to" in CODER_SYSTEM_PROMPT
    assert "不得越界" in CODER_SYSTEM_PROMPT


def test_coder_prompt_has_self_check_and_camera_guard():
    assert "自查清单" in CODER_SYSTEM_PROMPT
    assert "self.camera.frame" in CODER_SYSTEM_PROMPT
    assert "MovingCameraScene" in CODER_SYSTEM_PROMPT
    assert "if __name__" in CODER_SYSTEM_PROMPT
    assert "中文一律" in CODER_SYSTEM_PROMPT


def test_auto_fixer_preserves_xelatex_invariant():
    assert 'tex_compiler="xelatex"' in AUTO_FIXER_SYSTEM_PROMPT
    assert "ctex" in AUTO_FIXER_SYSTEM_PROMPT
    assert AutoFixerAgent.is_infrastructure_error(
        "FileNotFoundError: No such file or directory: 'xelatex'"
    )
    assert not AutoFixerAgent.is_infrastructure_error(
        "FileNotFoundError: No such file or directory: 'pdflatex'"
    )
