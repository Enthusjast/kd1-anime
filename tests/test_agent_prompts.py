from kd1_anime.agents.auto_fixer import AUTO_FIXER_SYSTEM_PROMPT, AutoFixerAgent
from kd1_anime.agents.coder import CODER_SYSTEM_PROMPT, build_coder_system_prompt
from kd1_anime.config import settings


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


def test_coder_guards_against_blind_subscript_access():
    assert "IndexError" in CODER_SYSTEM_PROMPT
    assert "get_part_by_tex" in CODER_SYSTEM_PROMPT
    assert "len()" in CODER_SYSTEM_PROMPT


def test_coder_prompt_has_continuity_contract():
    assert "opening_state" in CODER_SYSTEM_PROMPT
    assert "closing_state" in CODER_SYSTEM_PROMPT
    assert "persistent_elements" in CODER_SYSTEM_PROMPT
    assert "transition_out" in CODER_SYSTEM_PROMPT
    assert "RAG Reference Context" in CODER_SYSTEM_PROMPT


def test_coder_forbids_custom_mobject_subclass_for_opengl(monkeypatch):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "opengl")
    prompt = build_coder_system_prompt()
    assert "自定义 Mobject/VMobject 子类" in prompt
    assert "should_render" in prompt


def test_coder_allows_moving_camera_for_cairo(monkeypatch):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "cairo")
    prompt = build_coder_system_prompt()
    assert "MovingCameraScene" in prompt
    assert "self.camera.frame.animate" in prompt


def test_explicit_renderer_context_overrides_mutated_process_settings(monkeypatch):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "opengl")

    prompt = build_coder_system_prompt("cairo")

    assert "Renderer: Cairo" in prompt
    assert "self.camera.frame.animate" in prompt


def test_auto_fixer_has_should_render_pattern():
    assert "should_render" in AUTO_FIXER_SYSTEM_PROMPT
    assert "自定义 mobject 子类" in AUTO_FIXER_SYSTEM_PROMPT


def test_auto_fixer_has_index_error_pattern():
    assert "IndexError" in AUTO_FIXER_SYSTEM_PROMPT
    assert "下标越界" in AUTO_FIXER_SYSTEM_PROMPT
    assert "get_part_by_tex" in AUTO_FIXER_SYSTEM_PROMPT
    assert "ctex" in AUTO_FIXER_SYSTEM_PROMPT
    assert "RAG" in AUTO_FIXER_SYSTEM_PROMPT
    assert AutoFixerAgent.is_infrastructure_error(
        "FileNotFoundError: No such file or directory: 'xelatex'"
    )
    assert not AutoFixerAgent.is_infrastructure_error(
        "FileNotFoundError: No such file or directory: 'pdflatex'"
    )
