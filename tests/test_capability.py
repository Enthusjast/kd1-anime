from kd1_anime.agents.capability import (
    build_capability_contract,
    recommended_renderer,
    validate_capability_contract,
)
from kd1_anime.agents.planner import ScenePlan
from kd1_anime.agents.technical_planner import TechnicalObject, TechnicalSpec
from kd1_anime.rendering import RenderProfile


def make_plan(**updates):
    data = {
        "scene_id": 1,
        "title": "demo",
        "duration_seconds": 10,
        "purpose": "test",
        "math_concept": "circle",
        "visual_design": "simple",
        "camera_movement": "fixed",
        "visual_flow": ["show"],
        "key_moments": ["pause"],
        "computation": "radius=1",
    }
    data.update(updates)
    return ScenePlan(**data)


def profile(renderer: str):
    return RenderProfile(
        renderer=renderer,
        quality="h",
        pixel_width=1920,
        pixel_height=1080,
        frame_rate=60,
        opengl_platform="egl",
    )


def test_capability_contract_infers_3d_tex_numpy_and_gpu_requirements():
    spec = TechnicalSpec(
        scene_id=1,
        renderer="opengl",
        objects=[
            TechnicalObject(element_id="surface", constructor="Surface"),
            TechnicalObject(element_id="formula", constructor="MathTex"),
        ],
    )

    contract = build_capability_contract(
        make_plan(), spec, renderer="opengl", render_profile=profile("opengl"), gpu_type="A100"
    )

    assert contract.requires_3d is True
    assert contract.scene_parent == "ThreeDScene"
    assert contract.requires_tex is True
    assert contract.requires_numpy is True
    assert contract.requires_gpu is True
    assert recommended_renderer(contract) == "opengl"


def test_capability_contract_rejects_opengl_camera_frame():
    contract = build_capability_contract(
        make_plan(camera_movement="使用 self.camera.frame 推近"),
        renderer="opengl",
        render_profile=profile("opengl"),
        gpu_type="A100",
    )

    result = validate_capability_contract(
        contract,
        render_profile=profile("opengl"),
        gpu_type="A100",
        code="from manim import *\nclass Demo(MovingCameraScene):\n    def construct(self): self.camera.frame",
    )

    assert result.is_valid is False
    assert any("OpenGL" in error for error in result.errors)


def test_capability_contract_allows_cairo_moving_camera_and_warns_unknown_tex():
    contract = build_capability_contract(
        make_plan(camera_movement="使用 self.camera.frame 推近"),
        renderer="cairo",
        render_profile=profile("cairo"),
    )
    result = validate_capability_contract(
        contract,
        render_profile=profile("cairo"),
        code="from manim import *\nclass Demo(MovingCameraScene):\n    def construct(self): pass",
    )

    assert result.is_valid is True
    assert result.warnings == ()
    assert recommended_renderer(contract) == "cairo"
