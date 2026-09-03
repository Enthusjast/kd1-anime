import pytest

from kd1_anime.agents.planner import ScenePlan
from kd1_anime.cluster.resource_estimator import (
    RenderResourceProfile,
    estimate_render_resources,
)
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


def render_profile(renderer="cairo"):
    return RenderProfile(
        renderer=renderer,
        quality="h",
        pixel_width=1920,
        pixel_height=1080,
        frame_rate=60,
        opengl_platform="egl",
    )


def test_estimator_preserves_base_resources_for_simple_cairo_scene():
    result = estimate_render_resources(
        make_plan(),
        None,
        render_profile(),
        cpus_per_task=4,
        mem_gb="16G",
        time_limit="00:10:00",
        gpu_type="RTX5090",
        gpu_count=1,
    )

    assert result.cpus_per_task == 4
    assert result.mem_gb == 16
    assert result.time_limit == "00:10:00"
    assert result.gpu_type == ""


def test_estimator_scales_complex_opengl_scene_only_when_enabled():
    plan = make_plan(
        visual_design="三维 Surface 曲面",
        computation="updater 逐帧切割和旋转",
    )
    result = estimate_render_resources(
        plan,
        None,
        render_profile("opengl"),
        cpus_per_task=4,
        mem_gb="16G",
        time_limit="00:10:00",
        gpu_type="RTX5090",
        gpu_count=1,
        apply_estimate=True,
    )

    assert result.estimated is True
    assert result.cpus_per_task == 6
    assert result.mem_gb == 20
    assert result.time_limit == "00:40:00"
    assert result.gpu_type == "RTX5090"
    assert result.reasons


def test_resource_profile_rejects_unsafe_slurm_values():
    with pytest.raises(ValueError):
        RenderResourceProfile(
            cpus_per_task=4,
            time_limit="01:00:00",
            gpu_type="RTX5090\n#SBATCH --exclusive",
        )
