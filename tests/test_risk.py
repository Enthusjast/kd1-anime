from kd1_anime.agents.planner import ScenePlan
from kd1_anime.agents.risk import assess_scene_risk
from kd1_anime.agents.technical_planner import TechnicalSpec


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


def test_simple_scene_is_low_risk():
    assert assess_scene_risk(make_plan()).level == "low"


def test_complex_three_d_scene_is_high_risk():
    plan = make_plan(
        visual_design="三维曲面和切平面",
        camera_movement="3D camera rotation",
        computation="使用 updater 逐帧展示切割和旋转，面积守恒",
        visual_flow=["切割碎片" for _ in range(10)],
        key_moments=["逐帧" for _ in range(5)],
        geometry_specs=[{"geometry_id": f"g{i}", "shape": "polygon"} for i in range(3)],
    )
    spec = TechnicalSpec(
        scene_id=1,
        renderer="opengl",
        objects=[{"element_id": f"e{i}"} for i in range(8)],
        animations=[
            {
                "event_id": f"a{i}",
                "start_seconds": float(i),
                "end_seconds": float(i) + 0.5,
                "semantic_action": "hold",
            }
            for i in range(12)
        ],
        latex={"required": True},
    )

    risk = assess_scene_risk(plan, spec)

    assert risk.level == "high"
    assert risk.score >= 7
    assert risk.reasons
