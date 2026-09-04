from kd1_anime.agents.planner import ScenePlan
from kd1_anime.agents.review_policy import review_budget


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


def test_low_risk_review_has_small_llm_budget_but_keeps_deterministic_gate():
    budget = review_budget(make_plan(), None, global_max_rounds=5, low_risk_max_rounds=2)

    assert budget.risk_level == "low"
    assert budget.max_rounds == 2
    assert budget.deterministic_checks_required is True


def test_high_risk_review_uses_global_budget():
    budget = review_budget(
        make_plan(visual_design="三维 Surface 曲面", computation="复杂切割"),
        None,
        global_max_rounds=5,
        low_risk_max_rounds=1,
    )

    assert budget.risk_level == "high"
    assert budget.max_rounds == 5
