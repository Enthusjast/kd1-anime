from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.run_store import RunRepository


def test_deterministic_math_error_routes_to_bounded_planner_repair_before_code(agent_boundary):
    boundary, dependencies = agent_boundary
    original_planner = dependencies.planner_factory
    original_reviewer = dependencies.plan_reviewer_factory
    detailed = []
    reviewed = []

    class Planner(original_planner):
        def plan_detail(self, *args, **kwargs):
            plan = super().plan_detail(*args, **kwargs)
            computation = "1+1=3" if not detailed else "1+1=2"
            detailed.append(computation)
            return plan.model_copy(update={"computation": computation})

    class Reviewer(original_reviewer):
        def review(self, plan, **kwargs):
            assert plan.computation == "1+1=2"
            reviewed.append(plan.computation)
            return super().review(plan, **kwargs)

    dependencies.planner_factory = Planner
    dependencies.plan_reviewer_factory = Reviewer
    result = AgentRuntime(dependencies=dependencies).generate(
        "Show a correct pause", dry_run=True, approve_plan=True
    )
    assert result is None
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load(root.name)
    assert manifest.status == "dry_run_complete"
    assert detailed == ["1+1=3", "1+1=2"] and reviewed == ["1+1=2"]
    assert manifest.scenes[1].reviewed and manifest.scenes[1].plan.computation == "1+1=2"
    repairs = [item for item in manifest.agent.intents if item.action.kind == "revise_plan"]
    assert len(repairs) == 1 and repairs[0].status == "completed"
    assert not boundary.backend.sources
