from concurrent.futures import ThreadPoolExecutor
from threading import Event

from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.application import Application


def test_cancel_during_runtime_construction_never_dispatches_a_model_or_job(agent_boundary):
    boundary, dependencies = agent_boundary
    entered, release = Event(), Event()

    def create():
        entered.set()
        assert release.wait(5)
        return AgentRuntime(dependencies=dependencies)

    app = Application(runtime_factory=create)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(app.generate, "Show a pause", approve_plan=True)
        try:
            assert entered.wait(5)
            app.cancel_all()
        finally:
            release.set()
        assert task.result(timeout=10) is None
    assert not boundary.calls and not boundary.backend.sources
