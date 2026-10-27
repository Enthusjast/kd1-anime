import pytest
from test_agent_loop import setup


def test_process_interruption_observer_runs_after_commit_without_rewriting_intent():
    loop, _, executions, persisted = setup(progress=True)

    def interrupt():
        assert any(name == "commit" for name, _ in persisted)
        assert [value for name, value in persisted if name == "intent"][-1].status == "completed"
        raise OSError("isolated stage interruption")

    loop.after_step = interrupt
    with pytest.raises(OSError, match="isolated stage interruption"):
        loop.run()
    assert len(executions) == 1
    assert [value for name, value in persisted if name == "intent"][-1].status == "completed"
