from io import StringIO

from rich.console import Console
from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.application import Application
from kd1_anime.run_store import RunRepository
from kd1_anime.tui import ChatSession


def test_tui_approves_exact_plan_using_persisted_session_command(agent_boundary, monkeypatch):
    boundary, dependencies = agent_boundary
    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", lambda: AgentRuntime(dependencies=dependencies)
    )
    monkeypatch.setattr("kd1_anime.tui.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("kd1_anime.tui.Confirm.ask", lambda *args, **kwargs: True)
    output = StringIO()
    monkeypatch.setattr("kd1_anime.tui.console", Console(file=output, force_terminal=False))
    app = Application()
    session_id = app.create_session("Original goal")
    app.session_command(session_id, "ready", "Show a pause")
    app.session_command(session_id, "confirm", "Show a pause")
    session = ChatSession(dry_run=True)
    session.application, session.session_id = app, session_id
    assert session._run_pipeline("Show a pause") is True
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load(root.name)
    assert manifest.status == "dry_run_complete" and manifest.agent.session_id == session_id
    requests = app.approval_requests(root.name)
    assert requests and requests[-1].status == "consumed"
    assert "等待批准" in output.getvalue()


def test_non_tty_tui_leaves_approval_pending_without_reading_stdin(agent_boundary, monkeypatch):
    boundary, dependencies = agent_boundary
    monkeypatch.setattr(
        "kd1_anime.application._agent_runtime", lambda: AgentRuntime(dependencies=dependencies)
    )
    monkeypatch.setattr("kd1_anime.tui.sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(
        "kd1_anime.tui.Confirm.ask", lambda *a, **k: (_ for _ in ()).throw(AssertionError("stdin"))
    )
    session = ChatSession(dry_run=True)
    assert session._run_pipeline("Show a pause") is False
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load(root.name)
    assert manifest.agent.pending_approval is not None
    assert not boundary.backend.sources and session.exit_code == 1
