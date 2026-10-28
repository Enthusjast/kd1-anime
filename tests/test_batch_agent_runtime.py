"""Batch contracts use public Application methods and isolated v11 collaborators."""

import io
import threading
from types import SimpleNamespace
from typing import ClassVar

import pytest
from rich.console import Console
from test_agent_vertical_slice import agent_boundary as agent_boundary

import kd1_anime.application as application_module
import kd1_anime.batch as batch_module
from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.agent_runtime.session import SessionStore
from kd1_anime.batch import BatchConfig, BatchProcessor
from kd1_anime.config import settings
from kd1_anime.domain.review import ReviewResult
from kd1_anime.run_store import RunRepository

RUN_ID = "20260728-120000-1234abcd"


@pytest.fixture
def fake_application(monkeypatch):
    class FakeApplication:
        status = "dry_run_complete"
        state = "DONE"
        run_phase = "finished"
        stop_reason = ""
        result = None
        failure = None
        instances: ClassVar[list] = []

        def __init__(self, *, resource_coordinator):
            self.resources = resource_coordinator
            self.calls = []
            self.cancelled = False
            self.instances.append(self)

        def generate(self, prompt, **options):
            self.calls.append(("generate", prompt, options))
            options["callback"]("run_started", {"run_id": RUN_ID})
            if self.failure is not None:
                raise self.failure
            return self.result

        def run_incremental(self, prompt, base_run_id, **options):
            self.base_run_id = base_run_id
            return self.generate(prompt, **options)

        def run_view(self, run_id):
            assert run_id == RUN_ID
            return SimpleNamespace(
                status=self.status,
                state=self.state,
                run_phase=self.run_phase,
                stop_reason=self.stop_reason,
            )

        def cancel_all(self):
            self.cancelled = True

    monkeypatch.setattr(application_module, "Application", FakeApplication)
    return FakeApplication


@pytest.mark.parametrize(
    ("status", "state", "run_phase", "expected"),
    [
        ("running", "PLAN_REVIEWING", "planning", "waiting"),
        ("running", "REVIEWING", "executing", "waiting"),
        ("failed", "ERROR", "failed", "failed"),
        ("interrupted", "MONITORING", "executing", "interrupted"),
        ("dry_run_complete", "DONE", "finished", "dry_run_complete"),
        ("completed", "DONE", "finished", "completed"),
    ],
)
def test_status_comes_from_public_view_not_return_value(
    fake_application, tmp_path, status, state, run_phase, expected
):
    fake_application.status = status
    fake_application.state = state
    fake_application.run_phase = run_phase
    fake_application.stop_reason = "approval or execution detail"
    fake_application.result = tmp_path / "artifact.mp4" if status == "completed" else None
    processor = BatchProcessor(BatchConfig(dry_run=True))
    task = processor.add_task("demo", tmp_path / "requested.mp4")

    assert processor._execute_single_task(task) is task

    assert task.status == expected
    assert task.run_id == RUN_ID
    assert task.state == state and task.run_phase == run_phase
    assert task.start_time is not None and task.end_time is not None
    assert task.output == fake_application.result
    if expected in {"waiting", "failed", "interrupted"}:
        assert task.error == fake_application.stop_reason
    options = fake_application.instances[0].calls[0][2]
    assert options["approve_plan"] is False and options["interactive"] is False
    assert options["output_path"] == (tmp_path / "requested.mp4").resolve()


def test_failure_keeps_callback_identity_and_view_progress(fake_application):
    fake_application.failure = RuntimeError("render failed")
    fake_application.status = "failed"
    fake_application.state = "ERROR"
    fake_application.run_phase = "failed"
    processor = BatchProcessor()
    task = processor.add_task("demo")

    processor._execute_single_task(task)

    assert task.status == "failed" and task.error == "render failed"
    assert task.run_id == RUN_ID and task.state == "ERROR" and task.run_phase == "failed"


def test_missing_run_started_is_failure_not_success(fake_application, monkeypatch):
    monkeypatch.setattr(fake_application, "generate", lambda *args, **kwargs: None)
    processor = BatchProcessor(BatchConfig(dry_run=True))
    task = processor.add_task("demo")

    processor._execute_single_task(task)

    assert task.status == "failed" and "run_started" in task.error


def test_incremental_uses_public_application_and_explicit_plan_permission(
    fake_application, tmp_path
):
    processor = BatchProcessor(
        BatchConfig(
            incremental=True,
            base_run_ids={1: RUN_ID},
            approve_plan=True,
            backend="local",
            output_dir=tmp_path,
        )
    )
    task = processor.add_task("edit")

    processor._execute_single_task(task)

    application = fake_application.instances[0]
    assert application.base_run_id == RUN_ID
    options = application.calls[0][2]
    assert options["approve_plan"] is True and options["interactive"] is False
    assert options["backend"] == "local"
    assert options["output_path"] == tmp_path / "task_001.mp4"
    assert "base_manifest" not in options
    assert application.resources is processor.resources


def test_keyboard_interrupt_calls_public_application_cancel(fake_application, monkeypatch):
    started = threading.Event()
    cancelled = threading.Event()

    def generate(self, prompt, **options):
        options["callback"]("run_started", {"run_id": RUN_ID})
        started.set()
        assert cancelled.wait(timeout=5)
        return None

    def cancel(self):
        self.cancelled = True
        cancelled.set()

    def interrupt(futures):
        assert started.wait(timeout=5)
        raise KeyboardInterrupt

    monkeypatch.setattr(fake_application, "generate", generate)
    monkeypatch.setattr(fake_application, "cancel_all", cancel)
    monkeypatch.setattr(batch_module, "as_completed", interrupt)
    processor = BatchProcessor(BatchConfig(max_parallel=1))
    processor.add_task("demo")

    tasks = processor.execute_all()

    assert fake_application.instances[0].cancelled
    assert tasks[0].status == "interrupted" and tasks[0].run_id == RUN_ID
    assert tasks[0].output is None and "已停止" not in tasks[0].error


def test_cancel_failure_is_retained_without_claiming_job_stopped(fake_application, monkeypatch):
    processor = BatchProcessor()

    def generate(self, prompt, **options):
        options["callback"]("run_started", {"run_id": RUN_ID})
        processor.cancel_all()
        raise RuntimeError("execution interrupted")

    def cancel(self):
        raise RuntimeError("remote state unknown")

    monkeypatch.setattr(fake_application, "generate", generate)
    monkeypatch.setattr(fake_application, "cancel_all", cancel)
    task = processor.add_task("demo")

    processor._execute_single_task(task)

    assert task.status == "interrupted" and task.run_id == RUN_ID
    assert "remote state unknown" in task.error and "未确认" in task.error
    assert task.output is None


def test_waiting_output_preserves_identity_and_redacts_untrusted_text(
    fake_application, monkeypatch
):
    secret = "batch-configured-secret"
    monkeypatch.setattr(settings, "LLM_API_KEY", secret)
    fake_application.status = "running"
    fake_application.state = "PLAN_REVIEWING"
    fake_application.stop_reason = f"[bold]wait[/bold] {secret} https://example.test?token=abc"
    output = io.StringIO()
    monkeypatch.setattr(batch_module, "console", Console(file=output, width=220))
    processor = BatchProcessor()
    processor.add_task(f"[link=https://evil.test]prompt[/link] {secret}")

    tasks = processor.execute_all()
    text = output.getvalue()
    summary = processor.generate_summary(tasks)

    assert tasks[0].status == "waiting" and tasks[0].output is None
    assert "WAITING" in text and RUN_ID in text and RUN_ID in summary
    assert "[bold]wait[/bold]" in text and "[link=https://evil.test]" in text
    assert secret not in text + summary + tasks[0].error
    assert "token=abc" not in text + summary
    assert "等待: 1" in summary and "成功: 0" in summary


def test_output_conflicts_fail_before_constructing_any_application(
    fake_application, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "OVERWRITE_OUTPUT", False)
    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"keep")
    processor = BatchProcessor()
    processor.add_task("demo", existing)

    with pytest.raises(FileExistsError, match="OVERWRITE_OUTPUT"):
        processor.execute_all()

    assert not fake_application.instances and existing.read_bytes() == b"keep"


@pytest.mark.parametrize("approve_plan", [False, True])
def test_real_agent_runs_have_independent_contexts_and_one_shared_coordinator(
    agent_boundary, monkeypatch, approve_plan
):
    boundary, dependencies = agent_boundary
    runtimes = []

    def runtime_factory(**options):
        runtime = AgentRuntime(dependencies=dependencies, **options)
        runtimes.append((runtime, options["resource_coordinator"]))
        return runtime

    def no_stdin(*args, **kwargs):
        raise AssertionError("batch worker read stdin")

    monkeypatch.setattr(application_module, "_agent_runtime", runtime_factory)
    monkeypatch.setattr("builtins.input", no_stdin)
    processor = BatchProcessor(BatchConfig(max_parallel=2, dry_run=True, approve_plan=approve_plan))
    processor.add_task("Show the first pause")
    processor.add_task("Show the second pause")

    tasks = processor.execute_all()

    expected = "dry_run_complete" if approve_plan else "waiting"
    assert {task.status for task in tasks} == {expected}
    assert len({task.run_id for task in tasks}) == 2
    assert len(runtimes) == 2 and runtimes[0][0] is not runtimes[1][0]
    assert all(coordinator is processor.resources for _, coordinator in runtimes)
    manifests = [RunRepository(boundary.workspace).load(task.run_id) for task in tasks]
    assert all(manifest.schema_version == 11 for manifest in manifests)
    assert {manifest.user_prompt for manifest in manifests} == {task.prompt for task in tasks}
    assert len({manifest.agent.session_id for manifest in manifests}) == 2
    assert all(manifest.request_budget.reservations for manifest in manifests)
    first_ids, second_ids = [
        {receipt.request_id for receipt in manifest.request_budget.reservations}
        for manifest in manifests
    ]
    assert first_ids.isdisjoint(second_ids)
    assert all(task.output is None for task in tasks)
    assert not boundary.backend.sources
    if not approve_plan:
        assert all(manifest.agent.pending_approval is not None for manifest in manifests)
        assert all(
            not scene.code_file for manifest in manifests for scene in manifest.scenes.values()
        )


def test_explicit_batch_plan_permission_does_not_grant_code_diff(agent_boundary, monkeypatch):
    boundary, dependencies = agent_boundary
    original = dependencies.reviewer_factory

    class Reviewer(original):
        def review(self, code, plan, **kwargs):
            accepted = super().review(code, plan, **kwargs)
            if "self.wait(1.0)" in code:
                return accepted
            return ReviewResult(
                is_valid=False,
                severity="major",
                feedback="Use explicit floating duration",
                fixes=[{"find": "self.wait(1)", "replace": "self.wait(1.0)"}],
            )

    dependencies.reviewer_factory = Reviewer
    monkeypatch.setattr(
        application_module,
        "_agent_runtime",
        lambda **options: AgentRuntime(dependencies=dependencies, **options),
    )
    processor = BatchProcessor(BatchConfig(dry_run=True, approve_plan=True))
    task = processor.add_task("Show a pause")

    processor._execute_single_task(task)

    assert task.status == "waiting" and task.output is None
    manifest = RunRepository(boundary.workspace).load(task.run_id)
    assert manifest.agent.pending_approval.kind == "inspect_candidate"
    session = SessionStore(boundary.workspace / "sessions").load(manifest.agent.session_id)
    assert {grant.action for grant in session.grants} == {"review_plan"}
    assert not boundary.backend.sources
