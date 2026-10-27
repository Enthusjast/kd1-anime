"""Durable transactions, startup identity, cancellation and bounded dispatch."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_execution_contracts import CODE, plan, prepare_submission, submitted_job, workflow

from kd1_anime.agents.api_linter import lint_manim_api
from kd1_anime.agents.base import BaseAgent
from kd1_anime.clients import ClientPool, client_scope
from kd1_anime.cluster.jobs import PollObservation
from kd1_anime.cluster.monitoring import RenderMonitorCoordinator
from kd1_anime.commits import CommitCommand
from kd1_anime.config import settings
from kd1_anime.control import CURRENT_CONTROL, OperationDeadlineExceeded, RunCancelled, RunControl
from kd1_anime.domain.dependencies import StaleWorkError, WorkIdentity
from kd1_anime.domain.program import SceneProgram
from kd1_anime.domain.runtime import ExecutionSpec, RunConfig
from kd1_anime.eval.benchmark import fixed_tasks, task_plan
from kd1_anime.events import BufferedCallback, CallbackAdapter, PipelineEvent
from kd1_anime.models import validated_copy
from kd1_anime.objects import ObjectStore
from kd1_anime.orchestrator import Orchestrator, SceneState, State
from kd1_anime.request_budget import RequestContext, RequestReservation
from kd1_anime.run_store import RunRepository, atomic_write_text
from kd1_anime.scheduling import ContextPool, ReadyQueue, WorkItem
from kd1_anime.services.program_compiler import ProgramCompileError, compile_program
from kd1_anime.services.results import run_stage


def test_commit_applies_only_to_a_copy_and_publishes_after_disk(tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    orchestrator._checkpoint(ctx, State.INIT)
    observed = []

    def callback(name, data):
        if name == "commit_completed":
            saved = RunRepository(tmp_path).load(ctx.paths.run_id)
            assert saved.revision == data["manifest_revision"] == ctx.manifest_revision
            assert saved.user_prompt == ctx.user_prompt == "changed"
            observed.append(saved.revision)

    orchestrator._callback = callback
    original = ctx.manifest_revision
    orchestrator.commit(
        ctx,
        CommitCommand(
            original, "CODING", apply=lambda draft: setattr(draft, "user_prompt", "changed")
        ),
    )
    assert observed == [original + 1]


def test_failed_commit_restores_all_memory_and_manifest(monkeypatch, tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    orchestrator._checkpoint(ctx, State.INIT)
    scene = ctx.scene_states[1]
    previous = (ctx.paths.root / "manifest.json").read_bytes()
    scene.plan.title = "uncommitted title"
    ctx.user_prompt = "uncommitted prompt"
    monkeypatch.setattr(
        "kd1_anime.orchestrator.write_manifest", lambda *a: (_ for _ in ()).throw(OSError("full"))
    )
    with pytest.raises(OSError, match="full"):
        orchestrator._checkpoint(ctx, State.CODING)
    assert ctx.scene_states[1] is scene
    assert scene.plan.title == "example"
    assert ctx.user_prompt == "prompt"
    assert (ctx.paths.root / "manifest.json").read_bytes() == previous


@pytest.mark.parametrize("cut", ["source_saved", "before_replace", "submission_returned"])
def test_process_crash_exposes_only_the_last_complete_snapshot(tmp_path, cut):
    orchestrator, ctx, _state = prepare_submission(tmp_path)
    orchestrator._checkpoint(ctx, State.REVIEWING)
    before = (ctx.paths.root / "manifest.json").read_bytes()
    script = r"""
import os,sys
from pathlib import Path
from kd1_anime.orchestrator import Orchestrator,State
from kd1_anime.run_store import RunRepository
import kd1_anime.run_store as store
workspace=Path(sys.argv[1]);cut=sys.argv[2];run_id=sys.argv[3]
manifest=RunRepository(workspace).load_for_resume(run_id)
root=workspace/'runs'/run_id
o=Orchestrator();ctx=o._context_from_manifest(manifest,root)
o._ctx=ctx;o._manifest=manifest
if cut=='submission_returned':
    def submit(*a,**kw): os._exit(73)
    o.slurm.submit_scene=submit
    o._scene_submit(ctx,1,ctx.scene_states[1])
else:
    ctx.scene_states[1].code+='\n# new revision\n'
    o._mark_static_verification(ctx.scene_states[1],status='passed')
    original=store.os.replace
    def replace(source,target):
        if Path(target).name=='manifest.json': os._exit(73)
        return original(source,target)
    store.os.replace=replace
    if cut=='source_saved':
        from kd1_anime.objects import ObjectStore
        original_source=ObjectStore.source
        def source(self,*args):
            result=original_source(self,*args)
            os._exit(73)
        ObjectStore.source=source
    o._checkpoint(ctx,State.CODING)
raise SystemExit('fault injection did not run')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), cut, ctx.paths.run_id],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 73, result.stderr
    saved = RunRepository(tmp_path).load_for_resume(ctx.paths.run_id)
    restored = Orchestrator._context_from_manifest(saved, ctx.paths.root)
    assert restored.scene_states[1].code == CODE
    if cut == "submission_returned":
        assert saved.scenes[1].submission.status == "pending"
        with pytest.raises(RuntimeError, match="禁止自动重提"):
            Orchestrator()._scene_submit(restored, 1, restored.scene_states[1])
    else:
        assert (ctx.paths.root / "manifest.json").read_bytes() == before


def test_directory_sync_failure_rolls_back_the_published_name(monkeypatch, tmp_path):
    import stat

    import kd1_anime.run_store as store

    destination = tmp_path / "manifest.json"
    atomic_write_text(destination, "previous")
    original = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory sync failed")
        original(fd)

    monkeypatch.setattr(store.os, "fsync", fsync)
    with pytest.raises(OSError, match="directory sync"):
        atomic_write_text(destination, "new")
    assert destination.read_text() == "previous"


def test_immutable_objects_are_private_deduplicated_and_never_repaired(tmp_path):
    store = ObjectStore(tmp_path)
    relative = store.source(1, CODE)
    target = tmp_path / relative
    inode = target.stat().st_ino
    assert store.source(1, CODE) == relative
    assert target.stat().st_ino == inode
    assert target.stat().st_mode & 0o777 == 0o600
    assert all(
        folder.stat().st_mode & 0o777 == 0o700 for folder in [target.parent, target.parent.parent]
    )
    target.write_text("tampered")
    with pytest.raises(ValueError, match="identity mismatch"):
        store.source(1, CODE)
    assert target.read_text() == "tampered"


@pytest.mark.parametrize("ancestor", ["scenes", "scenes/revisions"])
def test_object_store_rejects_internal_symlink_ancestors(tmp_path, ancestor):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / ancestor
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe"):
        ObjectStore(tmp_path).source(1, CODE)


@pytest.mark.parametrize("field", ["run_config", "execution_spec", "snapshot", "projection"])
def test_v10_missing_or_conflicting_contracts_rejected_on_load(tmp_path, field):
    orchestrator, ctx = workflow(tmp_path)
    orchestrator._checkpoint(ctx, State.CODING)
    path = ctx.paths.root / "manifest.json"
    raw = json.loads(path.read_text())
    if field in {"run_config", "execution_spec"}:
        raw[field] = {}
    elif field == "snapshot":
        del raw["scenes"]["1"]["snapshot"]["lifecycle"]
    else:
        raw["scenes"]["1"]["plan"]["title"] = "conflicting flat projection"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        RunRepository(tmp_path).load_for_resume(ctx.paths.run_id)


@pytest.mark.parametrize("version", range(4, 10))
@pytest.mark.parametrize("entry", ["resume", "retry", "incremental"])
def test_all_legacy_execution_entries_reject_before_side_effects(
    tmp_path, monkeypatch, version, entry
):
    orchestrator, ctx = workflow(tmp_path)
    orchestrator._checkpoint(ctx, State.INIT)
    path = ctx.paths.root / "manifest.json"
    raw = json.loads(path.read_text())
    raw["schema_version"] = version
    before = json.dumps(raw)
    path.write_text(before)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(
        "kd1_anime.orchestrator.lock_run", lambda *a: pytest.fail("lock before version rejection")
    )
    monkeypatch.setattr(
        orchestrator.rag, "require_ready", lambda: pytest.fail("RAG before version rejection")
    )
    with pytest.raises(ValueError, match="只读"):
        if entry == "incremental":
            orchestrator.run_incremental("new", ctx.paths.run_id)
        else:
            orchestrator.resume(ctx.paths.run_id, retry_scene_id=1 if entry == "retry" else None)
    assert path.read_text() == before
    assert not (ctx.paths.root / ".lock").exists()
    exported = RunRepository(tmp_path).load(ctx.paths.run_id).model_dump(mode="json")
    assert exported == raw


def test_report_hash_is_checked_before_restore(tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    orchestrator._write_stage_artifact(ctx, "review.json", {"status": "passed"})
    orchestrator._checkpoint(ctx, State.CODING)
    reference = ctx.artifact_refs["review.json"]
    (ctx.paths.root / reference).write_text("{}")
    with pytest.raises(ValueError, match="report identity"):
        Orchestrator._context_from_manifest(orchestrator._manifest, ctx.paths.root)


def test_configuration_is_immutable_and_credentials_are_runtime_only(monkeypatch):
    monkeypatch.setattr(settings, "LLM_API_KEY", "private-main-key")
    monkeypatch.setattr(settings, "VISUAL_LLM_API_KEY", "private-visual-key")
    config = RunConfig.capture()
    original = settings.MAX_REVIEW_ROUNDS
    monkeypatch.setattr(settings, "MAX_REVIEW_ROUNDS", original + 1)
    monkeypatch.setattr(settings, "LLM_API_KEY", "rotated-main-key")
    assert "private-main-key" not in config.model_dump_json()
    assert "private-visual-key" not in config.model_dump_json()
    with config.activate():
        assert original == settings.MAX_REVIEW_ROUNDS
        assert settings.LLM_API_KEY == "rotated-main-key"
    assert original + 1 == settings.MAX_REVIEW_ROUNDS
    with pytest.raises(ValidationError):
        config.values_json = "{}"


def test_image_bytes_change_execution_identity_and_fail_verification(monkeypatch, tmp_path):
    image = tmp_path / "render.sif"
    image.write_bytes(b"image one")
    monkeypatch.setattr(settings, "SLURM_CONTAINER_IMAGE", str(image))
    first = ExecutionSpec.capture()
    image.write_bytes(b"image two")
    second = ExecutionSpec.capture()
    assert first.digest() != second.digest()
    with pytest.raises(ValueError, match="Container image"):
        first.verify_image()


def test_stage_result_rejects_stale_input_and_cancelled_results(tmp_path):
    identity = WorkIdentity(4, "a" * 64)
    result = run_stage(identity, lambda: {"accepted": True})
    assert result.revision == 4
    with pytest.raises(StaleWorkError):
        result.accept(WorkIdentity(5, "a" * 64))
    orchestrator, ctx = workflow(tmp_path)
    identity = orchestrator._work_identity(ctx, ctx.scene_states[1])
    orchestrator._cancel_requested.set()
    with pytest.raises(RunCancelled):
        orchestrator._accept_work(ctx, ctx.scene_states[1], identity)


@pytest.mark.parametrize("name", ["Circle", "np", "self", "_kd1_object_1"])
def test_compiler_preserves_contract_by_rejecting_export_name_conflicts(name):
    task = next(task for task in fixed_tasks() if task.task_id == "continuity_export")
    program = SceneProgram.model_validate(task.program)
    obj = validated_copy(program.objects[0], update={"variable_name": name})
    program = validated_copy(program, update={"objects": [obj]})
    with pytest.raises(ProgramCompileError, match=r"reserved|shadows"):
        compile_program(program, task_plan(task))


@pytest.mark.parametrize("name", ["Circle", "np", "self", "_kd1_scene"])
def test_compiler_rejects_class_names_that_shadow_runtime_symbols(name):
    task = next(task for task in fixed_tasks() if task.task_id == "geometry_circle")
    program = SceneProgram.model_validate(task.program).model_copy(update={"class_name": name})
    with pytest.raises(ProgramCompileError, match="Class name shadows"):
        compile_program(program, task_plan(task))


def test_nested_model_update_and_service_entry_revalidate_geometry():
    task = next(task for task in fixed_tasks() if task.task_id == "geometry_circle")
    program = SceneProgram.model_validate(task.program)
    with pytest.raises(ValidationError):
        program.objects[0].model_copy(update={"radius": -2})
    object.__setattr__(program.objects[0], "radius", -3)
    with pytest.raises(ValidationError):
        compile_program(program, task_plan(task))


@pytest.mark.parametrize(
    "code,valid",
    [
        ("import manim as m\nm.InventedShape()", False),
        ("from manim import *\ndef unrelated():\n InventedShape=object\nInventedShape()", False),
        ("from manim import Circle as C\nC(1,2,3,4,5)", False),
        ("import manim as m\ndef f(m):\n m.InventedShape()", True),
        ("from manim import *\ndef f(InventedShape):\n InventedShape()", True),
        ("from manim import *\n[InventedShape() for InventedShape in []]", True),
        (
            "from manim import *\ndef f():\n try: pass\n except Exception as InventedShape: InventedShape()",
            True,
        ),
        (
            "from manim import Circle\ndef outer():\n Circle=lambda:None\n def inner():\n  nonlocal Circle\n  Circle()",
            True,
        ),
    ],
)
def test_scope_and_static_signature_resolution(code, valid):
    assert lint_manim_api(code, manim_version="0.20.1").is_valid is valid
    assert lint_manim_api(code, manim_version="999.0").is_valid


def test_total_deadline_includes_waiting_for_model_slot(monkeypatch, tmp_path):
    orchestrator, _ = workflow(tmp_path)
    clock = [0.0]
    orchestrator.control = RunControl(threading.Event(), clock=lambda: clock[0])
    monkeypatch.setattr(settings, "LLM_TIMEOUT_TOTAL", 1)

    class Busy:
        def acquire(self, *, timeout):
            clock[0] += timeout
            return False

    orchestrator._llm_sem = Busy()
    with pytest.raises(OperationDeadlineExceeded), orchestrator._llm_slot():
        pytest.fail("expired queued request reached transport")


def test_retry_backoff_is_interrupted_without_refunding_request():
    cancelled = threading.Event()
    control = RunControl(cancelled)
    cancelled.set()
    with pytest.raises(RunCancelled):
        control.wait(30)


def test_clients_share_only_matching_profiles_and_close_once():
    closed = []
    pool = ClientPool()

    def factory():
        return SimpleNamespace(close=lambda: closed.append(1))

    with ThreadPoolExecutor(max_workers=4) as workers:
        clients = list(workers.map(lambda _: pool.get(("main", "endpoint"), factory), range(30)))
    assert len({id(client) for client in clients}) == 1
    assert pool.get(("visual", "endpoint"), factory) is not clients[0]
    pool.close()
    pool.close()
    assert closed == [1, 1]


def test_shared_client_retry_does_not_close_pool_client(monkeypatch):
    clients = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.closed = False
            clients.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr("kd1_anime.agents.base.OpenAI", FakeClient)
    profile = replace(settings.main_llm_profile(), api_key="test", trust_env=True)
    with client_scope():
        first, second = BaseAgent(profile), BaseAgent(profile)
        shared = first.client
        assert second.client is shared
        first.close()
        assert not shared.closed
        assert second.client is shared
    assert shared.closed
    with client_scope():
        assert second.client is not shared
    assert len(clients) == 2


def test_ui_overflow_resynchronizes_even_during_close():
    entered = threading.Event()
    release = threading.Event()
    delivered = []

    def callback(name, data):
        if name == "block":
            entered.set()
            release.wait(3)
        delivered.append(name)

    def resync():
        buffer("latest_snapshot", {"manifest_revision": 9})

    with BufferedCallback(callback, resync=resync, capacity=2) as buffer:
        buffer("block", {})
        assert entered.wait(2)
        for index in range(20):
            buffer("scene_finished", {"scene_id": index})
        assert len(buffer.queue) == 2
        release.set()
    assert delivered[-1] == "latest_snapshot"


def test_ui_callback_exception_is_isolated():
    CallbackAdapter(lambda *a: (_ for _ in ()).throw(RuntimeError("UI"))).publish(
        PipelineEvent("scene_rendered", {})
    )


def test_fifo_and_context_pool_bound_threads_and_preserve_control():
    queue = ReadyQueue()
    identity = WorkIdentity(1, "a" * 64)
    for sid in [3, 1, 2, 1]:
        queue.push(WorkItem(sid, "code", identity))
    assert [queue.pop().scene_id for _ in range(3)] == [3, 1, 2]
    control = RunControl(threading.Event())
    token = CURRENT_CONTROL.set(control)
    try:
        with ContextPool(max_workers=2) as pool:
            futures = [
                pool.submit(lambda: (threading.get_ident(), CURRENT_CONTROL.get()))
                for _ in range(100)
            ]
            results = [future.result() for future in futures]
        assert len({tid for tid, _ in results}) <= 2
        assert all(value is control for _, value in results)
    finally:
        CURRENT_CONTROL.reset(token)


def test_seeded_revision_sequence_keeps_all_old_sources_restorable(tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    rng = random.Random(7047)
    references = {}
    for index in range(20):
        code = CODE + f"# revision {rng.randrange(100000)}\n"
        ctx.scene_states[1].code = code
        orchestrator._checkpoint(ctx, State.CODING)
        manifest = RunRepository(tmp_path).load_for_resume(ctx.paths.run_id)
        references[manifest.scenes[1].code_file] = code
        restored = Orchestrator._context_from_manifest(manifest, ctx.paths.root)
        assert restored.scene_states[1].code == code
        assert manifest.revision == index + 1
    assert all(
        (ctx.paths.root / relative).read_text() == code for relative, code in references.items()
    )


def test_dependency_hashes_are_cached_for_unchanged_revisions(tmp_path):
    orchestrator, ctx = workflow(tmp_path)
    for sid in range(2, 51):
        ctx.scene_states[sid] = SceneState(plan(sid, dependencies_complete=True))
    orchestrator._checkpoint(ctx, State.CODING)
    count = orchestrator.dependency_digest_computations
    orchestrator._checkpoint(ctx, State.CODING)
    assert orchestrator.dependency_digest_computations == count
    ctx.scene_states[25].code = CODE
    orchestrator._checkpoint(ctx, State.CODING)
    assert orchestrator.dependency_digest_computations == count + 3


def test_stage_draft_is_invisible_until_manifest_publication(monkeypatch, tmp_path):
    from kd1_anime.run_store import write_manifest
    from kd1_anime.stage_transactions import CURRENT_TRANSACTION, StageTransaction

    orchestrator, ctx = workflow(tmp_path)
    orchestrator._checkpoint(ctx, State.CODING)
    identity = orchestrator._work_identity(ctx, ctx.scene_states[1])
    transaction = StageTransaction.capture(ctx, 1, identity)
    transaction.context.scene_states[1].code = CODE
    assert not ctx.scene_states[1].code

    def save(path, manifest):
        assert not ctx.scene_states[1].code
        return write_manifest(path, manifest)

    monkeypatch.setattr("kd1_anime.orchestrator.write_manifest", save)
    token = CURRENT_TRANSACTION.set(transaction)
    try:
        orchestrator._checkpoint(transaction.context, State.CODING)
    finally:
        CURRENT_TRANSACTION.reset(token)
    assert ctx.scene_states[1].code == CODE
    assert RunRepository(tmp_path).load(ctx.paths.run_id).scenes[1].code_sha256


def test_stage_commit_keeps_transport_reservations_and_counters(tmp_path):
    from kd1_anime.domain.execution import StageAttempts
    from kd1_anime.stage_transactions import CURRENT_TRANSACTION, StageTransaction

    orchestrator, ctx = workflow(tmp_path)
    orchestrator._checkpoint(ctx, State.CODING)
    identity = orchestrator._work_identity(ctx, ctx.scene_states[1])
    transaction = StageTransaction.capture(ctx, 1, identity)
    ctx.scene_states[1].stage_attempts = [
        StageAttempts(operation="generation", input_sha256="a" * 64, used=1)
    ]
    orchestrator._request_context(1).reserve("main", "fake")
    transaction.context.scene_states[1].code = CODE
    token = CURRENT_TRANSACTION.set(transaction)
    try:
        orchestrator._checkpoint(transaction.context, State.CODING)
    finally:
        CURRENT_TRANSACTION.reset(token)
    assert len(ctx.request_budget.reservations) == 1
    assert ctx.scene_states[1].stage_attempts[0].used == 1


@pytest.mark.parametrize("cancelled", [True, False])
def test_stopping_run_persists_cancellation_even_without_start_time(
    tmp_path, monkeypatch, cancelled
):
    orchestrator, ctx, state = prepare_submission(tmp_path)
    state.slurm_job = submitted_job(ctx, state)
    orchestrator._checkpoint(ctx, State.MONITORING)
    monkeypatch.setattr(orchestrator.slurm, "cancel_job", lambda job_id: cancelled)
    orchestrator.cancel_all()
    stored = RunRepository(tmp_path).load_for_resume(ctx.paths.run_id).scenes[1].slurm_job
    assert stored.started_at is None
    assert stored.status == ("CANCELLED" if cancelled else "CANCEL_FAILED")
    assert state.slurm_job.status == stored.status


def test_monitor_cancellation_observation_is_detached_and_checkpointed(tmp_path):
    orchestrator, ctx, state = prepare_submission(tmp_path)
    job = submitted_job(ctx, state)
    job.submitted_at = time.time()
    state.slurm_job = job
    orchestrator._checkpoint(ctx, State.MONITORING)
    updates = []

    def update(observed):
        orchestrator._checkpoint_slurm_job_update(ctx, observed)
        updates.append(observed.status)
        observed.status = "callback-owned-copy"

    backend = SimpleNamespace(
        poll=lambda ids: PollObservation(dict.fromkeys(ids, "PENDING"), {}),
        cancel_job=lambda job_id: False,
    )
    coordinator = RenderMonitorCoordinator(backend, poll_interval=60, on_job_update=update)
    try:
        from copy import deepcopy

        coordinator.register(deepcopy(job))
        coordinator.cancel_pending(reason="test cancellation failure")
        assert coordinator.wait(job.job_id) is False
        assert coordinator.observed_job(job.job_id).status == "CANCEL_FAILED"
    finally:
        coordinator.close()
    assert updates[-1] == "CANCEL_FAILED"
    stored = RunRepository(tmp_path).load_for_resume(ctx.paths.run_id).scenes[1].slurm_job
    assert stored.status == "CANCEL_FAILED"


@pytest.mark.parametrize("stop", ["cancel", "deadline", "response"])
def test_transport_rechecks_control_after_durable_reservation(stop):
    cancelled = threading.Event()
    clock = [0.0]
    control = RunControl(cancelled, deadline=10, clock=lambda: clock[0])
    reserved = []
    sent = []

    def reserve(*args, **kwargs):
        receipt = RequestReservation(stage="test", profile="main", model="fake")
        reserved.append(receipt)
        if stop == "cancel":
            cancelled.set()
        elif stop == "deadline":
            clock[0] = 10
        return receipt

    def create(**kwargs):
        sent.append(1)
        cancelled.set()
        return SimpleNamespace(usage=None)

    agent = BaseAgent()
    agent.request_context = RequestContext(reserve, lambda *args: None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    token = CURRENT_CONTROL.set(control)
    try:
        with pytest.raises(OperationDeadlineExceeded if stop == "deadline" else RunCancelled):
            agent._create_completion(client)
    finally:
        CURRENT_CONTROL.reset(token)
    assert len(reserved) == 1
    assert len(sent) == int(stop == "response")


def test_transport_timeout_uses_time_remaining_after_reservation():
    import httpx

    clock = [0.0]
    control = RunControl(threading.Event(), deadline=10, clock=lambda: clock[0])
    options = []

    def reserve(*args, **kwargs):
        clock[0] = 3
        return RequestReservation(stage="test", profile="main", model="fake")

    client = SimpleNamespace(
        timeout=httpx.Timeout(40, connect=2),
        with_options=lambda **kwargs: options.append(kwargs) or client,
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(usage=None))
        ),
    )
    agent = BaseAgent()
    agent.request_context = RequestContext(reserve, lambda *args: None)
    token = CURRENT_CONTROL.set(control)
    try:
        agent._create_completion(client)
    finally:
        CURRENT_CONTROL.reset(token)
    assert options[0]["timeout"].read == 7
    assert options[0]["timeout"].connect == 2


def test_logical_deadline_applies_under_an_inherited_run_control():
    from kd1_anime.control import controlled_call

    clock = [0.0]
    parent = RunControl(threading.Event(), clock=lambda: clock[0])

    class Call:
        profile = SimpleNamespace(timeout_total=1)

        @controlled_call
        def execute(self):
            clock[0] = 2

    token = CURRENT_CONTROL.set(parent)
    try:
        with pytest.raises(OperationDeadlineExceeded):
            Call().execute()
    finally:
        CURRENT_CONTROL.reset(token)


@pytest.mark.parametrize(
    "updates", [{"SLURM_PARTITION": "x\n#SBATCH --bad"}, {"LOCAL_RENDER_MAX_IN_FLIGHT": -1}]
)
def test_execution_contract_revalidates_resource_values(updates):
    spec = ExecutionSpec.capture()
    values = json.loads(spec.resources_json)
    values.update(updates)
    with pytest.raises(ValidationError):
        spec.model_copy(update={"resources_json": json.dumps(values)})


@pytest.mark.parametrize("argument,valid", [("2.0", True), ("'large'", False), ("unknown", True)])
def test_api_signature_checks_only_statically_known_scalar_arguments(argument, valid):
    code = "from manim import Circle\nCircle(radius=" + argument + ")"
    assert lint_manim_api(code, manim_version="0.20.1").is_valid is valid


def test_rag_cached_norms_and_stable_ties_require_one_norm_per_query(tmp_path, monkeypatch):
    import math

    from kd1_anime.rag.chunker import SourceChunk
    from kd1_anime.rag.store import RagIndex

    chunks = [
        SourceChunk(
            path=tmp_path / f"doc-{i}.md",
            source_kind="manim_doc",
            text=f"doc {i}",
            ordinal=0,
            source_sha256="a" * 64,
            metadata={},
        )
        for i in range(10)
    ]
    index = RagIndex(tmp_path / "index.sqlite3")
    RagIndex.build(index.path, chunks, [[1, 0]] * len(chunks), embedding_model="fake")
    snapshot = index.load_verified()
    calls = []
    original = math.sqrt
    monkeypatch.setattr(math, "sqrt", lambda value: calls.append(value) or original(value))
    for _ in range(3):
        found = index.search_verified([1, 0], top_k=3, info=snapshot.info, snapshot=snapshot)
        assert [item.chunk.chunk_id for item in found] == sorted(
            chunk.chunk_id for chunk in snapshot.chunks
        )[:3]
    assert len(calls) == 3


@pytest.mark.parametrize("cancelled", [True, False])
@pytest.mark.parametrize("identity", ["code_sha256", "execution_sha256"])
def test_unexpected_submitted_identity_cannot_publish_or_repeat(
    tmp_path, monkeypatch, cancelled, identity
):
    orchestrator, ctx, state = prepare_submission(tmp_path)
    job = submitted_job(ctx, state)
    setattr(job, identity, "b" * 64)
    monkeypatch.setattr(orchestrator.slurm, "submit_scene", lambda *args, **kwargs: job)
    monkeypatch.setattr(orchestrator.slurm, "cancel_job", lambda job_id: cancelled)
    events = []
    orchestrator._callback = lambda name, data: events.append(name)
    orchestrator._scene_submit(ctx, 1, state)
    assert state.slurm_job is None
    assert state.submission.status == "pending"
    assert "scene_submitted" not in events
    assert job.status == ("CANCELLED" if cancelled else "CANCEL_FAILED")
    assert orchestrator._uncommitted_jobs[job.job_id] is job
    stored = RunRepository(tmp_path).load_for_resume(ctx.paths.run_id).scenes[1]
    assert stored.slurm_job is None and stored.submission.status == "pending"
    with pytest.raises(RuntimeError, match="禁止自动重提"):
        orchestrator._scene_submit(ctx, 1, state)


@pytest.mark.parametrize("change", ["missing", "tool_version", "execution_digest", "policy"])
def test_incomplete_or_different_environment_refuses_reuse_and_preserves_video(
    tmp_path, monkeypatch, change
):
    from test_incremental import CODE as SOURCE
    from test_incremental import RUN_ID, make_base, make_paths, make_plan

    from kd1_anime.orchestrator import PipelineContext
    from kd1_anime.rendering import RenderProfile

    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    # Exercise a known version mismatch even when Manim is not installed on the test host.
    profile = RenderProfile.current().model_copy(update={"manim_version": "0.20.1"})
    base, video = make_base(tmp_path, profile)
    original = video.read_bytes()
    artifact = base.scenes[1].artifact
    spec = ExecutionSpec.capture(profile)
    if change == "missing":
        artifact.environment_fingerprint.clear()
    elif change == "tool_version":
        artifact.environment_fingerprint["manim"] = "different"
    elif change == "execution_digest":
        artifact.environment_fingerprint["execution_sha256"] = "b" * 64
    else:
        spec = spec.model_copy(update={"disable_network": not spec.disable_network})
    state = SceneState(make_plan(), code=SOURCE, class_name="Demo", reviewed=True)
    ctx = PipelineContext(
        "updated",
        paths=make_paths(tmp_path),
        scene_states={1: state},
        render_profile=profile,
        execution_spec=spec,
        incremental=True,
        base_run_id=RUN_ID,
        base_manifest=base,
    )
    Orchestrator()._apply_incremental_for_scene(ctx, 1, state)
    assert not state.rendered and state.artifact is None
    assert ctx.scenes_to_render == [1]
    assert video.read_bytes() == original
