"""Durability, cancellation and bounded dispatch through v11 public contracts."""

import json
import os
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_agent_vertical_slice import agent_boundary as agent_boundary
from test_execution_contracts import CODE, RUN_ID

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.agents.api_linter import lint_manim_api
from kd1_anime.agents.base import BaseAgent
from kd1_anime.application import Application
from kd1_anime.clients import ClientPool, client_scope
from kd1_anime.commits import CommitCommand, RunCommitter, RunPatch
from kd1_anime.config import settings
from kd1_anime.control import CURRENT_CONTROL, OperationDeadlineExceeded, RunCancelled, RunControl
from kd1_anime.domain.dependencies import StaleWorkError, WorkIdentity
from kd1_anime.domain.program import SceneProgram
from kd1_anime.domain.runtime import ExecutionSpec, RunConfig
from kd1_anime.eval.benchmark import fixed_tasks, task_plan
from kd1_anime.events import BufferedCallback, CallbackAdapter, PipelineEvent
from kd1_anime.models import validated_copy
from kd1_anime.objects import ObjectStore
from kd1_anime.request_budget import RequestContext, RequestReservation
from kd1_anime.run_store import RunRepository, atomic_write_text, scene_snapshot
from kd1_anime.scheduling import ContextPool, ReadyQueue, WorkItem
from kd1_anime.services.program_compiler import ProgramCompileError, compile_program
from kd1_anime.services.results import run_stage


def completed_run(agent_boundary):
    boundary, dependencies = agent_boundary
    Application(runtime_factory=lambda: AgentRuntime(dependencies=dependencies)).generate(
        "Show a one-second pause", approve_plan=True
    )
    root = next((boundary.workspace / "runs").iterdir())
    return root, RunRepository(boundary.workspace).load_for_resume(root.name)


def test_typed_commit_publishes_memory_only_after_disk(monkeypatch, agent_boundary):
    from kd1_anime.run_store import write_manifest

    root, manifest = completed_run(agent_boundary)
    committer = RunCommitter(root, manifest)
    original = manifest.revision
    observed = []

    def save(path, candidate):
        assert committer.current is manifest and manifest.error == ""
        write_manifest(path, candidate)
        observed.append(RunRepository(root.parent.parent).load(root.name).error)

    monkeypatch.setattr("kd1_anime.commits.write_manifest", save)
    published = committer.commit(
        CommitCommand(original, "DONE", status="completed", patch=RunPatch(error="typed change"))
    )
    assert published is committer.current and published is not manifest
    assert published.revision == original + 1
    assert manifest.error == "" and observed == ["typed change"]


def test_failed_typed_commit_preserves_memory_and_manifest(monkeypatch, agent_boundary):
    root, manifest = completed_run(agent_boundary)
    before = (root / "manifest.json").read_bytes()
    committer = RunCommitter(root, manifest)
    monkeypatch.setattr(
        "kd1_anime.commits.write_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        committer.commit(
            CommitCommand(
                manifest.revision, "DONE", status="completed", patch=RunPatch(error="uncommitted")
            )
        )
    assert committer.current is manifest and manifest.error == ""
    assert (root / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("cut", ["source_saved", "before_replace"])
def test_process_crash_exposes_only_last_complete_snapshot(agent_boundary, cut):
    root, manifest = completed_run(agent_boundary)
    before = (root / "manifest.json").read_bytes()
    script = r"""import os,sys
from pathlib import Path
from kd1_anime.objects import ObjectStore
from kd1_anime.run_store import atomic_write_text
root=Path(sys.argv[1]); cut=sys.argv[2]
ObjectStore(root).source(1, "from manim import *\n# new immutable revision\n")
if cut == "source_saved": os._exit(73)
os.replace=lambda *a, **kw: os._exit(73)
atomic_write_text(root / "manifest.json", "not a complete manifest")
"""
    result = subprocess.run([sys.executable, "-c", script, str(root), cut], check=False)
    assert result.returncode == 73
    assert (root / "manifest.json").read_bytes() == before
    restored = RunRepository(root.parent.parent).load_for_resume(root.name)
    assert restored.revision == manifest.revision
    source = root / restored.scenes[1].code_file
    assert source.is_file()


@pytest.mark.parametrize("version", range(4, 11))
@pytest.mark.parametrize("entry", ["resume", "retry", "incremental"])
@pytest.mark.parametrize("public_boundary", ["application", "runtime"])
def test_all_legacy_execution_entries_reject_before_side_effects(
    tmp_path, monkeypatch, version, entry, public_boundary
):
    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    root = tmp_path / "runs" / RUN_ID
    root.mkdir(parents=True)
    path = root / "manifest.json"
    path.write_text(json.dumps({"schema_version": version, "run_id": RUN_ID}))
    root.chmod(0o755)
    path.chmod(0o644)
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("legacy execution touched runtime/lock/probe/backend")

    monkeypatch.setattr("kd1_anime.agent_runtime.runtime.lock_run", forbidden)
    app = Application(runtime_factory=forbidden)
    if public_boundary == "runtime":
        from kd1_anime.agent_runtime.runtime import AgentDependencies

        dependencies = AgentDependencies(
            backend_factory=forbidden, preflight=forbidden, probe=forbidden, rag_factory=forbidden
        )
        runtime = AgentRuntime(dependencies=dependencies)
        call = (
            (
                lambda: runtime.run_incremental(
                    "new goal",
                    RUN_ID,
                    base_manifest=SimpleNamespace(run_id=RUN_ID, schema_version=version),
                )
            )
            if entry == "incremental"
            else (lambda: runtime.resume(RUN_ID, retry_scene_id=1 if entry == "retry" else None))
        )
    else:
        call = {
            "resume": lambda: app.resume(RUN_ID),
            "retry": lambda: app.retry(RUN_ID, 1),
            "incremental": lambda: app.run_incremental("new goal", RUN_ID),
        }[entry]
    with pytest.raises(ValueError, match=r"只读|v11"):
        call()
    assert path.read_bytes() == before
    assert root.stat().st_mode & 0o777 == 0o755
    assert path.stat().st_mode & 0o777 == 0o644
    assert {item.name for item in root.iterdir()} == {"manifest.json"}
    assert not (tmp_path / "sessions").exists()


@pytest.mark.parametrize("version", range(4, 11))
def test_historical_manifest_remains_readable_and_exports_exact_original(agent_boundary, version):
    root, _ = completed_run(agent_boundary)
    path = root / "manifest.json"
    raw = json.loads(path.read_text())
    raw["schema_version"] = version
    raw.pop("agent")
    path.write_text(json.dumps(raw))
    path.chmod(0o644)
    before = path.read_bytes()
    repository = RunRepository(root.parent.parent)
    loaded = repository.load(root.name)
    assert loaded.schema_version == version
    assert loaded.model_dump(mode="json") == raw
    with pytest.raises(ValueError, match="只读"):
        repository.load_for_resume(root.name)
    assert path.read_bytes() == before
    assert path.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize(
    "field", ["run_config", "execution_spec", "snapshot", "projection", "stage_attempts"]
)
def test_v11_missing_or_conflicting_contracts_rejected_on_load(agent_boundary, field):
    root, manifest = completed_run(agent_boundary)
    path = root / "manifest.json"
    raw = json.loads(path.read_text())
    if field in {"run_config", "execution_spec"}:
        raw[field] = {}
    elif field in {"snapshot", "projection"}:
        raw["scenes"]["1"]["snapshot"] = scene_snapshot(manifest.scenes[1]).model_dump(mode="json")
        if field == "snapshot":
            raw["scenes"]["1"]["snapshot"]["lifecycle"]["revision"] += 1
        else:
            raw["scenes"]["1"]["plan"]["title"] = "conflicting flat projection"
    else:
        del raw["scenes"]["1"]["stage_attempts"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        RunRepository(root.parent.parent).load_for_resume(root.name)


def test_stage_result_rejects_changed_revision_and_input():
    identity = WorkIdentity(4, "a" * 64)
    result = run_stage(identity, lambda: {"accepted": True})
    assert result.accept(identity) == {"accepted": True}
    for replaced in (WorkIdentity(5, "a" * 64), WorkIdentity(4, "b" * 64)):
        with pytest.raises(StaleWorkError):
            result.accept(replaced)


def test_seeded_revision_sequence_keeps_old_immutable_sources_restorable(tmp_path):
    store = ObjectStore(tmp_path)
    rng = random.Random(7047)
    references = {}
    for _ in range(20):
        code = CODE + f"# revision {rng.randrange(100000)}\n"
        references[store.source(1, code)] = code
    assert len(references) == 20
    assert all((tmp_path / path).read_text() == code for path, code in references.items())


@pytest.mark.parametrize("cancelled", [True, False])
def test_stopping_public_run_persists_cancellation_without_start_time(
    agent_boundary, monkeypatch, cancelled
):
    boundary, dependencies = agent_boundary
    app = Application(runtime_factory=lambda: AgentRuntime(dependencies=dependencies))
    stopped = []
    monkeypatch.setattr(boundary.backend, "cancel_job", lambda job_id: cancelled)

    def stop_after_submission(manifest):
        if not stopped and manifest.scenes and manifest.scenes[1].slurm_job is not None:
            stopped.append(True)
            app.cancel_all()

    dependencies.after_commit = stop_after_submission
    assert app.generate("Show a one-second pause", approve_plan=True) is None
    root = next((boundary.workspace / "runs").iterdir())
    stored = RunRepository(boundary.workspace).load_for_resume(root.name).scenes[1].slurm_job
    assert stopped and stored.started_at is None
    assert stored.status == ("CANCELLED" if cancelled else "CANCEL_FAILED")


def test_uncertain_submission_is_durable_and_not_repeated(agent_boundary, monkeypatch):
    boundary, dependencies = agent_boundary
    app = Application(runtime_factory=lambda: AgentRuntime(dependencies=dependencies))
    calls = []

    def uncertain(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("no acknowledged external job id")

    monkeypatch.setattr(boundary.backend, "submit_scene", uncertain)
    monkeypatch.setattr(
        boundary.backend, "recover_submission", lambda *a, **kw: None, raising=False
    )
    with pytest.raises(RuntimeError):
        app.generate("Show a one-second pause", approve_plan=True)
    root = next((boundary.workspace / "runs").iterdir())
    manifest = RunRepository(boundary.workspace).load_for_resume(root.name)
    assert manifest.agent.submissions
    assert manifest.agent.submissions[-1].attempt.status in {"pending", "unknown"}
    assert (
        Application(runtime_factory=lambda: AgentRuntime(dependencies=dependencies)).resume(
            root.name
        )
        is None
    )
    assert calls == [1]


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
