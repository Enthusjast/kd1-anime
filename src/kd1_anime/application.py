"""Public animation use cases composed exclusively with the agent runtime.

Callers provide goals, commands and callbacks, never a mutable pipeline context.
The runtime factory is the injection boundary for isolated execution backends.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from kd1_anime.cluster.render_backend import RenderBackendName
from kd1_anime.config import settings
from kd1_anime.domain.dependencies import WorkIdentity
from kd1_anime.domain.planning import ScenePlan
from kd1_anime.events import CallbackAdapter, PipelineEvent
from kd1_anime.run_store import RunRepository, atomic_write_json

if TYPE_CHECKING:
    from kd1_anime.agent_runtime.runtime import AgentRuntime
    from kd1_anime.cluster.jobs import RenderJob

Callback = Callable[[str, dict], None]


def _agent_runtime(*, resource_coordinator=None) -> AgentRuntime:
    from kd1_anime.agent_runtime.runtime import AgentRuntime

    return AgentRuntime(resource_coordinator=resource_coordinator)


class Application:
    """Stable use-case boundary; durable state remains owned by the agent runtime."""

    def __init__(
        self,
        runtime_factory: Callable[[], AgentRuntime] | None = None,
        *,
        resource_coordinator=None,
    ):
        self._runtime_factory = runtime_factory
        self._resource_coordinator = resource_coordinator
        self._active_runtime = None
        self._cancel_requested = Event()

    def _runtime(self):
        if self._runtime_factory is not None:
            runtime = self._runtime_factory()
        elif self._resource_coordinator is not None:
            runtime = _agent_runtime(resource_coordinator=self._resource_coordinator)
        else:
            runtime = _agent_runtime()
        self._active_runtime = runtime
        if self._cancel_requested.is_set():
            runtime.cancel_all()
        return runtime

    def cancel_all(self) -> None:
        self._cancel_requested.set()
        runtime = self._active_runtime
        if runtime is not None:
            runtime.cancel_all()

    @staticmethod
    def _sessions():
        from kd1_anime.agent_runtime.session import SessionStore
        from kd1_anime.config import resolve_runtime_path

        secrets = tuple(
            value
            for key, value in settings.model_dump().items()
            if key.endswith("API_KEY") and value
        )
        return SessionStore(
            resolve_runtime_path(settings.WORKSPACE_DIR) / "sessions", secrets=secrets
        )

    def create_session(self, goal: str) -> str:
        from kd1_anime.domain.runtime import RunConfig
        from kd1_anime.request_budget import RequestBudget

        config = RunConfig.capture()
        budget = RequestBudget(
            scene_limit=settings.MAX_SCENE_LLM_REQUESTS,
            run_limit=settings.MAX_RUN_LLM_REQUESTS,
            token_limit=settings.MAX_RUN_LLM_TOKENS,
            cost_limit_usd=settings.MAX_RUN_LLM_COST_USD,
        )
        return self._sessions().create(goal, draft_config=config, draft_budget=budget).session_id

    def session_request_context(self, session_id: str):
        from kd1_anime.agent_runtime.contracts import UsageReceipt
        from kd1_anime.request_budget import RequestContext, RequestReservation

        store = self._sessions()
        session = store.load(session_id)
        if session.draft_config is None:
            raise ValueError("Clarification session lacks its pinned configuration")
        config = session.draft_config.resolve()

        def reserve(profile, model, **metadata):
            prices = config.LLM_MODEL_PRICES_USD.get(model)
            prompt = metadata.get("estimated_prompt_tokens")
            output = metadata.get("reserved_output_tokens")
            cost = (
                (prompt * prices[0] + output * prices[1]) / 1_000_000
                if prices is not None and prompt is not None and output is not None
                else None
            )
            receipt = RequestReservation(
                stage="clarification",
                profile=profile,
                model=model,
                prompt_sha256=metadata.get("prompt_sha256"),
                estimated_prompt_tokens=prompt,
                reserved_output_tokens=output,
                reserved_cost_usd=cost,
                input_price_usd_per_million=prices[0] if prices else None,
                output_price_usd_per_million=prices[1] if prices else None,
                **{
                    key: metadata[key]
                    for key in ("call_id", "parent_span_id", "attempt", "json_repair_attempt")
                    if key in metadata
                },
            )
            store.reserve_draft(session_id, receipt)
            return receipt

        def emit(name, data):
            if name != "model_request_finished":
                return
            receipt = next(
                item
                for item in store.load(session_id).draft_budget.reservations
                if item.request_id == data["request_id"]
            )
            reported = data.get("usage") or {}
            tokens = reported or data.get("estimated_usage") or {}
            cost = None
            if (
                tokens.get("prompt_tokens") is not None
                and tokens.get("completion_tokens") is not None
                and receipt.input_price_usd_per_million is not None
                and receipt.output_price_usd_per_million is not None
            ):
                cost = (
                    tokens["prompt_tokens"] * receipt.input_price_usd_per_million
                    + tokens["completion_tokens"] * receipt.output_price_usd_per_million
                ) / 1_000_000
            store.complete_draft(
                session_id,
                UsageReceipt(
                    request_id=receipt.request_id,
                    prompt_sha256=receipt.prompt_sha256,
                    usage_source="provider_reported"
                    if reported
                    else "locally_estimated"
                    if tokens
                    else "unknown",
                    cost_status="estimated" if cost is not None else "unknown",
                    cost=cost,
                    **tokens,
                ),
            )

        return RequestContext(reserve, emit)

    def session_command(self, session_id: str, kind: str, content: str = "") -> None:
        from kd1_anime.agent_runtime.session import UserCommand

        if kind not in {"clarification", "ready", "confirm", "cancel"}:
            raise ValueError("Unsupported clarification command")
        self._sessions().command(
            session_id, UserCommand(session_id=session_id, kind=kind, content=content)
        )

    def run_view(self, run_id: str, *, lock_held: bool = False):
        from kd1_anime.queries import QueryService

        return QueryService(RunRepository(settings.WORKSPACE_DIR), self._sessions()).run(
            run_id, lock_held=lock_held
        )

    def approval_requests(self, run_id: str):
        from kd1_anime.queries import QueryService

        return QueryService(RunRepository(settings.WORKSPACE_DIR), self._sessions()).approvals(
            run_id
        )

    def approval_diff(self, run_id: str) -> str:
        from difflib import unified_diff

        from kd1_anime.rendering import sha256_file
        from kd1_anime.run_store import lock_run, restore_run_path

        repository = RunRepository(settings.WORKSPACE_DIR)
        repository.load_for_resume(run_id)
        with lock_run(repository.run_root(run_id)):
            manifest = repository.load_for_resume(run_id)
            action = manifest.agent.pending_approval
            if action is None or action.kind != "inspect_candidate" or action.proposal is None:
                return ""
            if action.proposal.digest() != action.input_sha256:
                raise ValueError("Pending diff identity mismatch")
            texts = []
            for reference in (action.proposal.base_source, action.proposal.candidate_source):
                path = restore_run_path(repository.run_root(run_id), reference.path)
                if (
                    reference.kind != "code"
                    or not path.is_file()
                    or sha256_file(path) != reference.sha256
                ):
                    raise ValueError("Pending diff source identity mismatch")
                texts.append(path.read_text(encoding="utf-8").splitlines(keepends=True))
            return "".join(unified_diff(texts[0], texts[1], fromfile="base", tofile="candidate"))

    def decide_approval(
        self,
        run_id: str,
        approve: bool,
        *,
        input_sha256: str,
        resource_id: str,
        work_identity: WorkIdentity | None,
    ) -> None:
        from kd1_anime.agent_runtime.session import UserCommand
        from kd1_anime.run_store import lock_run

        repository = RunRepository(settings.WORKSPACE_DIR)
        repository.load_for_resume(run_id)
        with lock_run(repository.run_root(run_id)):
            manifest = repository.load_for_resume(run_id)
            action = manifest.agent.pending_approval
            if action is None:
                raise ValueError("No pending approval for this run")
            if (
                action.input_sha256 != input_sha256
                or action.resource_id != resource_id
                or action.work_identity != work_identity
            ):
                raise ValueError("Displayed approval no longer matches the current pending action")
            command = UserCommand(
                session_id=manifest.agent.session_id,
                kind="approve" if approve else "reject",
                action=action,
            )
            store = self._sessions()
            if approve:
                store.issue_grant(manifest.agent.session_id, command)
            else:
                store.command(manifest.agent.session_id, command)

    @staticmethod
    def load_for_resume(run_id: str):
        repository = RunRepository(settings.WORKSPACE_DIR)
        manifest = repository.load_for_resume(run_id)
        from kd1_anime.prompt_identity import verify_generation_identity

        verify_generation_identity(manifest.agent.generation_identity)
        return manifest

    def generate(
        self,
        user_prompt: str,
        callback: Callback | None = None,
        dry_run: bool = False,
        interactive: bool = False,
        output_path: Path | None = None,
        approve_plan: bool = False,
        smoke: bool = False,
        backend: RenderBackendName | None = None,
        session_id: str | None = None,
    ) -> Path | None:
        run_id: str | None = None
        adapter = CallbackAdapter(callback)

        def observe(name: str, data: dict) -> None:
            nonlocal run_id
            if name == "run_started":
                run_id = data["run_id"]
            adapter.publish(PipelineEvent(name, data))

        options = {"dry_run": dry_run, "callback": observe}
        if interactive:
            options["interactive"] = True
        if output_path is not None:
            options["output_path"] = output_path
        if approve_plan:
            options["approve_plan"] = True
        if smoke:
            options["smoke"] = True
        if backend is not None:
            options["backend"] = backend
        if session_id is not None:
            options["session_id"] = session_id
        result = self._runtime().generate(user_prompt, **options)
        if result is None and run_id is not None:
            manifest = RunRepository(settings.WORKSPACE_DIR).load(run_id)
            if manifest.status == "failed" and not self._cancel_requested.is_set():
                raise RuntimeError(manifest.error or manifest.agent.stop_reason or "运行失败")
        return result

    def plan_only(
        self,
        user_prompt: str,
        callback: Callback | None = None,
        interactive: bool = False,
        output_path: Path | None = None,
        approve_plan: bool = False,
        review: bool = True,
        preflight: bool = True,
        *,
        export_path: Path | None = None,
    ) -> list[ScenePlan]:
        repository = RunRepository(settings.WORKSPACE_DIR)
        run_id: str | None = None
        adapter = CallbackAdapter(callback)

        def observe(name: str, data: dict) -> None:
            nonlocal run_id
            if name == "run_started":
                run_id = data["run_id"]
            adapter.publish(PipelineEvent(name, data))

        scenes = self._runtime().plan_only(
            user_prompt,
            callback=observe if export_path is not None else callback,
            interactive=interactive,
            output_path=output_path,
            approve_plan=approve_plan,
            review=review,
            preflight=preflight,
        )
        if export_path is not None:
            if run_id is None:
                raise RuntimeError("计划已返回，但缺少已提交运行的标识，拒绝导出")
            manifest = repository.load(run_id)
            payload = {
                "schema_version": 1,
                "run_id": manifest.run_id,
                "user_prompt": manifest.user_prompt,
                "lesson_spec": manifest.lesson_spec.model_dump(mode="json"),
                "teaching_graph": manifest.teaching_graph.model_dump(mode="json"),
                "continuity_bible": (
                    manifest.continuity_bible.model_dump(mode="json")
                    if manifest.continuity_bible is not None
                    else None
                ),
                "items": [
                    state.plan.model_dump(mode="json")
                    for _, state in sorted(manifest.scenes.items())
                ],
            }
            atomic_write_json(export_path.expanduser().resolve(), payload)
        return scenes

    def run_from_plan(
        self,
        plan_file: Path,
        *,
        callback: Callback | None = None,
        dry_run: bool = False,
        interactive: bool = False,
        output_path: Path | None = None,
        approve_plan: bool = False,
        smoke: bool = False,
        backend: RenderBackendName | None = None,
    ) -> Path | None:
        return self._runtime().run_from_plan(
            plan_file,
            callback=callback,
            dry_run=dry_run,
            interactive=interactive,
            output_path=output_path,
            approve_plan=approve_plan,
            smoke=smoke,
            backend=backend,
        )

    def submit_existing_scene(
        self,
        source_code: str,
        class_name: str,
        *,
        scene_id: int = 1,
        wait: bool = False,
        backend: RenderBackendName | None = None,
    ) -> tuple[RenderJob, Path | None, str]:
        return self._runtime().submit_existing_scene(
            source_code, class_name, scene_id=scene_id, wait=wait, backend=backend
        )

    def resume(
        self,
        run_id: str,
        callback: Callback | None = None,
        interactive: bool = False,
    ) -> Path | None:
        manifest = self.load_for_resume(run_id)
        options = {"callback": callback, "interactive": interactive}
        if (
            manifest.incremental
            and manifest.base_run_id
            and manifest.state != "DONE"
            and (
                not manifest.scenes
                or any(
                    not scene.rendered
                    and manifest.agent.incremental_checked.get(identifier) != scene.input_sha256
                    for identifier, scene in manifest.scenes.items()
                )
            )
        ):
            options["base_manifest"] = self.load_for_resume(manifest.base_run_id)
        return self._runtime().resume(run_id, **options)

    def retry(
        self,
        run_id: str,
        scene_id: int,
        *,
        callback: Callback | None = None,
        interactive: bool = False,
    ) -> Path | None:
        manifest = self.load_for_resume(run_id)
        if manifest.state == "DONE" or manifest.status == "completed":
            raise RuntimeError("Completed runs cannot retry in place; create a new incremental run")
        options = {"callback": callback, "interactive": interactive, "retry_scene_id": scene_id}
        if (
            manifest.incremental
            and manifest.base_run_id
            and (
                scene_id not in manifest.scenes
                or manifest.agent.incremental_checked.get(scene_id)
                != manifest.scenes[scene_id].input_sha256
            )
        ):
            options["base_manifest"] = self.load_for_resume(manifest.base_run_id)
        return self._runtime().resume(run_id, **options)

    def run_incremental(
        self,
        user_prompt: str,
        base_run_id: str,
        callback: Callback | None = None,
        dry_run: bool = False,
        interactive: bool = False,
        output_path: Path | None = None,
        approve_plan: bool = False,
        smoke: bool = False,
        backend: RenderBackendName | None = None,
    ) -> Path | None:
        base = self.load_for_resume(base_run_id)
        return self._runtime().run_incremental(
            user_prompt,
            base_run_id,
            base_manifest=base,
            callback=callback,
            dry_run=dry_run,
            interactive=interactive,
            output_path=output_path,
            approve_plan=approve_plan,
            smoke=smoke,
            backend=backend,
        )
