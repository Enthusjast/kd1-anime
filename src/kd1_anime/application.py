"""Public animation use cases and version-aware runtime composition.

Callers provide goals, commands and callbacks, never a mutable pipeline context.
The runtime factory is the injection boundary for isolated execution backends.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from kd1_anime.cluster.render_backend import RenderBackendName
from kd1_anime.config import settings
from kd1_anime.domain.planning import ScenePlan
from kd1_anime.events import CallbackAdapter, PipelineEvent
from kd1_anime.run_store import RunRepository, atomic_write_json

if TYPE_CHECKING:
    from kd1_anime.agent_runtime.runtime import AgentRuntime
    from kd1_anime.cluster.jobs import RenderJob
    from kd1_anime.orchestrator import Orchestrator

Callback = Callable[[str, dict], None]


def _legacy_runtime() -> Orchestrator:
    from kd1_anime.orchestrator import Orchestrator

    return Orchestrator()


def _agent_runtime() -> AgentRuntime:
    from kd1_anime.agent_runtime.runtime import AgentRuntime

    return AgentRuntime()


class Application:
    """Stable use-case boundary; legacy state remains owned by its runtime."""

    def __init__(self, runtime_factory: Callable[[], Orchestrator | AgentRuntime] | None = None):
        self._runtime_factory = runtime_factory

    def _runtime(self, *, agent: bool = True):
        if self._runtime_factory is not None:
            return self._runtime_factory()
        return _agent_runtime() if agent else _legacy_runtime()

    @staticmethod
    def load_for_resume(run_id: str):
        manifest = RunRepository(settings.WORKSPACE_DIR).load(run_id)
        selected = "agent" if manifest.schema_version == 11 else "legacy"
        manifest.validate_for_resume(runtime=selected)
        if manifest.agent is not None:
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
    ) -> Path | None:
        options = {"dry_run": dry_run}
        if callback is not None:
            options["callback"] = callback
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
        runtime = self._runtime()
        generate = getattr(runtime, "generate", None) or runtime.run
        return generate(user_prompt, **options)

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
        return self._runtime(agent=False).submit_existing_scene(
            source_code, class_name, scene_id=scene_id, wait=wait, backend=backend
        )

    def resume(
        self,
        run_id: str,
        callback: Callback | None = None,
        interactive: bool = False,
    ) -> Path | None:
        repository = RunRepository(settings.WORKSPACE_DIR)
        manifest = self.load_for_resume(run_id)
        options = {"callback": callback, "interactive": interactive}
        if manifest.incremental and manifest.base_run_id:
            options["base_manifest"] = repository.load_for_resume(manifest.base_run_id)
        return self._runtime(agent=manifest.schema_version == 11).resume(run_id, **options)

    def retry(
        self,
        run_id: str,
        scene_id: int,
        *,
        callback: Callback | None = None,
        interactive: bool = False,
    ) -> Path | None:
        repository = RunRepository(settings.WORKSPACE_DIR)
        manifest = self.load_for_resume(run_id)
        options = {
            "callback": callback,
            "interactive": interactive,
            "retry_scene_id": scene_id,
        }
        if manifest.incremental and manifest.base_run_id:
            options["base_manifest"] = repository.load_for_resume(manifest.base_run_id)
        return self._runtime(agent=manifest.schema_version == 11).resume(run_id, **options)

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
        base = RunRepository(settings.WORKSPACE_DIR).load_for_resume(base_run_id)
        return self._runtime(agent=False).run_incremental(
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
