"""恢复阶段的后端安全策略。"""

from __future__ import annotations

from collections.abc import Callable


class RecoveryService:
    """处理不需要了解具体 FSM 的恢复策略。"""

    @staticmethod
    def detach_unresumable_local_jobs(ctx, emit: Callable[..., None]) -> None:
        """丢弃无法跨进程认领的本地 Job，保留代码并让调度器重启。"""

        for scene_id, state in sorted(ctx.scene_states.items()):
            job = state.slurm_job
            if job is None or state.rendered:
                continue
            state.slurm_job = None
            if not state.failed and not state.give_up:
                state.failure_reason = (
                    f"恢复时不认领旧本地渲染任务 {job.job_id}，将使用相同代码重新启动"
                )
                state.failure_category = "infrastructure"
            emit(
                "local_job_not_resumed",
                scene_id=scene_id,
                job_id=job.job_id,
                reason="本地进程句柄未持久化，恢复时安全重启",
            )


__all__ = ["RecoveryService"]
