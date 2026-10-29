"""Backend-neutral batched monitoring and timeout/cancellation coordination."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

from rich.console import Console

from kd1_anime.cluster.adapters import adapt_backend
from kd1_anime.cluster.jobs import RenderJob
from kd1_anime.cluster.status import FAILURE_STATES, QUEUED_STATES, RUNNING_STATES, TERMINAL_STATES
from kd1_anime.config import settings

if TYPE_CHECKING:
    from kd1_anime.cluster.render_backend import RenderBackend
console = Console()


def _dashboard_quiet():
    from kd1_anime.dashboard import quiet

    return quiet()


class JobMonitor:
    """可增量推进的渲染作业监控器（非阻塞轮询）。

    与 wait_for_all_jobs 不同, 它把"轮询一次"与"等待"解耦:
    - add_job(): 加入新提交的作业 (调度器可以边提交边监控)
    - poll_once(): 推进一次轮询, 返回本轮是否有作业结束
    - pending/results: 查询未结束/已结束作业

    供场景级调度使用，让渲染与其他场景的 LLM 阶段并行推进。
    """

    def __init__(
        self,
        dispatcher: RenderBackend,
        *,
        queue_timeout: int | None = None,
        run_timeout: int | None = None,
        unknown_timeout: int | None = None,
        artifact_grace: int | None = None,
        max_unknown: int | None = None,
        poll_interval: int | None = None,
        use_legacy_timeout: bool = True,
    ) -> None:
        self.dispatcher = adapt_backend(dispatcher)
        self.queue_timeout = (
            settings.MONITOR_QUEUE_TIMEOUT if queue_timeout is None else queue_timeout
        )
        self.run_timeout = settings.MONITOR_RUN_TIMEOUT if run_timeout is None else run_timeout
        self.unknown_timeout = (
            settings.MONITOR_UNKNOWN_TIMEOUT if unknown_timeout is None else unknown_timeout
        )
        self.artifact_grace = (
            settings.MONITOR_ARTIFACT_GRACE if artifact_grace is None else artifact_grace
        )
        self.max_unknown = settings.MONITOR_MAX_UNKNOWN if max_unknown is None else max_unknown
        self.poll_interval = (
            settings.MONITOR_POLL_INTERVAL if poll_interval is None else poll_interval
        )
        # 兼容只设置旧 MONITOR_TIMEOUT 的配置；显式修改新的拆分项时以新项为准。
        legacy_timeout = settings.MONITOR_TIMEOUT
        if use_legacy_timeout and legacy_timeout is not None:
            if self.queue_timeout == 3600:
                self.queue_timeout = legacy_timeout
            if self.run_timeout == 3600:
                self.run_timeout = legacy_timeout
            if self.unknown_timeout == 300:
                self.unknown_timeout = legacy_timeout
        self.pending: dict[str, RenderJob] = {}
        self.results: dict[str, bool] = {}
        self.jobs: dict[str, RenderJob] = {}
        self.indeterminate_streaks: dict[str, int] = {}
        self.indeterminate_since: dict[str, float] = {}
        self.gone_since: dict[str, float] = {}
        self.completed_since: dict[str, float] = {}
        self.running_since: dict[str, float] = {}
        self.queued_since: dict[str, float] = {}
        self.log_positions: dict[str, int] = {}

    def add_job(self, job: RenderJob) -> None:
        if job.job_id in self.results:
            return
        self.pending.setdefault(job.job_id, job)
        self.jobs[job.job_id] = job
        self.indeterminate_streaks.setdefault(job.job_id, 0)
        if job.status == "COMPLETED":
            self.completed_since.setdefault(job.job_id, time.time())
        # resume 时清单可能已经记录作业处于 RUNNING；保守地从已知启动/提交
        # 时间计时，避免恢复后把一项运行很久的作业重新从零开始计时。
        if job.status in RUNNING_STATES:
            self.running_since.setdefault(job.job_id, job.started_at or job.submitted_at)
        elif job.status not in TERMINAL_STATES:
            self.queued_since.setdefault(job.job_id, job.submitted_at)

    def _quiet(self) -> bool:
        """Live 仪表盘激活时抑制 Monitor 文本输出, 避免破坏 Rich Live。"""
        return _dashboard_quiet()

    def poll_once(self) -> bool:
        """单次轮询所有 pending 作业, 更新状态; 返回本轮是否有作业结束。"""
        if not self.pending:
            return False
        now = time.time()
        try:
            snapshot = self.dispatcher.poll(list(self.pending))
            statuses = snapshot.statuses
            start_times = snapshot.start_times
            diagnostic = snapshot.diagnostic
        except Exception as exc:
            # 查询命令的瞬时异常不能等同于远端作业失败，更不能立即取消
            # 所有健康作业。把本轮当作 UNKNOWN，沿用统一的次数/时间宽限；
            # 若下一轮恢复为 RUNNING/PENDING，连续未知计数会自动清零。
            diagnostic = f"Slurm 状态查询异常: {exc}"
            quiet = self._quiet()
            finished: list[str] = []
            for job_id, job in self.pending.items():
                job.status = "UNKNOWN"
                self.indeterminate_streaks[job_id] += 1
                unknown_since = self.indeterminate_since.setdefault(job_id, now)
                if (
                    self.indeterminate_streaks[job_id] >= self.max_unknown
                    and now - unknown_since >= self.unknown_timeout
                ):
                    reason = f"状态连续未知，已停止监控并尝试取消远端任务；查询诊断: {diagnostic}"
                    self.dispatcher.cancel_for_monitor_failure(
                        job,
                        status="UNKNOWN_TIMEOUT",
                        reason=reason,
                    )
                    self.results[job_id] = False
                    finished.append(job_id)
                elif not quiet:
                    console.print(
                        f"[dim][Monitor][/] Scene {job.scene_id}: UNKNOWN "
                        f"({self.indeterminate_streaks[job_id]}/{self.max_unknown})"
                    )
            for job_id in finished:
                self.pending.pop(job_id, None)
            return bool(finished)
        finished: list[str] = []
        quiet = self._quiet()
        for job_id, job in self.pending.items():
            status = statuses.get(job_id, "UNKNOWN")
            job.status = status
            if job_id in start_times:
                job.started_at = start_times[job_id]

            if status == "COMPLETED":
                self.indeterminate_streaks[job_id] = 0
                self.indeterminate_since.pop(job_id, None)
                self.gone_since.pop(job_id, None)
                completed_at = self.completed_since.setdefault(job_id, now)
                job.elapsed_seconds = max(
                    0.0, now - self.running_since.get(job_id, job.submitted_at)
                )
                self.dispatcher.forward_log(job, self.log_positions)
                valid = self.dispatcher.validate_completed_job(job)
                if not valid and now - completed_at < self.artifact_grace:
                    # Slurm 状态已经完成，但共享文件系统上的最终文件可能还未
                    # 传播，或文件仍在由 ffmpeg 关闭。保留 pending，下一轮重验。
                    if not quiet:
                        console.print(
                            f"[dim][Monitor][/] Scene {job.scene_id} 已完成，等待最终 MP4 同步"
                        )
                    continue
                self.results[job_id] = valid
                finished.append(job_id)
                if not valid:
                    job.status = "FAILED"
                if not quiet and valid:
                    console.print(f"[bold green][Monitor][/] Scene {job.scene_id} 渲染成功")
                elif not quiet:
                    console.print(
                        f"[bold red][Monitor][/] Scene {job.scene_id} 作业结束但产物无效: "
                        f"{job.failure_reason}",
                        markup=False,
                    )
            elif status in FAILURE_STATES:
                self.indeterminate_streaks[job_id] = 0
                self.gone_since.pop(job_id, None)
                self.dispatcher.forward_log(job, self.log_positions)
                job.failure_reason = f"Slurm 状态: {status}"
                self.results[job_id] = False
                finished.append(job_id)
                if not quiet:
                    console.print(f"[bold red][Monitor][/] Scene {job.scene_id} 渲染失败: {status}")
            elif status in ("UNKNOWN", "GONE"):
                if status == "GONE":
                    # 作业已确认不在调度器: 依据产物判定结果, 避免秒退/刚结束的作业被误杀。
                    gone_at = self.gone_since.setdefault(job_id, now)
                    outcome = self.dispatcher.classify_gone(job)
                    if outcome == "COMPLETED":
                        self.indeterminate_streaks[job_id] = 0
                        self.gone_since.pop(job_id, None)
                        job.status = "COMPLETED"
                        job.elapsed_seconds = max(0.0, now - job.submitted_at)
                        self.dispatcher.forward_log(job, self.log_positions)
                        self.results[job_id] = True
                        finished.append(job_id)
                        if not quiet:
                            console.print(f"[bold green][Monitor][/] Scene {job.scene_id} 渲染成功")
                        continue
                    if outcome == "FAILED":
                        self.indeterminate_streaks[job_id] = 0
                        self.gone_since.pop(job_id, None)
                        job.status = "FAILED"
                        self.dispatcher.forward_log(job, self.log_positions)
                        self.results[job_id] = False
                        finished.append(job_id)
                        if not quiet:
                            console.print(
                                f"[bold red][Monitor][/] Scene {job.scene_id} 渲染失败（作业已消失，依据日志判定）"
                            )
                        continue
                    # 作业消失与 COMPLETED 一样可能只是共享文件系统尚未同步。
                    # 不能依赖 completed_since：sacct 尚未返回终态时该字段为空，
                    # 旧逻辑会在 max_unknown 次轮询后过早判死。
                    if now - gone_at < self.artifact_grace:
                        if not quiet:
                            console.print(
                                f"[dim][Monitor][/] Scene {job.scene_id} 已结束，等待最终 MP4 同步"
                            )
                        continue
                else:
                    self.gone_since.pop(job_id, None)
                self.indeterminate_streaks[job_id] += 1
                self.indeterminate_since.setdefault(job_id, now)
                if self.indeterminate_streaks[job_id] >= self.max_unknown:
                    if status == "GONE":
                        # 作业消失且无日志 → 按失败交给修复流程, 而非永久判死
                        job.status = "FAILED"
                        job.failure_reason = "作业已从集群消失且无输出日志，无法确认渲染结果"
                        self.results[job_id] = False
                        finished.append(job_id)
                        if not quiet:
                            console.print(
                                f"[bold red][Monitor][/] Scene {job.scene_id} 渲染失败：作业消失且无输出"
                            )
                    else:
                        unknown_since = self.indeterminate_since[job_id]
                        if now - unknown_since < self.unknown_timeout:
                            if not quiet:
                                console.print(
                                    f"[dim][Monitor][/] Scene {job.scene_id}: UNKNOWN "
                                    f"({int(now - unknown_since)}s/{self.unknown_timeout}s)"
                                )
                            continue
                        reason = "状态连续未知，已停止监控并尝试取消远端任务"
                        if diagnostic:
                            reason += f"；查询诊断: {diagnostic}"
                        self.dispatcher.cancel_for_monitor_failure(
                            job,
                            status="UNKNOWN_TIMEOUT",
                            reason=reason,
                        )
                        self.results[job_id] = False
                        finished.append(job_id)
            else:
                self.indeterminate_streaks[job_id] = 0
                self.indeterminate_since.pop(job_id, None)
                self.gone_since.pop(job_id, None)
                self.completed_since.pop(job_id, None)
                if status in RUNNING_STATES:
                    self.queued_since.pop(job_id, None)
                    self.running_since.setdefault(job_id, job.started_at or now)
                elif job_id in self.running_since:
                    # 作业被抢占退回排队等非运行状态: 停止累计运行时长,
                    # 否则之前累计的 run_timeout 会错误地继续计时
                    self.running_since.pop(job_id, None)
                    self.queued_since[job_id] = now
                elif status in QUEUED_STATES:
                    self.queued_since.setdefault(job_id, now)
                if job_id in self.running_since:
                    self.dispatcher.forward_log(job, self.log_positions)
                    if now - self.running_since[job_id] > self.run_timeout:
                        self.dispatcher.cancel_for_monitor_failure(
                            job,
                            status="RUN_TIMEOUT",
                            reason=f"运行超过 {self.run_timeout} 秒",
                        )
                        self.results[job_id] = False
                        finished.append(job_id)
                        continue
                elif now - self.queued_since.get(job_id, job.submitted_at) > self.queue_timeout:
                    self.dispatcher.cancel_for_monitor_failure(
                        job,
                        status="QUEUE_TIMEOUT",
                        reason=f"排队超过 {self.queue_timeout} 秒",
                    )
                    self.results[job_id] = False
                    finished.append(job_id)
                    continue
                if not quiet:
                    console.print(f"[dim][Monitor][/] Scene {job.scene_id}: {status}")

        for job_id in finished:
            self.pending.pop(job_id, None)
        return bool(finished)


class RenderMonitorCoordinator:
    """在一个后台线程中批量监控同一运行的所有 Slurm Job。

    场景 worker 只负责注册 Job 和等待结果，不再各自创建 JobMonitor。
    这样既避免重复查询 Slurm 控制面，也让每个 Job 的未知/宽限/超时
    计数只存在一份。
    """

    def __init__(
        self,
        dispatcher: RenderBackend,
        *,
        queue_timeout: int | None = None,
        run_timeout: int | None = None,
        unknown_timeout: int | None = None,
        artifact_grace: int | None = None,
        max_unknown: int | None = None,
        poll_interval: int | None = None,
        on_job_update: Callable[[RenderJob], None] | None = None,
    ) -> None:
        self.monitor = JobMonitor(
            dispatcher,
            queue_timeout=queue_timeout,
            run_timeout=run_timeout,
            unknown_timeout=unknown_timeout,
            artifact_grace=artifact_grace,
            max_unknown=max_unknown,
            poll_interval=poll_interval,
            use_legacy_timeout=False,
        )
        self._condition = threading.Condition()
        self._monitor_lock = threading.RLock()
        self._queued: list[RenderJob] = []
        self._on_job_update = on_job_update
        self._last_observations: dict[str, RenderJob] = {}
        self._registered: set[str] = set()
        self._events: dict[str, threading.Event] = {}
        self._outcomes: dict[str, bool] = {}
        self._closed = False
        self._thread = threading.Thread(
            target=__import__("contextvars").copy_context().run,
            args=(self._run,),
            name="slurm-monitor",
            daemon=True,
        )
        self._thread.start()

    def register(self, job: RenderJob) -> None:
        """注册一个新 Job；重复注册同一 Job ID 是幂等操作。"""

        with self._condition:
            if self._closed:
                raise RuntimeError("Slurm 监控协调器已关闭")
            if job.job_id in self._registered:
                return
            self._registered.add(job.job_id)
            self._events[job.job_id] = threading.Event()
            self._queued.append(job)
            self._condition.notify_all()

    def observed_job(self, job_id):
        """Return a detached observation; callers commit it before publishing."""
        with self._monitor_lock:
            return deepcopy(self.monitor.jobs.get(job_id))

    def outcome(self, job_id):
        with self._condition:
            return self._outcomes.get(job_id)

    def finished(self, job_id):
        with self._condition:
            return job_id in self._outcomes

    def wait(
        self,
        job_id: str,
        *,
        stop_event: threading.Event | None = None,
    ) -> bool | None:
        """等待 Job 结果；外部流水线停止时返回 None，不篡改 Job 结果。"""

        with self._condition:
            event = self._events.get(job_id)
            if event is None:
                raise KeyError(f"未注册 Slurm Job: {job_id}")
        while not event.wait(0.25):
            if stop_event is not None and stop_event.is_set():
                return None
        with self._condition:
            return self._outcomes.get(job_id)

    def close(self, *, timeout: float = 30.0) -> None:
        """停止接收新 Job，并等待当前轮询尽量完成。"""

        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, timeout))

    def cancel_pending(self, *, reason: str = "流水线停止") -> None:
        """取消仍由协调器持有的作业，避免停止后遗留后台轮询线程。"""

        with self._monitor_lock:
            with self._condition:
                queued = self._queued
                self._queued = []
            pending = [*self.monitor.pending.values(), *queued]
            seen: set[str] = set()
            for job in pending:
                if job.job_id in seen:
                    continue
                seen.add(job.job_id)
                if not job.cancelled and job.status not in TERMINAL_STATES:
                    try:
                        cancelled = self.monitor.dispatcher.cancel_job(job.job_id)
                    except Exception as exc:
                        cancelled = False
                        job.failure_reason = f"{reason}；取消作业时发生异常: {exc}"
                    if cancelled:
                        job.cancelled = True
                        job.status = "CANCELLED"
                        job.failure_reason = f"{reason}，已取消远端作业"
                    else:
                        job.status = "CANCEL_FAILED"
                        if not job.failure_reason:
                            job.failure_reason = f"{reason}；取消远端作业失败"
                self.monitor.jobs[job.job_id] = job
                self.monitor.results[job.job_id] = False
                self.monitor.pending.pop(job.job_id, None)
            self._publish_results()

    def _drain_queue(self) -> None:
        with self._condition:
            queued = self._queued
            self._queued = []
        for job in queued:
            self.monitor.add_job(job)

    def _publish_results(self) -> None:
        if self._on_job_update is not None:
            for job_id, job in self.monitor.jobs.items():
                # Elapsed time is diagnostic. Persist identity, status and
                # cancellation changes even when a job never acquired a start time.
                observation = deepcopy(replace(job, elapsed_seconds=None))
                if observation == self._last_observations.get(job_id):
                    continue
                try:
                    self._on_job_update(deepcopy(job))
                    self._last_observations[job_id] = observation
                except Exception as exc:
                    # 检查点回调失败不应杀死监控线程并让 worker 永久等待；
                    # 交给外层通过 checkpoint_error/stop_event 收尾。
                    if not _dashboard_quiet():
                        console.print(
                            f"[yellow][Monitor][/] 更新作业检查点失败: {exc}",
                            markup=False,
                        )
        with self._condition:
            for job_id, outcome in self.monitor.results.items():
                if job_id in self._outcomes:
                    continue
                self._outcomes[job_id] = outcome
                event = self._events.get(job_id)
                if event is not None:
                    event.set()

    def _run(self) -> None:
        while True:
            with self._monitor_lock:
                self._drain_queue()
                has_pending = bool(self.monitor.pending)
            with self._condition:
                if self._closed and not self._queued and not has_pending:
                    return
                if not has_pending:
                    self._condition.wait()
                    continue
            with self._monitor_lock:
                self.monitor.poll_once()
                self._publish_results()
                has_pending = bool(self.monitor.pending)
            with self._condition:
                if has_pending:
                    self._condition.wait(timeout=self.monitor.poll_interval)
