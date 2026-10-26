"""Artifact identity, log handling and cancellation policy shared by render adapters."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rich.console import Console

from kd1_anime.cluster.jobs import RenderJob
from kd1_anime.config import settings
from kd1_anime.rendering import sha256_file, verify_video

console = Console()


def _dashboard_quiet():
    from kd1_anime.dashboard import quiet

    return quiet()


class RenderArtifactValidator:
    @staticmethod
    def _verify_video(video, profile):
        return verify_video(video, profile)

    @staticmethod
    def _validate_scene_id(scene_id: int) -> int:
        if isinstance(scene_id, bool) or not isinstance(scene_id, int) or scene_id < 1:
            raise ValueError("scene_id 必须是大于 0 的整数")
        return scene_id

    @staticmethod
    def _validate_scene_class_name(scene_class_name: str) -> str:
        if not isinstance(scene_class_name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,199}", scene_class_name
        ):
            raise ValueError("Scene 类名必须是合法的 Python 标识符")
        return scene_class_name

    @staticmethod
    def _find_final_video(job: RenderJob) -> Path | None:
        """递归查找作业产出的最终渲染视频。

        manim 0.20 会把成品写到嵌套路径
        <media_dir>/videos/<源文件名>/<quality>/<SceneClass>.mp4
        (不同版本/平台层级略有差异), 因此必须递归查找, 与 VideoMerger 的
        定位逻辑保持一致; 同时排除 partial_movie_files, 并只接受 mtime
        不早于作业提交时间的产物, 避免误复用上一次修复尝试的旧视频。
        """
        if not job.media_dir.is_dir():
            return None
        candidates: list[tuple[Path, float]] = []
        try:
            paths = job.media_dir.rglob(f"{job.scene_class_name}.mp4")
            for path in paths:
                if (
                    path.is_symlink()
                    or "partial_movie_files" in path.parts
                    or "__smoke__" in path.parts
                ):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    # ffmpeg/manim 可能在扫描期间替换或删除文件；跳过该候选，
                    # 不应让一次竞态把成功作业判成监控器异常。
                    continue
                # submit_scene 为每次尝试创建独立的 attempt_<token> 目录，目录
                # 隔离已经排除了旧尝试污染，因此不再依赖登录节点与计算节点的
                # wall-clock 一致性。旧版/手工构造的共享目录仍保留宽松的 mtime
                # 保护，兼容历史清单并避免误拾取明显陈旧产物。
                isolated_attempt = bool(re.fullmatch(r"attempt_[0-9a-f]{12}", job.media_dir.name))
                if isolated_attempt:
                    fresh_enough = True
                else:
                    fresh_enough = stat.st_mtime >= job.submitted_at - 300.0
                if stat.st_size > 0 and fresh_enough:
                    candidates.append((path, stat.st_mtime))
        except OSError:
            return None
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])[0]

    @staticmethod
    def _read_environment_fingerprint(job: RenderJob) -> dict[str, str]:
        """读取计算节点写入的环境身份；旧 Job 缺少该文件时返回空。"""

        marker = job.media_dir / "environment.json"
        if not job.execution_sha256:
            marker = (
                job.script_path.parent.parent
                / "artifacts"
                / f"environment_scene_{job.scene_id}.json"
            )
        try:
            if not marker.is_file() or marker.is_symlink():
                return {}
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): str(value)[:300]
            for key, value in payload.items()
            if isinstance(key, str) and isinstance(value, (str, int, float))
        }

    def validate_completed_job(self, job: RenderJob) -> bool:
        """确认当前作业确实生成了可解析且匹配配置的最终视频。"""

        video = self._find_final_video(job)
        if video is None:
            job.failure_reason = "Slurm 状态为 COMPLETED，但未找到本次作业的最终 MP4"
            return False
        try:
            metadata = self._verify_video(video, job.render_profile)
            digest = sha256_file(video)
        except (OSError, RuntimeError, ValueError) as exc:
            job.failure_reason = f"Slurm 状态为 COMPLETED，但视频验证失败: {exc}"
            return False
        job.output_path = video
        job.output_metadata = metadata
        # 记录经过 ffprobe 验证的精确文件身份，供恢复和公共 merge_jobs
        # 继续拒绝目录中被替换的同名文件。
        job.output_sha256 = digest
        job.environment_fingerprint = self._read_environment_fingerprint(job)
        expected = {
            "manim": job.render_profile.manim_version,
            "ffmpeg": job.render_profile.ffmpeg_version,
            "xelatex": job.render_profile.xelatex_version,
            "renderer": job.render_profile.renderer,
            "pyopengl_platform": job.render_profile.opengl_platform,
        }
        mismatches = [
            f"{key}: {actual!r} != {value!r}"
            for key, value in expected.items()
            if value and (actual := job.environment_fingerprint.get(key)) and actual != value
        ]
        if mismatches:
            job.environment_warning = "计算节点环境与提交端配置不一致: " + "; ".join(mismatches)
        return True

    @staticmethod
    def _job_log_tail(job: RenderJob, limit: int = 4000) -> str:
        """合并作业 .out/.err 尾部文本, 用于判定已消失作业的真实结果。"""
        chunks: list[str] = []
        for path in (job.log_err, job.log_out):
            try:
                if path.is_file() and path.stat().st_size > 0:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                    chunks.append("\n".join(lines[-limit:]))
            except OSError:
                continue
        return "\n".join(chunks)[-4000:]

    def classify_gone(self, job):
        return self._classify_gone(job)

    def forward_log(self, job, positions):
        return self._forward_log(job, positions)

    def cancel_for_monitor_failure(self, job, *, status, reason):
        return self._cancel_for_monitor_failure(job, status=status, reason=reason)

    def _classify_gone(self, job: RenderJob) -> str | None:
        """作业已确认不在调度器 (squeue/sacct 均无记录) 时, 依据产物判定结果。

        - 有最终视频 → "COMPLETED" (作业在轮询间隙已结束并成功产出)
        - 有明确失败证据的日志 → "FAILED" (作业实际运行过, 可走自动修复)
        - 无任何产物 → None (可能是刚提交尚未注册, 交给计数延后决断)
        """
        if self.validate_completed_job(job):
            return "COMPLETED"
        log_tail = self._job_log_tail(job)
        # stdout 常包含正常的启动/进度/完成前日志，不能仅凭“有日志”就
        # 把作业判为失败；只有出现 traceback/异常/非零退出等明确证据时
        # 才进入 AutoFix，否则继续等待产物传播或 sacct 记录。
        # 不能使用裸的 ``error ``/``failed`` 子串：正常输出、课程文字或
        # 警告中出现这些词时，作业也可能只是短暂地从队列中消失。只匹配
        # traceback、异常类型、退出码和明确的作业终态证据。
        failure_markers = (
            re.compile(r"traceback\s*\(most recent call last\)", re.IGNORECASE),
            re.compile(r"\b(?:[a-z_][a-z0-9_]*(?:error|exception))\s*:", re.IGNORECASE),
            re.compile(r"\bfatal(?:\s+error)?\b", re.IGNORECASE),
            re.compile(r"\bnon[- ]zero(?:\s+exit)?\b", re.IGNORECASE),
            re.compile(r"\bexit(?:ed)?\s+(?:with\s+)?(?:code|status)\b", re.IGNORECASE),
            re.compile(r"\b(?:killed|cancelled|time limit)\b", re.IGNORECASE),
            re.compile(r"\b(?:render|job|process)\s+(?:failed|failure)\b", re.IGNORECASE),
        )
        if log_tail and any(marker.search(log_tail) for marker in failure_markers):
            job.failure_reason = f"作业已从集群消失，依据日志判定为失败:\n{log_tail}"
            return "FAILED"
        return None

    def _cancel_for_monitor_failure(self, job: RenderJob, *, status: str, reason: str) -> None:
        if not _dashboard_quiet():
            console.print(
                f"[bold red][Monitor][/] Scene {job.scene_id} {reason}，取消 Job {job.job_id}"
            )
        try:
            cancelled = self.cancel_job(job.job_id)
        except Exception as exc:
            cancelled = False
            reason = f"{reason}；调用 scancel 时发生异常: {exc}"
        if cancelled:
            job.cancelled = True
            job.status = status
            job.failure_reason = reason
            return
        job.status = "CANCEL_FAILED"
        job.failure_reason = f"{reason}；scancel 失败，禁止自动重提以避免重复作业"

    @staticmethod
    def _forward_log(job: RenderJob, positions: dict[str, int]) -> None:
        last_pos = positions.get(job.job_id, 0)
        try:
            if not job.log_out.exists():
                return
            size = job.log_out.stat().st_size
            if size < last_pos:
                last_pos = 0
            with job.log_out.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(last_pos)
                new = handle.read()
                positions[job.job_id] = handle.tell()
            if not _dashboard_quiet():
                for line in new.rstrip().splitlines():
                    if line.strip():
                        console.print(f"  {line}", markup=False, style="dim")
        except OSError:
            return

    def get_error_log(
        self,
        scene_id: int | None = None,
        job_id: str | None = None,
        *,
        job: RenderJob | None = None,
    ) -> str | None:
        if job is not None:
            log_paths = (job.log_err, job.log_out)
        elif scene_id is not None and job_id is not None:
            log_paths = (settings.LOGS_DIR / f"scene_{scene_id}_{job_id}.err",)
        else:
            raise ValueError("必须提供 job，或同时提供 scene_id 和 job_id")
        try:
            logs: list[tuple[str, str]] = []
            for path in log_paths:
                if not path.is_file():
                    continue
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                if lines:
                    label = "stderr" if path == log_paths[0] else "stdout"
                    logs.append((label, "\n".join(lines[-settings.LOG_TAIL_LINES :])))
            if not logs:
                return None
            # stderr 是 AutoFix 的首要诊断来源；截断时优先保留 stderr，避免
            # 大量 stdout 进度日志把真正的 traceback 顶出发送窗口。
            stderr = next((text for label, text in logs if label == "stderr"), "")
            stdout = next((text for label, text in logs if label == "stdout"), "")
            limit = settings.MAX_LOG_CHARS
            if stderr:
                stderr_block = f"[stderr]\n{stderr}"[-limit:]
                remaining = max(0, limit - len(stderr_block) - 2)
                stdout_block = f"[stdout]\n{stdout}"[-remaining:] if remaining else ""
                return f"{stderr_block}\n\n{stdout_block}".strip()
            return f"[stdout]\n{stdout}"[-limit:]
        except OSError as exc:
            if not _dashboard_quiet():
                console.print(f"[yellow][Monitor][/] 读取日志失败: {exc}", markup=False)
            return None
