"""使用 FFmpeg 拼接当前 run 的 Manim 场景视频。"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from rich.console import Console

from kd1_anime.cluster.slurm import SlurmJob
from kd1_anime.config import settings

console = Console()


class VideoMerger:
    def find_job_video(self, job: SlurmJob) -> Path:
        """按当前 job 的 media_dir 和 Scene 类名精确定位最终视频。"""

        if not job.media_dir.exists():
            raise RuntimeError(f"Scene {job.scene_id} 的媒体目录不存在: {job.media_dir}")
        candidates = [
            path
            for path in job.media_dir.rglob(f"{job.scene_class_name}.mp4")
            if "partial_movie_files" not in path.parts and path.stat().st_size > 0
        ]
        if not candidates:
            raise RuntimeError(
                f"Scene {job.scene_id} 未找到当前类 {job.scene_class_name!r} 的最终 MP4"
            )
        # 当前 run 的目录是隔离的；若 Manim 产生多个质量目录，取最新完成的文件。
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def collect_job_videos(self, jobs: list[SlurmJob]) -> list[Path]:
        videos: list[Path] = []
        for job in sorted(jobs, key=lambda item: item.scene_id):
            video = self.find_job_video(job)
            videos.append(video)
            console.print(f"[dim][Merger][/] Scene {job.scene_id}: {video}")
        return videos

    def merge(
        self,
        video_paths: list[Path],
        output_path: Path,
    ) -> Path:
        if not video_paths:
            raise RuntimeError("没有视频文件可供拼接")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("未找到 ffmpeg，请先激活包含 FFmpeg 的环境")

        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        resolved_inputs = [path.expanduser().resolve() for path in video_paths]
        if output in resolved_inputs:
            raise RuntimeError("输出文件不能与任一输入视频相同")
        if output.exists() and not settings.OVERWRITE_OUTPUT:
            raise RuntimeError(f"输出文件已存在，拒绝覆盖: {output}（使用 --force 允许覆盖）")
        temporary_output = output.with_name(
            f".{output.stem}.{uuid4().hex[:8]}.tmp{output.suffix or '.mp4'}"
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="kd1-anime-concat-",
            suffix=".txt",
            dir=output.parent,
            delete=False,
        ) as filelist_handle:
            filelist = Path(filelist_handle.name)
            for video in video_paths:
                safe = str(video.resolve()).replace("'", "'\\''")
                filelist_handle.write(f"file '{safe}'\n")

        try:
            copy_cmd = [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(filelist),
                "-c",
                "copy",
                str(temporary_output),
            ]
            if self._run_ffmpeg(copy_cmd, temporary_output, "stream copy"):
                temporary_output.replace(output)
                output.chmod(0o600)
                return output
            with suppress(OSError):
                temporary_output.unlink(missing_ok=True)

            width = settings.MANIM_PIXEL_WIDTH
            height = settings.MANIM_PIXEL_HEIGHT
            fps = settings.MANIM_FRAME_RATE
            video_filter = (
                f"fps={fps},"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                "setsar=1,format=yuv420p"
            )
            reencode_cmd = [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(filelist),
                "-vf",
                video_filter,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(temporary_output),
            ]
            if self._run_ffmpeg(reencode_cmd, temporary_output, "re-encode"):
                temporary_output.replace(output)
                output.chmod(0o600)
                return output
            raise RuntimeError("FFmpeg 拼接失败（stream copy 和重编码均失败）")
        finally:
            with suppress(OSError):
                filelist.unlink(missing_ok=True)
            with suppress(OSError):
                temporary_output.unlink(missing_ok=True)

    @staticmethod
    def _run_ffmpeg(cmd: list[str], output: Path, label: str) -> bool:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, check=False)
        except subprocess.TimeoutExpired:
            console.print(f"[red][Merger][/] ffmpeg {label} 超时")
            return False
        if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            console.print(
                f"[red][Merger][/] ffmpeg {label} 失败: {result.stderr[-1000:]}",
                markup=False,
            )
            return False
        console.print(
            f"[bold green][Merger][/] 拼接完成 ({label}): "
            f"{output.stat().st_size / (1024 * 1024):.1f} MB"
        )
        return True

    def merge_jobs(
        self,
        jobs: list[SlurmJob],
        *,
        output_path: Path,
    ) -> Path:
        return self.merge(self.collect_job_videos(jobs), output_path)
