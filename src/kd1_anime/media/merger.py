"""使用 FFmpeg 拼接当前 run 的 Manim 场景视频。"""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
from contextlib import contextmanager, suppress
from pathlib import Path
from uuid import uuid4

from rich.console import Console

from kd1_anime.cluster.slurm import SlurmJob
from kd1_anime.config import settings
from kd1_anime.rendering import RenderProfile, VideoMetadata, sha256_file, verify_video

console = Console()


def _dashboard_quiet() -> bool:
    """Rich Live 仪表盘运行时抑制合并器的普通输出。"""

    try:
        from kd1_anime.dashboard import quiet

        return quiet()
    except Exception:
        return False


def _print(*args, **kwargs) -> None:
    if not _dashboard_quiet():
        console.print(*args, **kwargs)


class VideoMerger:
    @staticmethod
    @contextmanager
    def _output_lock(output: Path):
        """锁定最终输出目标，避免两个独立进程同时转场合并/替换同一文件。"""

        lock_path = output.with_name(f".{output.name}.lock")
        if lock_path.is_symlink():
            raise RuntimeError(f"输出锁不能是符号链接: {lock_path}")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise RuntimeError(f"无法打开输出锁: {lock_path}") from exc
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"输出文件正被另一个拼接进程使用: {output}") from exc
            os.fchmod(descriptor, 0o600)
            yield
        finally:
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _exact_job_video(job: SlurmJob) -> Path:
        """只接受 SlurmJob 已验证并记录的当前产物，不扫描目录猜测。"""

        if job.output_path is None:
            raise RuntimeError(
                f"Scene {job.scene_id} 缺少当前 Slurm Job 的 output_path，拒绝扫描目录选择视频"
            )
        video = job.output_path.expanduser().resolve()
        media_dir = job.media_dir.expanduser().resolve()
        try:
            video.relative_to(media_dir)
        except ValueError as exc:
            raise RuntimeError(f"Scene {job.scene_id} 的 output_path 不在当前媒体目录内") from exc
        if "partial_movie_files" in video.parts or video.name != f"{job.scene_class_name}.mp4":
            raise RuntimeError(f"Scene {job.scene_id} 的 output_path 不是当前类的最终 MP4")
        if not video.is_file() or video.stat().st_size <= 0:
            raise RuntimeError(f"Scene {job.scene_id} 的当前产物不存在或为空: {video}")
        if job.output_sha256 and sha256_file(video) != job.output_sha256:
            raise RuntimeError(f"Scene {job.scene_id} 的当前产物哈希与 Slurm 检查点不一致")
        return video

    def find_job_video(self, job: SlurmJob) -> Path:
        """返回当前 SlurmJob 已验证并记录的最终视频。"""

        return self._exact_job_video(job)

    def discover_job_video(self, job: SlurmJob) -> Path:
        """显式的旧版目录发现 helper，仅用于诊断，不可用于 merge_jobs。"""

        if not job.media_dir.exists():
            raise RuntimeError(f"Scene {job.scene_id} 的媒体目录不存在: {job.media_dir}")
        candidates: list[tuple[Path, float]] = []
        try:
            for path in job.media_dir.rglob(f"{job.scene_class_name}.mp4"):
                if "partial_movie_files" in path.parts:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    # 成品可能正在由 Manim/ffmpeg 替换；忽略本轮消失的候选，
                    # 避免一次目录竞态导致整个合并失败。
                    continue
                if stat.st_size > 0:
                    candidates.append((path, stat.st_mtime))
        except OSError as exc:
            raise RuntimeError(f"Scene {job.scene_id} 的媒体目录无法读取: {job.media_dir}") from exc
        if not candidates:
            raise RuntimeError(
                f"Scene {job.scene_id} 未找到当前类 {job.scene_class_name!r} 的最终 MP4"
            )
        # 当前 run 的目录是隔离的；若 Manim 产生多个质量目录，取最新完成的文件。
        return max(candidates, key=lambda item: item[1])[0]

    def collect_job_videos(
        self,
        jobs: list[SlurmJob],
        *,
        render_profile: RenderProfile | None = None,
    ) -> list[Path]:
        videos: list[Path] = []
        for job in sorted(jobs, key=lambda item: item.scene_id):
            video = self._exact_job_video(job)
            # 旧调用方可能只有 output_path；对新检查点记录的元数据再做一次
            # ffprobe 校验，避免 merge_jobs 盲信同名 MP4。
            if job.output_metadata is not None:
                metadata = verify_video(video, render_profile or job.render_profile)
                if metadata != job.output_metadata:
                    raise RuntimeError(f"Scene {job.scene_id} 的视频元数据与 Slurm 检查点不一致")
            videos.append(video)
            _print(f"[dim][Merger][/] Scene {job.scene_id}: {video}")
        return videos

    def merge(
        self,
        video_paths: list[Path],
        output_path: Path,
        *,
        replace_existing: bool = False,
        render_profile: RenderProfile | None = None,
    ) -> Path:
        if not video_paths:
            raise RuntimeError("没有视频文件可供拼接")
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with self._output_lock(output):
            return self._merge_unlocked(
                video_paths,
                output,
                replace_existing=replace_existing,
                render_profile=render_profile,
            )

    def _merge_unlocked(
        self,
        video_paths: list[Path],
        output: Path,
        *,
        replace_existing: bool = False,
        render_profile: RenderProfile | None = None,
    ) -> Path:
        resolved_inputs = [path.expanduser().resolve() for path in video_paths]
        if output in resolved_inputs:
            raise RuntimeError("输出文件不能与任一输入视频相同")
        if output.exists() and not (settings.OVERWRITE_OUTPUT or replace_existing):
            raise RuntimeError(f"输出文件已存在，拒绝覆盖: {output}（使用 --force 允许覆盖）")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("未找到 ffmpeg，请先激活包含 FFmpeg 的环境")
        profile = render_profile or RenderProfile.current()
        temporary_output = output.with_name(
            f".{output.stem}.{uuid4().hex[:8]}.tmp{output.suffix or '.mp4'}"
        )

        try:
            if len(resolved_inputs) == 1:
                # 单场景不需要转场，直接 remux；多场景一律走 xfade，避免
                # 多场景不能回退到无转场的直接拼接。
                single_cmd = [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(resolved_inputs[0]),
                    "-c",
                    "copy",
                    str(temporary_output),
                ]
                if self._run_ffmpeg(
                    single_cmd, temporary_output, "single-video"
                ) and self._verify_output(temporary_output, profile, "single-video"):
                    temporary_output.replace(output)
                    output.chmod(0o600)
                    return output
                raise RuntimeError("FFmpeg 拼接失败（单视频处理失败）")

            metadata = [verify_video(path, profile) for path in resolved_inputs]
            xfade_cmd, expected_audio, expected_duration = self._build_xfade_command(
                ffmpeg,
                resolved_inputs,
                metadata,
                profile,
                temporary_output,
            )
            if self._run_ffmpeg(xfade_cmd, temporary_output, "xfade") and self._verify_output(
                temporary_output,
                profile,
                "xfade",
                expected_audio=expected_audio,
                expected_duration=expected_duration,
            ):
                temporary_output.replace(output)
                output.chmod(0o600)
                return output
            raise RuntimeError("FFmpeg xfade 拼接失败")
        finally:
            with suppress(OSError):
                temporary_output.unlink(missing_ok=True)

    @staticmethod
    def _build_xfade_command(
        ffmpeg: str,
        inputs: list[Path],
        metadata: list[VideoMetadata],
        profile: RenderProfile,
        output: Path,
    ) -> tuple[list[str], bool, float]:
        """构造链式 xfade/acrossfade 命令，并返回音频和时长预期。"""

        if len(inputs) < 2:
            raise ValueError("xfade 至少需要两个输入视频")
        if len(inputs) != len(metadata):
            raise ValueError("输入视频与视频元数据数量不一致")
        transition = min(
            settings.TRANSITION_DURATION, min(item.duration_seconds for item in metadata) / 2
        )
        if transition <= 0.01:
            raise RuntimeError("视频时长过短，无法安全添加转场")
        width = profile.pixel_width
        height = profile.pixel_height
        fps = profile.frame_rate
        video_filter = (
            f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            "settb=AVTB,format=yuv420p"
        )
        filter_parts = [f"[{index}:v]{video_filter}[v{index}]" for index in range(len(inputs))]
        current_label = "v0"
        elapsed = 0.0
        for index in range(1, len(inputs)):
            elapsed += metadata[index - 1].duration_seconds
            offset = max(0.0, elapsed - index * transition)
            next_label = f"vxf{index}"
            filter_parts.append(
                f"[{current_label}][v{index}]xfade=transition={settings.TRANSITION_TYPE}:"
                f"duration={transition:.6f}:offset={offset:.6f}[{next_label}]"
            )
            current_label = next_label
        command = [ffmpeg, "-y"]
        command.extend(item for path in inputs for item in ("-i", str(path)))
        has_any_audio = any(item.has_audio for item in metadata)
        expected_audio = has_any_audio
        audio_indices: list[int] = list(range(len(inputs)))
        next_audio_input = len(inputs)
        if has_any_audio:
            # 对无音频场景补同长度静音，使 acrossfade 在混合输入时仍可用。
            for index, item in enumerate(metadata):
                if not item.has_audio:
                    audio_indices[index] = next_audio_input
                    next_audio_input += 1
                    command.extend(
                        [
                            "-f",
                            "lavfi",
                            "-t",
                            f"{item.duration_seconds:.6f}",
                            "-i",
                            (
                                "anullsrc=channel_layout="
                                f"{settings.MERGE_AUDIO_CHANNEL_LAYOUT}:"
                                f"sample_rate={settings.MERGE_AUDIO_SAMPLE_RATE}"
                            ),
                        ]
                    )
            audio_labels: list[str] = []
            for index, source_index in enumerate(audio_indices):
                label = f"a{index}"
                filter_parts.append(
                    f"[{source_index}:a]aresample={settings.MERGE_AUDIO_SAMPLE_RATE},"
                    f"aformat=sample_fmts=fltp:sample_rates={settings.MERGE_AUDIO_SAMPLE_RATE}:"
                    f"channel_layouts={settings.MERGE_AUDIO_CHANNEL_LAYOUT},"
                    f"atrim=duration={metadata[index].duration_seconds:.6f},"
                    f"apad=whole_dur={metadata[index].duration_seconds:.6f},"
                    f"asetpts=PTS-STARTPTS[{label}]"
                )
                audio_labels.append(label)
            current_audio = audio_labels[0]
            for index in range(1, len(audio_labels)):
                next_audio = f"axf{index}"
                filter_parts.append(
                    f"[{current_audio}][{audio_labels[index]}]acrossfade=d={transition:.6f}:"
                    f"c1=tri:c2=tri[{next_audio}]"
                )
                current_audio = next_audio
            audio_output_label = current_audio
        else:
            audio_output_label = ""

        filter_complex = ";".join(filter_parts)
        command.extend(["-filter_complex", filter_complex, "-map", f"[{current_label}]"])
        if expected_audio:
            command.extend(["-map", f"[{audio_output_label}]"])
            command.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            command.append("-an")
        command.extend(
            [
                "-c:v",
                settings.MERGE_VIDEO_CODEC,
                "-preset",
                settings.MERGE_VIDEO_PRESET,
                "-crf",
                str(settings.MERGE_VIDEO_CRF),
                "-pix_fmt",
                "yuv420p",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output),
            ]
        )
        expected_duration = sum(item.duration_seconds for item in metadata) - transition * (
            len(metadata) - 1
        )
        return command, expected_audio, expected_duration

    @staticmethod
    def _run_ffmpeg(cmd: list[str], output: Path, label: str) -> bool:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, check=False)
        except subprocess.TimeoutExpired:
            _print(f"[red][Merger][/] ffmpeg {label} 超时")
            return False
        except OSError as exc:
            _print(f"[red][Merger][/] ffmpeg {label} 无法启动: {exc}", markup=False)
            return False
        try:
            output_size = output.stat().st_size if output.is_file() else 0
        except OSError as exc:
            _print(f"[red][Merger][/] ffmpeg {label} 产物无法读取: {exc}", markup=False)
            return False
        if result.returncode != 0 or output_size == 0:
            _print(
                f"[red][Merger][/] ffmpeg {label} 失败: {result.stderr[-1000:]}",
                markup=False,
            )
            return False
        _print(f"[bold green][Merger][/] 拼接完成 ({label}): {output_size / (1024 * 1024):.1f} MB")
        return True

    @staticmethod
    def _verify_output(
        output: Path,
        profile: RenderProfile,
        label: str,
        *,
        expected_audio: bool | None = None,
        expected_duration: float | None = None,
    ) -> bool:
        try:
            metadata = verify_video(output, profile)
            if expected_audio is not None and metadata.has_audio != expected_audio:
                raise ValueError(f"音频流状态不符合预期: {metadata.has_audio} != {expected_audio}")
            if (
                expected_duration is not None
                and abs(metadata.duration_seconds - expected_duration) > 0.25
            ):
                raise ValueError(
                    f"输出时长不符合预期: {metadata.duration_seconds:.3f} != "
                    f"{expected_duration:.3f}"
                )
        except (OSError, RuntimeError, ValueError) as exc:
            _print(
                f"[red][Merger][/] ffmpeg {label} 产物验证失败: {exc}",
                markup=False,
            )
            return False
        return True

    def merge_jobs(
        self,
        jobs: list[SlurmJob],
        *,
        output_path: Path,
        render_profile: RenderProfile | None = None,
    ) -> Path:
        scene_ids = [job.scene_id for job in jobs]
        if len(set(scene_ids)) != len(scene_ids):
            raise ValueError("合并任务包含重复的 scene_id")
        return self.merge(
            self.collect_job_videos(jobs, render_profile=render_profile),
            output_path,
            render_profile=render_profile,
        )
