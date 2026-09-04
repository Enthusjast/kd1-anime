"""渲染配置、视频探测和可复用产物模型。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kd1_anime.config import settings


class RenderProfile(BaseModel):
    """会影响最终视频字节或兼容性的完整渲染配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    renderer: Literal["cairo", "opengl"]
    quality: Literal["l", "m", "h", "p", "k"]
    pixel_width: int = Field(gt=0)
    pixel_height: int = Field(gt=0)
    frame_rate: int = Field(gt=0)
    opengl_platform: Literal["egl", "glx"] = "egl"
    # 运行环境也会影响 TeX、OpenGL 和视频编码结果。旧 manifest 没有这些
    # 字段时默认为空，读取仍兼容；新运行会把探测到的版本纳入产物身份。
    manim_version: str = Field(default="", max_length=200)
    ffmpeg_version: str = Field(default="", max_length=300)
    xelatex_version: str = Field(default="", max_length=300)

    @staticmethod
    @lru_cache(maxsize=1)
    def _environment_versions() -> tuple[str, str, str]:
        try:
            manim_version = importlib.metadata.version("manim")
        except (importlib.metadata.PackageNotFoundError, ValueError):
            manim_version = ""

        def command_version(command: str, flag: str) -> str:
            executable = shutil.which(command)
            if not executable:
                return ""
            try:
                result = subprocess.run(
                    [executable, flag],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return ""
            output = (result.stdout or result.stderr).splitlines()
            return output[0].strip()[:300] if result.returncode == 0 and output else ""

        return (
            str(manim_version)[:200],
            command_version("ffmpeg", "-version"),
            command_version("xelatex", "--version"),
        )

    @classmethod
    def current(cls) -> RenderProfile:
        manim_version, ffmpeg_version, xelatex_version = cls._environment_versions()
        return cls(
            renderer=settings.MANIM_RENDERER,
            quality=settings.MANIM_QUALITY,
            pixel_width=settings.MANIM_PIXEL_WIDTH,
            pixel_height=settings.MANIM_PIXEL_HEIGHT,
            frame_rate=settings.MANIM_FRAME_RATE,
            opengl_platform=settings.MANIM_OPENGL_PLATFORM,
            manim_version=manim_version,
            ffmpeg_version=ffmpeg_version,
            xelatex_version=xelatex_version,
        )

    def digest(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def effective_transition_duration(durations: Iterable[float]) -> float:
    """返回 FFmpeg xfade 实际会采用的统一转场时长。"""

    values = [float(duration) for duration in durations]
    if len(values) < 2:
        return 0.0
    shortest = min(values)
    return min(settings.TRANSITION_DURATION, max(0.0, shortest / 2))


class VideoMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    size_bytes: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_rate: float = Field(gt=0)
    has_audio: bool = False


class SceneArtifact(BaseModel):
    """已经验证、可供 merge 或增量复用的视频产物。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    origin: Literal["rendered", "reused"]
    source_run_id: str = Field(pattern=r"^\d{8}-\d{6}-[0-9a-f]{8}$")
    job_id: str | None = Field(default=None, pattern=r"^\d+$")
    scene_id: int = Field(ge=1)
    scene_class_name: str = Field(min_length=1, max_length=200)
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    render_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    video_path: str = Field(min_length=1, max_length=4096)
    video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: VideoMetadata
    verified: bool = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_rate(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    denominator_value = float(denominator)
    if denominator_value == 0:
        raise ValueError("视频帧率分母为 0")
    return float(numerator) / denominator_value


def probe_video(path: Path) -> VideoMetadata:
    """使用 ffprobe 验证视频容器并返回关键元数据。"""

    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"视频不存在或为空: {path}")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("未找到 ffprobe，无法验证渲染产物")
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,width,height,avg_frame_rate:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe 验证视频超时: {path.name}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise ValueError(f"ffprobe 无法解析视频 {path.name}: {detail}")
    try:
        payload = json.loads(result.stdout)
        streams = payload["streams"]
        stream = next(item for item in streams if item.get("codec_type", "video") == "video")
        duration = float(payload["format"]["duration"])
        frame_rate = _parse_rate(str(stream["avg_frame_rate"]))
        return VideoMetadata(
            size_bytes=path.stat().st_size,
            duration_seconds=duration,
            width=int(stream["width"]),
            height=int(stream["height"]),
            frame_rate=frame_rate,
            has_audio=any(item.get("codec_type") == "audio" for item in streams),
        )
    except (
        KeyError,
        IndexError,
        StopIteration,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(f"ffprobe 返回的元数据不完整: {path}") from exc


def verify_video(path: Path, profile: RenderProfile) -> VideoMetadata:
    """验证视频可解析且与请求的 RenderProfile 一致。"""

    metadata = probe_video(path)
    if (metadata.width, metadata.height) != (profile.pixel_width, profile.pixel_height):
        raise ValueError(
            "视频分辨率与渲染配置不一致: "
            f"{metadata.width}x{metadata.height} != "
            f"{profile.pixel_width}x{profile.pixel_height}"
        )
    if abs(metadata.frame_rate - profile.frame_rate) > 0.05:
        raise ValueError(
            f"视频帧率与渲染配置不一致: {metadata.frame_rate:.3f} != {profile.frame_rate}"
        )
    return metadata
