"""渲染配置、视频探测和可复用产物模型。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
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

    @classmethod
    def current(cls) -> RenderProfile:
        return cls(
            renderer=settings.MANIM_RENDERER,
            quality=settings.MANIM_QUALITY,
            pixel_width=settings.MANIM_PIXEL_WIDTH,
            pixel_height=settings.MANIM_PIXEL_HEIGHT,
            frame_rate=settings.MANIM_FRAME_RATE,
            opengl_platform=settings.MANIM_OPENGL_PLATFORM,
        )

    def digest(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VideoMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    size_bytes: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_rate: float = Field(gt=0)


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
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate:format=duration",
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
        stream = payload["streams"][0]
        duration = float(payload["format"]["duration"])
        frame_rate = _parse_rate(str(stream["avg_frame_rate"]))
        return VideoMetadata(
            size_bytes=path.stat().st_size,
            duration_seconds=duration,
            width=int(stream["width"]),
            height=int(stream["height"]),
            frame_rate=frame_rate,
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
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
