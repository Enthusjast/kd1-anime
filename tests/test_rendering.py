import subprocess
from types import SimpleNamespace

import pytest

from kd1_anime.config import settings
from kd1_anime.rendering import RenderProfile, probe_video, verify_video


def test_render_profile_digest_changes_with_output_settings(monkeypatch):
    original = RenderProfile.current()
    monkeypatch.setattr(settings, "MANIM_FRAME_RATE", original.frame_rate + 1)
    changed = RenderProfile.current()

    assert original.digest() != changed.digest()


def test_render_profile_rejects_unsafe_script_values():
    with pytest.raises(ValueError):
        RenderProfile(
            renderer="opengl",
            quality="h; touch /tmp/untrusted",
            pixel_width=1920,
            pixel_height=1080,
            frame_rate=60,
        )

    with pytest.raises(ValueError):
        RenderProfile(
            renderer="opengl",
            quality="h",
            pixel_width=1920,
            pixel_height=1080,
            frame_rate=60,
            opengl_platform="evil",
        )


def test_probe_video_parses_fractional_frame_rate(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr("kd1_anime.rendering.shutil.which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        "kd1_anime.rendering.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                '{"streams":[{"width":1920,"height":1080,'
                '"avg_frame_rate":"30000/1001","codec_type":"video"},'
                '{"codec_type":"audio"}],"format":{"duration":"2.5"}}'
            ),
            stderr="",
        ),
    )

    metadata = probe_video(video)

    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.duration_seconds == 2.5
    assert metadata.frame_rate == pytest.approx(29.97003)
    assert metadata.has_audio is True


def test_probe_video_converts_timeout_to_runtime_error(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr("kd1_anime.rendering.shutil.which", lambda name: "/usr/bin/ffprobe")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout=30)

    monkeypatch.setattr("kd1_anime.rendering.subprocess.run", timeout)

    with pytest.raises(RuntimeError, match="ffprobe 验证视频超时"):
        probe_video(video)


def test_verify_video_rejects_wrong_resolution(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    profile = RenderProfile.current()
    from kd1_anime.rendering import VideoMetadata

    monkeypatch.setattr(
        "kd1_anime.rendering.probe_video",
        lambda path: VideoMetadata(
            size_bytes=5,
            duration_seconds=1,
            width=profile.pixel_width // 2,
            height=profile.pixel_height // 2,
            frame_rate=profile.frame_rate,
        ),
    )

    with pytest.raises(ValueError, match="分辨率"):
        verify_video(video, profile)


def test_verify_video_rejects_wrong_frame_rate(monkeypatch, tmp_path):
    from kd1_anime.rendering import VideoMetadata

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    profile = RenderProfile.current()
    monkeypatch.setattr(
        "kd1_anime.rendering.probe_video",
        lambda path: VideoMetadata(
            size_bytes=5,
            duration_seconds=1,
            width=profile.pixel_width,
            height=profile.pixel_height,
            frame_rate=profile.frame_rate + 1,
        ),
    )

    with pytest.raises(ValueError, match="帧率"):
        verify_video(video, profile)
