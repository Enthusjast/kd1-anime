import time

import pytest

import kd1_anime.media.merger as merger_module
from kd1_anime.cluster.slurm import SlurmJob
from kd1_anime.media.merger import VideoMerger
from kd1_anime.rendering import MergeProfile, RenderProfile, VideoMetadata


@pytest.fixture(autouse=True)
def fake_ffmpeg_binary(monkeypatch):
    """拼接行为测试使用 fake _run_ffmpeg，不依赖 CI 主机安装 FFmpeg。"""

    monkeypatch.setattr(merger_module.shutil, "which", lambda name: "/usr/bin/ffmpeg")


def test_selects_expected_class_and_ignores_partial(tmp_path):
    media = tmp_path / "media"
    final = media / "videos" / "scene" / "1080p60" / "Demo.mp4"
    partial = media / "videos" / "scene" / "1080p60" / "partial_movie_files" / "Demo.mp4"
    other = media / "videos" / "scene" / "1080p60" / "Other.mp4"
    for path in (final, partial, other):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
    job = SlurmJob(
        "1",
        1,
        tmp_path / "x.sh",
        tmp_path / "o",
        tmp_path / "e",
        media,
        "Demo",
        time.time(),
        output_path=final,
    )
    assert VideoMerger().find_job_video(job) == final


def test_merge_jobs_uses_checkpointed_output_not_newest_directory_candidate(tmp_path):
    media = tmp_path / "media"
    exact = media / "videos" / "scene" / "1080p60" / "Demo.mp4"
    stale = media / "videos" / "scene" / "1080p60" / "newer" / "Demo.mp4"
    exact.parent.mkdir(parents=True)
    stale.parent.mkdir(parents=True)
    exact.write_bytes(b"exact")
    stale.write_bytes(b"stale")
    job = SlurmJob(
        "1",
        1,
        tmp_path / "x.sh",
        tmp_path / "o",
        tmp_path / "e",
        media,
        "Demo",
        time.time(),
        output_path=exact,
    )

    assert VideoMerger().collect_job_videos([job]) == [exact]


def test_merge_jobs_rejects_unverified_job_without_output_path(tmp_path):
    job = SlurmJob(
        "1",
        1,
        tmp_path / "x.sh",
        tmp_path / "o",
        tmp_path / "e",
        tmp_path / "media",
        "Demo",
        time.time(),
    )
    try:
        VideoMerger().collect_job_videos([job])
    except RuntimeError as exc:
        assert "output_path" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_merge_jobs_rejects_duplicate_scene_ids(tmp_path):
    jobs = [
        SlurmJob(
            "1",
            1,
            tmp_path / "x1.sh",
            tmp_path / "o1",
            tmp_path / "e1",
            tmp_path / "media1",
            "Demo1",
            time.time(),
        ),
        SlurmJob(
            "2",
            1,
            tmp_path / "x2.sh",
            tmp_path / "o2",
            tmp_path / "e2",
            tmp_path / "media2",
            "Demo2",
            time.time(),
        ),
    ]

    with pytest.raises(ValueError, match="重复的 scene_id"):
        VideoMerger().merge_jobs(jobs, output_path=tmp_path / "output.mp4")


def test_existing_output_requires_force(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    video = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    video.write_bytes(b"video")
    output.write_bytes(b"old")
    monkeypatch.setattr(settings, "OVERWRITE_OUTPUT", False)

    try:
        VideoMerger().merge([video], output)
    except RuntimeError as exc:
        assert "拒绝覆盖" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert output.read_bytes() == b"old"


def test_explicit_same_run_replacement_does_not_require_global_force(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    video = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    video.write_bytes(b"video")
    output.write_bytes(b"old")
    monkeypatch.setattr(settings, "OVERWRITE_OUTPUT", False)
    merger = VideoMerger()

    def fake_ffmpeg(cmd, temporary_output, label):
        temporary_output.write_bytes(b"new")
        return True

    monkeypatch.setattr(merger, "_run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(merger, "_verify_output", lambda *args: True)

    result = merger.merge([video], output, replace_existing=True)

    assert result == output
    assert output.read_bytes() == b"new"


def test_merge_uses_atomic_temporary_output(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    video = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    video.write_bytes(b"video")
    output.write_bytes(b"old")
    monkeypatch.setattr(settings, "OVERWRITE_OUTPUT", True)
    merger = VideoMerger()

    def fake_ffmpeg(cmd, temporary_output, label):
        temporary_output.write_bytes(b"new")
        return True

    monkeypatch.setattr(merger, "_run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(merger, "_verify_output", lambda *args: True)

    result = merger.merge([video], output)

    assert result == output
    assert output.read_bytes() == b"new"
    assert not list(tmp_path.glob(".*.tmp.mp4"))


def test_invalid_temporary_video_never_replaces_existing_output(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    video = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    video.write_bytes(b"video")
    output.write_bytes(b"old")
    monkeypatch.setattr(settings, "OVERWRITE_OUTPUT", True)
    merger = VideoMerger()

    def fake_ffmpeg(cmd, temporary_output, label):
        temporary_output.write_bytes(b"corrupt")
        return True

    monkeypatch.setattr(merger, "_run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(merger, "_verify_output", lambda *args: False)

    try:
        merger.merge([video], output)
    except RuntimeError as exc:
        assert "FFmpeg 拼接失败" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert output.read_bytes() == b"old"


def test_multi_video_merge_uses_xfade_and_expected_offsets(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    output = tmp_path / "output.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    profile = RenderProfile(
        renderer="cairo", quality="l", pixel_width=320, pixel_height=180, frame_rate=30
    )
    monkeypatch.setattr(settings, "TRANSITION_DURATION", 0.5)
    monkeypatch.setattr(
        merger_module,
        "verify_video",
        lambda path, render_profile: VideoMetadata(
            size_bytes=path.stat().st_size,
            duration_seconds=2.0 if path == first else 3.0,
            width=render_profile.pixel_width,
            height=render_profile.pixel_height,
            frame_rate=render_profile.frame_rate,
        ),
    )
    merger = VideoMerger()
    captured = {}

    def fake_ffmpeg(cmd, temporary_output, label):
        captured["cmd"] = cmd
        captured["label"] = label
        temporary_output.write_bytes(b"merged")
        return True

    monkeypatch.setattr(merger, "_run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(merger, "_verify_output", lambda *args, **kwargs: True)

    result = merger.merge([first, second], output, replace_existing=True, render_profile=profile)

    command = " ".join(captured["cmd"])
    assert result == output
    assert captured["label"] == "xfade"
    assert "xfade=transition=fade" in command
    assert "duration=0.500000" in command
    assert "offset=1.500000" in command
    assert "concat" not in command


def test_xfade_uses_configured_encoding_and_silent_audio_settings(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    output = tmp_path / "output.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    profile = RenderProfile(
        renderer="cairo", quality="l", pixel_width=320, pixel_height=180, frame_rate=30
    )
    monkeypatch.setattr(settings, "MERGE_VIDEO_CODEC", "libx265")
    monkeypatch.setattr(settings, "MERGE_VIDEO_PRESET", "fast")
    monkeypatch.setattr(settings, "MERGE_VIDEO_CRF", 23)
    monkeypatch.setattr(settings, "MERGE_AUDIO_SAMPLE_RATE", 44100)
    monkeypatch.setattr(settings, "MERGE_AUDIO_CHANNEL_LAYOUT", "stereo")
    monkeypatch.setattr(
        merger_module,
        "verify_video",
        lambda path, render_profile: VideoMetadata(
            size_bytes=path.stat().st_size,
            duration_seconds=2.0,
            width=render_profile.pixel_width,
            height=render_profile.pixel_height,
            frame_rate=render_profile.frame_rate,
            has_audio=path == first,
        ),
    )
    merger = VideoMerger()
    captured = {}

    def fake_ffmpeg(cmd, temporary_output, label):
        captured["cmd"] = cmd
        temporary_output.write_bytes(b"merged")
        return True

    monkeypatch.setattr(merger, "_run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(merger, "_verify_output", lambda *args, **kwargs: True)

    merger.merge([first, second], output, replace_existing=True, render_profile=profile)

    command = " ".join(captured["cmd"])
    assert "anullsrc=channel_layout=stereo:sample_rate=44100" in command
    assert "-c:v libx265" in command
    assert "-preset fast" in command
    assert "-crf 23" in command


def test_xfade_uses_captured_merge_profile_instead_of_mutated_settings(monkeypatch, tmp_path):
    from kd1_anime.config import settings

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    output = tmp_path / "output.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    render_profile = RenderProfile(
        renderer="cairo", quality="l", pixel_width=320, pixel_height=180, frame_rate=30
    )
    merge_profile = MergeProfile(
        transition_duration=0.25,
        video_codec="libx264",
        video_preset="fast",
        video_crf=27,
        audio_sample_rate=44100,
        audio_channel_layout="stereo",
    )
    monkeypatch.setattr(settings, "TRANSITION_DURATION", 0.5)
    monkeypatch.setattr(settings, "MERGE_VIDEO_PRESET", "slow")
    monkeypatch.setattr(
        merger_module,
        "verify_video",
        lambda path, profile: VideoMetadata(
            size_bytes=path.stat().st_size,
            duration_seconds=2.0,
            width=profile.pixel_width,
            height=profile.pixel_height,
            frame_rate=profile.frame_rate,
        ),
    )
    merger = VideoMerger()
    captured = {}

    def fake_ffmpeg(cmd, temporary_output, label):
        captured["cmd"] = cmd
        temporary_output.write_bytes(b"merged")
        return True

    monkeypatch.setattr(merger, "_run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(merger, "_verify_output", lambda *args, **kwargs: True)

    merger.merge(
        [first, second],
        output,
        replace_existing=True,
        render_profile=render_profile,
        merge_profile=merge_profile,
    )

    command = " ".join(captured["cmd"])
    assert "duration=0.250000" in command
    assert "-preset fast" in command
    assert "-crf 27" in command


def test_run_ffmpeg_handles_process_start_failure(monkeypatch, tmp_path):
    def fail_run(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(merger_module.subprocess, "run", fail_run)

    assert not VideoMerger._run_ffmpeg(["ffmpeg"], tmp_path / "output.mp4", "test")
