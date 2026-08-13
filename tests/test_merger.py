import time

import pytest

import kd1_anime.media.merger as merger_module
from kd1_anime.cluster.slurm import SlurmJob
from kd1_anime.media.merger import VideoMerger


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
