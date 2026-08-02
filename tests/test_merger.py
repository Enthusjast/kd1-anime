import time

from kd1_anime.cluster.slurm import SlurmJob
from kd1_anime.media.merger import VideoMerger


def test_selects_expected_class_and_ignores_partial(tmp_path):
    media = tmp_path / "media"
    final = media / "videos" / "scene" / "1080p60" / "Demo.mp4"
    partial = media / "videos" / "scene" / "1080p60" / "partial_movie_files" / "Demo.mp4"
    other = media / "videos" / "scene" / "1080p60" / "Other.mp4"
    for path in (final, partial, other):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
    job = SlurmJob(
        "1", 1, tmp_path / "x.sh", tmp_path / "o", tmp_path / "e", media, "Demo", time.time()
    )
    assert VideoMerger().find_job_video(job) == final


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

    result = merger.merge([video], output)

    assert result == output
    assert output.read_bytes() == b"new"
    assert not list(tmp_path.glob(".*.tmp.mp4"))
