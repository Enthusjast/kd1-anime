from pathlib import Path

import pytest

from kd1_anime.eval.boundary_checks import check_boundary_samples
from kd1_anime.eval.visual_eval import FrameSample
from kd1_anime.rendering import sha256_file

Image = pytest.importorskip("PIL.Image")


def make_sample(path: Path, frame_id: str, role: str, boundary_id: str = "B01"):
    return FrameSample(
        frame_id=frame_id,
        path=path,
        scene_id=1 if role == "boundary_end" else 2,
        boundary_id=boundary_id,
        role=role,
        image_sha256=sha256_file(path),
    )


def write_image(path: Path, color, size=(32, 18)):
    image = Image.new("RGB", size, color=color)
    image.save(path, format="PNG")


def test_boundary_check_passes_for_healthy_matching_frames(tmp_path):
    end = tmp_path / "end.png"
    start = tmp_path / "start.png"
    write_image(end, (30, 40, 50))
    write_image(start, (35, 45, 55))

    report = check_boundary_samples(
        [make_sample(end, "F01", "boundary_end"), make_sample(start, "F02", "boundary_start")]
    )

    assert report.status == "passed"
    assert report.checks[0].status == "passed"


def test_boundary_check_finds_black_frame_and_dimension_mismatch(tmp_path):
    end = tmp_path / "end.png"
    start = tmp_path / "start.png"
    write_image(end, (0, 0, 0), size=(32, 18))
    write_image(start, (255, 255, 255), size=(16, 9))

    report = check_boundary_samples(
        [make_sample(end, "F01", "boundary_end"), make_sample(start, "F02", "boundary_start")]
    )

    assert report.status == "failed"
    assert any("黑帧" in message for message in report.checks[0].messages)
    assert any("尺寸" in message for message in report.checks[0].messages)


def test_boundary_check_warns_on_large_brightness_jump(tmp_path):
    end = tmp_path / "end.png"
    start = tmp_path / "start.png"
    write_image(end, (10, 10, 10))
    write_image(start, (240, 240, 240))

    report = check_boundary_samples(
        [make_sample(end, "F01", "boundary_end"), make_sample(start, "F02", "boundary_start")]
    )

    assert report.status == "warning"
    assert "边界平均亮度变化较大" in report.checks[0].messages
