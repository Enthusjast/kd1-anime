"""不依赖 LLM 的场景边界帧健康检查。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from kd1_anime.eval.visual_eval import FrameSample

BoundaryStatus = Literal["passed", "warning", "failed", "unknown"]


@dataclass(frozen=True, slots=True)
class BoundaryCheck:
    boundary_id: str
    status: BoundaryStatus
    metrics: dict[str, float | int | str]
    messages: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BoundaryCheckReport:
    status: BoundaryStatus
    checks: tuple[BoundaryCheck, ...]
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "checks": [check.to_dict() for check in self.checks],
            "reason": self.reason,
        }


def _frame_metrics(path: Path) -> tuple[int, int, float, float]:
    """读取缩小 RGB 图像，返回宽、高、黑帧比例和平均亮度。"""

    from PIL import Image, ImageStat

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        small = rgb.resize((32, 18))
        get_pixels = getattr(small, "get_flattened_data", small.getdata)
        pixels = list(get_pixels())
        dark = sum(1 for red, green, blue in pixels if max(red, green, blue) <= 8)
        mean = ImageStat.Stat(small).mean
        brightness = (0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2]) / 255.0
        return width, height, dark / max(1, len(pixels)), brightness


def check_boundary_samples(samples: list[FrameSample]) -> BoundaryCheckReport:
    """检查每个相邻场景的首尾帧尺寸、黑帧和亮度突变。"""

    if not samples:
        return BoundaryCheckReport(status="unknown", checks=(), reason="没有边界帧")
    try:
        import PIL  # noqa: F401  # 仅探测可选运行时依赖
    except ImportError:
        return BoundaryCheckReport(status="unknown", checks=(), reason="未安装 Pillow")

    grouped: dict[str, dict[str, FrameSample]] = {}
    for sample in samples:
        if not sample.boundary_id or sample.role not in {"boundary_start", "boundary_end"}:
            continue
        grouped.setdefault(sample.boundary_id, {})[sample.role] = sample
    if not grouped:
        return BoundaryCheckReport(status="unknown", checks=(), reason="样本没有有效边界标识")

    checks: list[BoundaryCheck] = []
    for boundary_id, pair in sorted(grouped.items()):
        if set(pair) != {"boundary_start", "boundary_end"}:
            checks.append(
                BoundaryCheck(
                    boundary_id=boundary_id,
                    status="unknown",
                    metrics={},
                    messages=("缺少首帧或尾帧",),
                )
            )
            continue
        try:
            end_metrics = _frame_metrics(pair["boundary_end"].path)
            start_metrics = _frame_metrics(pair["boundary_start"].path)
        except (OSError, ValueError, RuntimeError) as exc:
            checks.append(
                BoundaryCheck(
                    boundary_id=boundary_id,
                    status="unknown",
                    metrics={},
                    messages=(f"边界图片无法读取: {type(exc).__name__}",),
                )
            )
            continue
        end_width, end_height, end_black, end_brightness = end_metrics
        start_width, start_height, start_black, start_brightness = start_metrics
        messages: list[str] = []
        status: BoundaryStatus = "passed"
        if (end_width, end_height) != (start_width, start_height):
            status = "failed"
            messages.append("相邻场景帧尺寸不一致")
        if max(end_black, start_black) >= 0.995:
            status = "failed"
            messages.append("边界存在近似全黑帧")
        brightness_delta = abs(end_brightness - start_brightness)
        if brightness_delta >= 0.45 and status == "passed":
            status = "warning"
            messages.append("边界平均亮度变化较大")
        checks.append(
            BoundaryCheck(
                boundary_id=boundary_id,
                status=status,
                metrics={
                    "end_width": end_width,
                    "end_height": end_height,
                    "start_width": start_width,
                    "start_height": start_height,
                    "end_black_ratio": round(end_black, 6),
                    "start_black_ratio": round(start_black, 6),
                    "brightness_delta": round(brightness_delta, 6),
                },
                messages=tuple(messages),
            )
        )
    statuses = {check.status for check in checks}
    if "failed" in statuses:
        status = "failed"
    elif "warning" in statuses:
        status = "warning"
    elif checks and all(item.status == "unknown" for item in checks):
        status = "unknown"
    else:
        status = "passed"
    return BoundaryCheckReport(status=status, checks=tuple(checks))


__all__ = ["BoundaryCheck", "BoundaryCheckReport", "BoundaryStatus", "check_boundary_samples"]
