"""终端可显示的 kd1-anime Logo。

原始 PNG 在终端中不能直接稳定显示，因此安装包内保存为低分辨率的
Unicode 半块 + ANSI 真彩色表示。该模块不依赖 Pillow，既可由 Rich TUI
转换为 ``Text``，也可由 install.sh 通过 ``python -m kd1_anime.logo`` 直接输出。
"""

from __future__ import annotations

import sys
from typing import TextIO

_PALETTE: tuple[tuple[int, int, int] | None, ...] = (
    None,
    (0, 113, 188),
    (117, 76, 36),
    (102, 102, 102),
    (153, 153, 153),
)

# 每个字符编码一列的两个像素：上半块、下半块。0 表示透明背景。
_LOGO_ROWS = (
    "0000000000000000000000000002022211010100000000000000000000000000",
    "0000000000000000000002022222222211111111010100000000000000000000",
    "0000000000000202222222222222222211111111111111110101000000000000",
    "0000000202222222222222222222222211111111111111111111110101000000",
    "0222222222222222222222222220200000101011111111111111111111111101",
    "2222222222222222222220000000000303000000001011111111111111111111",
    "2222222222222220000000030333333333333303030000001011111111111111",
    "2222222222220000000333333333333333333333333304000000111111111111",
    "2222222222220000003333333333333333333434444444000000111111111111",
    "2222222222220000003333333333333334444444444444000000111111111111",
    "2222222222220000003333333333333344444444444444000000111111111111",
    "2222222222220000003033333333333344444444444440000000111111111111",
    "2222222223213101000000303033333344444440400000000111111111111111",
    "2223213111111111111101000000003040000000000111111111111111111111",
    "1011111111111111111111111101010000010111111111111111111111111110",
    "0000001010111111111111111111111111111111111111111111111010000000",
    "0000000000000010111111111111111111111111111111111000000000000000",
    "0000000000000000000010101111111111111111101000000000000000000000",
    "0000000000000000000000000010101111101000000000000000000000000000",
)


def _color(index: int, *, background: bool = False) -> str:
    color = _PALETTE[index]
    if color is None:
        return ""
    channel = 48 if background else 38
    return f"\x1b[{channel};2;{color[0]};{color[1]};{color[2]}m"


def ansi_logo() -> str:
    """返回可直接写入支持 ANSI 真彩色终端的 Logo。"""

    lines: list[str] = []
    for row in _LOGO_ROWS:
        cells: list[str] = []
        for offset in range(0, len(row), 2):
            top = int(row[offset])
            bottom = int(row[offset + 1])
            if top == 0 and bottom == 0:
                cells.append(" ")
            elif bottom == 0:
                cells.append(f"{_color(top)}▀")
            elif top == 0:
                cells.append(f"{_color(bottom)}▄")
            else:
                cells.append(f"{_color(top)}{_color(bottom, background=True)}▀")
        lines.append("".join(cells) + "\x1b[0m")
    return "\n".join(lines) + "\n"


ANSI_LOGO = ansi_logo()


def print_logo(stream: TextIO | None = None) -> None:
    """把 Logo 写入终端；供安装器和非 Rich 环境使用。"""

    output = stream or sys.stdout
    output.write(ANSI_LOGO)
    output.flush()


if __name__ == "__main__":  # pragma: no cover - exercised by install.sh
    print_logo()
