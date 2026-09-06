"""终端可显示的彩色 ASCII Logo。

原始 PNG 已在构建时降采样为纯 ASCII 字符并硬编码在这里，因此 TUI、安装器
和没有真彩色能力的普通终端都能稳定显示。字符与颜色的对应关系也在这里
固定编码，不依赖 Pillow 或其它图片库。
"""

from __future__ import annotations

import sys
from typing import TextIO

ASCII_LOGO = """             ###@@@
          ######@@@@@@
      ##########@@@@@@@@@@
   #############@@@@@@@@@@@@@
#############      @@@@@@@@@@@@@
##########    ++++    @@@@@@@@@@
#######    ++++++++++    @@@@@@@
######   ++++++++++++..  @@@@@@@
#######  +++++++++.....  @@@@@@@
#######  +++++++.......  @@@@@@@
######   +++++++.......  @@@@@@@
######+    +++++.....    @@@@@@@
##++@@@@@@    ++..    @@@@@@@@@@
 +@@@@@@@@@@@      @@@@@@@@@@@@@
   @@@@@@@@@@@@@@@@@@@@@@@@@@
      @@@@@@@@@@@@@@@@@@@@
          @@@@@@@@@@@@
             @@@@@@
"""

# Logo 原图的颜色语义：蓝色外框、棕色侧面、深灰/浅灰立方体。
ASCII_LOGO_COLORS: dict[str, tuple[int, int, int]] = {
    "@": (0, 113, 188),
    "#": (117, 76, 36),
    "+": (102, 102, 102),
    ".": (153, 153, 153),
}

BRAND_TITLE = "kd1-anime"
BRAND_SUBTITLE = "LLM 驱动的 Manim 动画 Agent"


def colored_ascii_logo() -> str:
    """给硬编码 ASCII 字符附加 ANSI 真彩色控制码。"""

    lines: list[str] = []
    for line in ASCII_LOGO.splitlines():
        rendered: list[str] = []
        active_color: tuple[int, int, int] | None = None
        for char in line:
            color = ASCII_LOGO_COLORS.get(char)
            if color != active_color:
                if active_color is not None:
                    rendered.append("\x1b[0m")
                if color is not None:
                    rendered.append(f"\x1b[38;2;{color[0]};{color[1]};{color[2]}m")
                active_color = color
            rendered.append(char)
        if active_color is not None:
            rendered.append("\x1b[0m")
        lines.append("".join(rendered))
    return "\n".join(lines) + "\n"


COLORED_ASCII_LOGO = colored_ascii_logo()


def colored_ascii_brand() -> str:
    """在 Logo 右侧附加品牌名称和简介。"""

    lines = COLORED_ASCII_LOGO.rstrip("\n").split("\n")
    title_row = max(0, len(lines) // 2 - 1)
    subtitle_row = min(len(lines) - 1, title_row + 2)
    for index, label in ((title_row, BRAND_TITLE), (subtitle_row, BRAND_SUBTITLE)):
        lines[index] += f"    {label}"
    return "\n".join(lines) + "\n"


COLORED_ASCII_BRAND = colored_ascii_brand()


def print_logo(stream: TextIO | None = None) -> None:
    """把带颜色的 ASCII Logo 写入终端；供安装器和 Rich TUI 使用。"""

    output = stream or sys.stdout
    output.write(COLORED_ASCII_LOGO)
    output.flush()


def print_brand(stream: TextIO | None = None) -> None:
    """把带右侧品牌文字的 Logo 写入终端。"""

    output = stream or sys.stdout
    output.write(COLORED_ASCII_BRAND)
    output.flush()


if __name__ == "__main__":  # pragma: no cover - exercised by install.sh
    if "--brand" in sys.argv[1:]:
        print_brand()
    else:
        print_logo()
