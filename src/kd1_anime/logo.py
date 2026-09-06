"""终端可显示的 kd1-anime ASCII Logo。

原始 PNG 已在构建时降采样为纯 ASCII 字符并硬编码在这里，因此 TUI、安装器
和没有真彩色能力的普通终端都能稳定显示，不依赖 Pillow 或其它图片库。
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


def print_logo(stream: TextIO | None = None) -> None:
    """把纯 ASCII Logo 写入终端；供安装器和 Rich TUI 使用。"""

    output = stream or sys.stdout
    output.write(ASCII_LOGO)
    output.flush()


if __name__ == "__main__":  # pragma: no cover - exercised by install.sh
    print_logo()
