"""源码仓库兼容入口；安装后的命令使用 kd1_anime.cli:app。"""

import sys
from pathlib import Path

# 允许从源码仓库直接执行 ``python main.py``，不要求先安装 editable
# package；安装后的 console script 仍然直接走 kd1_anime.cli:app。
_src = Path(__file__).resolve().parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from kd1_anime.cli import app  # noqa: E402

if __name__ == "__main__":
    app()
