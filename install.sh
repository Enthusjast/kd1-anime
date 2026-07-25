#!/bin/bash
# ============================================================
# install.sh — 全自动 Manim 环境安装脚本
# 适用于 Ubuntu HPC 集群 (无 sudo 权限)
#
# 安装内容:
#   - Manim Community Edition (conda-forge)
#   - FFmpeg
#   - TeX Live (最小安装, 仅 Manim 所需的 LaTeX 包)
#   - ~/.bashrc 快捷激活 alias (manim-env)
#   - 终端编码配置
#
# 用法: bash install.sh  (可重复执行, 已存在的组件会跳过)
# ============================================================
set -eo pipefail

# ----------------------------------------------------------
# 输出工具函数
# ----------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*" >&2; }
info() { echo -e "${CYAN}[→]${NC} $*"; }

# ----------------------------------------------------------
# Step 0: 定位 conda — 先试 module, 不行则用已知直接路径
# ----------------------------------------------------------
ENV_NAME="manim_env"

# 集群已知路径 (module 系统不稳定时的 fallback)
KNOWN_CONDA_BASE="/public/app/miniconda3/py312_24.4.0-0"
KNOWN_CONDA="${KNOWN_CONDA_BASE}/bin/conda"

find_conda() {
    # 策略 1: 通过 module 系统加载
    if type module &>/dev/null 2>&1; then
        info "检测到 module 系统, 尝试加载 miniconda..."
        # 只加载 miniconda (自带 Python 3.12), 不加载 python3.12/3.12,
        # 避免 PYTHONHOME 污染导致 encodings/module 找不到等问题
        module load miniconda/py312 2>/dev/null || true
        if command -v conda &>/dev/null 2>&1; then
            log "通过 module 找到 conda: $(command -v conda)"
            return 0
        fi
        warn "module 加载成功但找不到 conda, 尝试直接路径..."
    fi

    # 策略 2: 直接用已知路径 (绕过损坏的 module 系统)
    if [ -x "${KNOWN_CONDA}" ]; then
        info "使用已知 conda 路径: ${KNOWN_CONDA}"
        # 把 conda 加入 PATH (但不设 PYTHONHOME)
        export PATH="${KNOWN_CONDA_BASE}/bin:$PATH"
        unset PYTHONHOME
        log "conda $(conda --version 2>/dev/null || echo 'unknown')"
        return 0
    fi

    err "无法定位 conda — 既没有 module 系统, 也找不到 ${KNOWN_CONDA}"
    err "请联系集群管理员确认 miniconda 安装路径."
    exit 1
}

# ----------------------------------------------------------
# 用干净的 env 跑 conda 命令 (不受 PYTHONHOME 污染)
# ----------------------------------------------------------
conda_run() {
    # 所有 conda 命令包裹在此, 确保 PYTHONHOME 为空
    PYTHONHOME= command conda "$@"
}

# conda 环境下的 python 调用也用 conda run, 避免 shell activate 的复杂性
env_python() {
    PYTHONHOME= command conda run -n "${ENV_NAME}" --no-capture-output python "$@"
}

env_pip() {
    PYTHONHOME= command conda run -n "${ENV_NAME}" --no-capture-output pip "$@"
}

env_manim() {
    PYTHONHOME= command conda run -n "${ENV_NAME}" --no-capture-output manim "$@"
}

# ----------------------------------------------------------
# 开始安装
# ----------------------------------------------------------
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     kd1-anime 环境安装脚本               ║${NC}"
echo -e "${CYAN}║     Manim + FFmpeg + TeX Live            ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ---- Step 1: 定位 conda ----
find_conda

# ---- Step 2: 创建 conda 环境 (幂等) ----
info "检查 conda 环境 '${ENV_NAME}'..."
if conda_run env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
    warn "conda 环境 '${ENV_NAME}' 已存在, 跳过创建"
else
    info "创建 conda 环境 '${ENV_NAME}' (Python 3.12)..."
    PYTHONHOME= command conda create -n "${ENV_NAME}" python=3.12 -y
fi
log "conda 环境 '${ENV_NAME}' 就绪"

# ---- Step 3: 安装 Manim + FFmpeg (幂等) ----
info "安装/更新 Manim Community Edition 和 FFmpeg..."
PYTHONHOME= command conda install -n "${ENV_NAME}" -c conda-forge manim ffmpeg -y

info "验证 Manim 安装..."
MANIM_VER=$(env_python -c "import manim; print(manim.__version__)" 2>/dev/null || echo "?")
log "Manim ${MANIM_VER}"

info "验证 FFmpeg 安装..."
FFMPEG_VER=$(env_python -c "import subprocess; r=subprocess.run(['ffmpeg','-version'],capture_output=True,text=True); print(r.stdout.split(chr(10))[0])" 2>/dev/null || echo "?")
log "FFmpeg: ${FFMPEG_VER}"

# ---- Step 4: TeX Live ----
TEXLIVE_DIR="$HOME/texlive"

detect_tex_year() {
    ls -d "$HOME"/texlive/20*/ 2>/dev/null | sort -V | tail -1 | xargs -r basename 2>/dev/null
}
TEXLIVE_YEAR="$(detect_tex_year)"
if [ -z "${TEXLIVE_YEAR}" ]; then
    TEXLIVE_YEAR="2024"
fi
TEX_BIN="${TEXLIVE_DIR}/${TEXLIVE_YEAR}/bin/x86_64-linux"
info "TeX Live 年份: ${TEXLIVE_YEAR}"

HISTORIC_REPO="https://ftp.math.utah.edu/pub/tex/historic/system/texlive/${TEXLIVE_YEAR}/tlnet-final"
LIVE_REPO="https://mirrors.ustc.edu.cn/CTAN/systems/texlive/tlnet"

install_texlive() {
    local TMPDIR
    TMPDIR=$(mktemp -d)
    trap 'rm -rf "${TMPDIR}"' RETURN

    info "从 USTC 镜像下载 TeX Live 安装器..."
    wget -q --show-progress \
        -O "${TMPDIR}/install-tl-unx.tar.gz" \
        "${LIVE_REPO}/install-tl-unx.tar.gz"

    tar xzf "${TMPDIR}/install-tl-unx.tar.gz" -C "${TMPDIR}"

    local INSTALL_DIR
    INSTALL_DIR=$(find "${TMPDIR}" -maxdepth 1 -type d -name 'install-tl-*' | head -1)
    if [ -z "${INSTALL_DIR}" ]; then
        err "解压后未找到 install-tl 目录"
        exit 1
    fi
    cd "${INSTALL_DIR}"

    cat > texlive.profile << PROFILE
selected_scheme scheme-basic
TEXDIR ${TEXLIVE_DIR}/${TEXLIVE_YEAR}
TEXMFCONFIG ${TEXLIVE_DIR}/${TEXLIVE_YEAR}/texmf-config
TEXMFHOME ${TEXLIVE_DIR}/texmf
TEXMFLOCAL ${TEXLIVE_DIR}/${TEXLIVE_YEAR}/texmf-local
TEXMFSYSCONFIG ${TEXLIVE_DIR}/${TEXLIVE_YEAR}/texmf-config
TEXMFSYSVAR ${TEXLIVE_DIR}/${TEXLIVE_YEAR}/texmf-var
TEXMFVAR ${TEXLIVE_DIR}/${TEXLIVE_YEAR}/texmf-var
install_doc 0
install_src 0
binary_x86_64-linux 1
PROFILE

    info "运行 TeX Live 安装器 (约 2-5 分钟)..."
    ./install-tl --profile=texlive.profile --repository="${LIVE_REPO}"
}

if [ -x "${TEX_BIN}/pdflatex" ]; then
    warn "TeX Live 已安装于 ${TEXLIVE_DIR}/${TEXLIVE_YEAR}, 跳过基础安装"
else
    install_texlive
    log "TeX Live 基础安装完成"
fi

export PATH="${TEX_BIN}:$PATH"
if ! command -v tlmgr &>/dev/null; then
    err "未找到 tlmgr (TEX_BIN=${TEX_BIN})"
    exit 1
fi

# ---- Step 5: LaTeX 包 ----
info "安装 Manim 所需的 LaTeX 包..."

TL_PACKAGES=(
    standalone preview amsmath amssymb amsfonts mathtools
    xcolor pgf dvisvgm physics cancel ulem xparse
)

tlmgr_install_repo() {
    local repo="$1"
    tlmgr option repository "$repo" >/dev/null 2>&1 || true
    tlmgr install "${TL_PACKAGES[@]}"
}

if ! tlmgr_install_repo "${LIVE_REPO}"; then
    warn "当前仓库安装失败, 切换到历史 frozen 仓库重试..."
    if ! tlmgr_install_repo "${HISTORIC_REPO}"; then
        err "LaTeX 包安装失败, 请手动: tlmgr install ${TL_PACKAGES[*]}"
        exit 1
    fi
fi
log "LaTeX 包安装完成"

# ---- Step 6: conda 激活脚本 (TeX Live PATH) ----
info "配置 conda 激活钩子..."
CONDA_ENV_DIR="$(PYTHONHOME= command conda info --envs 2>/dev/null | grep "^${ENV_NAME} " | awk '{print $NF}')"
CONDA_ACTIVATE_DIR="${CONDA_ENV_DIR}/etc/conda/activate.d"
mkdir -p "${CONDA_ACTIVATE_DIR}"

cat > "${CONDA_ACTIVATE_DIR}/texlive.sh" << 'ACTIVATE'
#!/bin/bash
# install.sh 自动生成 — conda activate 时自动加入最新 TeX Live PATH
_latest_tex_bin() {
    local latest
    latest=$(ls -d "$HOME"/texlive/20*/ 2>/dev/null | sort -V | tail -1)
    if [ -n "${latest}" ]; then
        echo "${latest}bin/x86_64-linux"
    fi
}
TEX_BIN="$(_latest_tex_bin 2>/dev/null)"
if [ -n "${TEX_BIN}" ] && [ -d "${TEX_BIN}" ]; then
    case ":${PATH}:" in
        *":${TEX_BIN}:"*) ;;
        *) export PATH="${TEX_BIN}:$PATH" ;;
    esac
fi
unset -f _latest_tex_bin
unset TEX_BIN
ACTIVATE
chmod +x "${CONDA_ACTIVATE_DIR}/texlive.sh"
log "激活脚本已写入"

# ---- Step 7: ~/.bashrc 快捷设置 ----
info "配置 ~/.bashrc..."

# 标记块 — 幂等, 不会重复追加
MARKER="# >>> kd1-anime install.sh >>>
# <<< kd1-anime install.sh <<<"

# 先移除旧标记块 (如果存在)
if grep -q "kd1-anime install.sh" "$HOME/.bashrc" 2>/dev/null; then
    sed -i '/^# >>> kd1-anime/,/^# <<< kd1-anime/d' "$HOME/.bashrc"
fi

cat >> "$HOME/.bashrc" << BASHCFG
# >>> kd1-anime install.sh >>>
# 清除 PYTHONHOME — 集群 module 会设错该变量, 导致 Python 找不到标准库
unset PYTHONHOME
# 激活 manim 环境
alias manim-env='source ${KNOWN_CONDA_BASE}/bin/activate manim_env'
# 终端编码 — 修复浏览器终端中文 backspace 乱码
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
# <<< kd1-anime install.sh <<<
BASHCFG

log "~/.bashrc 已配置 (manim-env alias + 编码设置)"

# ---- Step 8: 最终验证 ----
echo ""
info "最终验证..."
echo ""

echo -n "  Python + Manim:  "
env_python -c "from manim import *; print('OK')" && log "通过" || err "失败"

echo -n "  LaTeX:           "
pdflatex --version >/dev/null 2>&1 && log "通过" || warn "未安装或不在 PATH"

echo -n "  FFmpeg:          "
ffmpeg -version >/dev/null 2>&1 && log "通过" || warn "未安装或不在 PATH"

# ---- 完成 ----
cd "$HOME"
echo ""
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  安装完成!${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo ""
echo -e "  🚀 下次登录直接输入:     ${CYAN}manim-env${NC}"
echo -e "  然后就可以正常用:        ${CYAN}manim -qh scene.py ClassName${NC}"
echo ""
echo -e "  验证:  ${CYAN}manim-env && manim --version${NC}"
echo -e "  测试:  ${CYAN}manim-env && python -c \"from manim import *; print('OK')\"${NC}"
echo ""
