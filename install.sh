#!/bin/bash
# ============================================================
# install.sh — 全自动 Manim 环境安装脚本
# 适用于 Ubuntu HPC 集群 (无 sudo 权限)
#
# 安装内容:
#   - Conda 环境 "manim_env" (Python 3.12)
#   - Manim Community Edition
#   - FFmpeg
#   - TeX Live (最小安装, 仅 Manim 所需的 LaTeX 包)
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
# Step 0: 初始化 module 系统
# ----------------------------------------------------------
info "初始化 module 系统..."
if ! type module &>/dev/null; then
    for f in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [ -f "$f" ]; then source "$f" && break; fi
    done
fi
if ! type module &>/dev/null; then
    err "module 命令不可用"
    exit 1
fi

# ----------------------------------------------------------
# Step 1: 加载系统模块
# ----------------------------------------------------------
info "加载 python3.12/3.12 ..."
module load python3.12/3.12

info "加载 miniconda/py312 ..."
module load miniconda/py312

if ! command -v conda &>/dev/null; then
    err "加载 miniconda 模块后仍找不到 conda"
    exit 1
fi
log "conda $(conda --version)"

# ----------------------------------------------------------
# Step 2: 创建/复用 conda 环境 (幂等, 不重复销毁)
# ----------------------------------------------------------
ENV_NAME="manim_env"

if conda env list | grep -q "^${ENV_NAME} "; then
    warn "conda 环境 '${ENV_NAME}' 已存在, 跳过创建 (如需重建: conda env remove -n ${ENV_NAME})"
else
    info "创建 conda 环境 '${ENV_NAME}' (Python 3.12)..."
    conda create -n "${ENV_NAME}" python=3.12 -y
fi
log "conda 环境 '${ENV_NAME}' 就绪"

# 激活环境
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# 守卫: 确认 CONDA_PREFIX 已设置, 避免后续误写系统目录
if [ -z "${CONDA_PREFIX:-}" ]; then
    err "CONDA_PREFIX 未设置, conda activate 似乎失败"
    exit 1
fi

# ----------------------------------------------------------
# Step 3: 安装/补装 Manim + FFmpeg (通过 conda-forge, 幂等)
# ----------------------------------------------------------
info "确保 Manim Community Edition 和 FFmpeg 已安装..."
conda install -c conda-forge manim ffmpeg -y

log "Manim: $(manim --version 2>/dev/null || echo '安装检查失败')"
log "FFmpeg: $(ffmpeg -version 2>/dev/null | head -1 || echo '安装检查失败')"

# ----------------------------------------------------------
# Step 4: 探测/安装最小 TeX Live (到 ~/texlive, 无需 sudo)
# ----------------------------------------------------------
TEXLIVE_DIR="$HOME/texlive"

# 动态探测已安装的 TeX Live 年份, 兼容多年份共存
detect_tex_year() {
    ls -d "$HOME"/texlive/20*/ 2>/dev/null | sort -V | tail -1 | xargs -r basename 2>/dev/null
}
TEXLIVE_YEAR="$(detect_tex_year)"
if [ -z "${TEXLIVE_YEAR}" ]; then
    TEXLIVE_YEAR="2024"
fi
TEX_BIN="${TEXLIVE_DIR}/${TEXLIVE_YEAR}/bin/x86_64-linux"
info "TeX Live 年份: ${TEXLIVE_YEAR} (TEX_BIN=${TEX_BIN})"

# 历史 frozen 仓库 (年份不匹配时回退用)
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

    # 安全解析解压目录 (glob 可能匹配 0 或多个)
    local INSTALL_DIR
    INSTALL_DIR=$(find "${TMPDIR}" -maxdepth 1 -type d -name 'install-tl-*' | head -1)
    if [ -z "${INSTALL_DIR}" ]; then
        err "解压后未找到 install-tl 目录, 归档布局异常"
        exit 1
    fi
    cd "${INSTALL_DIR}"

    # 生成安装配置文件 (scheme-basic: 最小安装, 不装文档和源码)
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
    ./install-tl \
        --profile=texlive.profile \
        --repository="${LIVE_REPO}"
}

if [ -x "${TEX_BIN}/pdflatex" ]; then
    warn "TeX Live 已安装于 ${TEXLIVE_DIR}/${TEXLIVE_YEAR}, 跳过基础安装"
else
    install_texlive
    log "TeX Live 基础安装完成"
fi

# 确保 tlmgr 在 PATH
export PATH="${TEX_BIN}:$PATH"
if ! command -v tlmgr &>/dev/null; then
    err "未找到 tlmgr (TEX_BIN=${TEX_BIN})"
    exit 1
fi

# ----------------------------------------------------------
# Step 5: 安装 Manim 所需的 LaTeX 包
# ----------------------------------------------------------
info "安装 Manim 所需的 LaTeX 包..."

# standalone   — 独立数学公式文档类
# preview      — standalone 的裁剪依赖
# amsmath      — 核心数学排版
# amssymb      — 数学符号
# amsfonts     — 数学字体
# mathtools    — 扩展数学工具
# xcolor       — 颜色支持
# pgf          — TikZ 图形 (部分 Manim 功能依赖)
# dvisvgm      — DVI → SVG 转换器 (Manim 渲染核心)
# physics      — 物理符号 (bra-kel, 微分算子等)
# cancel       — 取消线 (分数线上的删除线)
# ulem         — 下划线/删除线
# xparse       — LaTeX3 命令定义

TL_PACKAGES=(
    standalone preview amsmath amssymb amsfonts mathtools
    xcolor pgf dvisvgm physics cancel ulem xparse
)

# 先尝试当前仓库; 若年份不匹配则切换到历史 frozen 仓库重试
tlmgr_install_repo() {
    local repo="$1"
    tlmgr option repository "$repo" >/dev/null 2>&1 || true
    tlmgr install "${TL_PACKAGES[@]}"
}

if ! tlmgr_install_repo "${LIVE_REPO}"; then
    warn "当前仓库安装失败 (可能年份不匹配), 切换到历史 frozen 仓库重试..."
    if ! tlmgr_install_repo "${HISTORIC_REPO}"; then
        err "LaTeX 包安装失败, 请检查网络或手动运行: tlmgr install ${TL_PACKAGES[*]}"
        exit 1
    fi
fi

log "LaTeX 包安装完成"

# 验证
log "pdflatex: $(pdflatex --version 2>/dev/null | head -1 || echo '检查失败')"
log "dvisvgm:  $(dvisvgm --version 2>/dev/null | head -1 || echo '检查失败')"

# ----------------------------------------------------------
# Step 6: 配置 conda activate 时自动设置 PATH (动态解析年份)
# ----------------------------------------------------------
info "配置 conda 激活脚本..."

CONDA_ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${CONDA_ACTIVATE_DIR}"

cat > "${CONDA_ACTIVATE_DIR}/texlive.sh" << 'ACTIVATE'
#!/bin/bash
# install.sh 自动生成 — conda activate 时自动加入最新 TeX Live PATH
# 动态解析 ~/texlive 下最新年份, 避免硬编码年份过期
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

log "激活脚本已写入 ${CONDA_ACTIVATE_DIR}/texlive.sh"

# ----------------------------------------------------------
# 完成
# ----------------------------------------------------------
cd "$HOME"
echo ""
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  安装完成!${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo ""
echo -e "激活环境:    ${CYAN}conda activate manim_env${NC}"
echo -e "验证 Manim:  ${CYAN}manim --version${NC}"
echo -e "验证 LaTeX:  ${CYAN}pdflatex --version${NC}"
echo -e "验证 FFmpeg: ${CYAN}ffmpeg -version${NC}"
echo -e "快速测试:    ${CYAN}python -c \"from manim import *; print('OK')\"${NC}"
echo ""
