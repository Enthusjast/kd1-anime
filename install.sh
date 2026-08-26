#!/usr/bin/env bash
# kd1-anime 一站式安装器：Ubuntu/HPC，无 sudo。
set -Eeuo pipefail
umask 077

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*" >&2; }
info() { echo -e "${CYAN}[→]${NC} $*"; }

ENV_NAME="${KD1_ANIME_ENV_NAME:-manim_env}"
MANIM_SPEC="${KD1_ANIME_MANIM_SPEC:-manim>=0.20,<0.21}"
KNOWN_CONDA_BASE="${KD1_ANIME_CONDA_BASE:-}"
LIVE_REPO="https://mirrors.ustc.edu.cn/CTAN/systems/texlive/tlnet"
TEXLIVE_HOME="$HOME/texlive"
case "${KD1_ANIME_TEXLIVE_PLATFORM:-$(uname -m)}" in
    x86_64) TEXLIVE_PLATFORM="x86_64-linux" ;;
    aarch64|arm64) TEXLIVE_PLATFORM="aarch64-linux" ;;
    *-linux) TEXLIVE_PLATFORM="${KD1_ANIME_TEXLIVE_PLATFORM}" ;;
    *)
        err "不支持的 TeX Live 平台: $(uname -m); 可设置 KD1_ANIME_TEXLIVE_PLATFORM 覆盖"
        exit 1
        ;;
esac
if [[ ! "$TEXLIVE_PLATFORM" =~ ^[A-Za-z0-9_-]+$ ]]; then
    err "KD1_ANIME_TEXLIVE_PLATFORM 包含不安全字符: $TEXLIVE_PLATFORM"
    exit 1
fi
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/kd1-anime"
CONFIG_FILE="$CONFIG_DIR/.env"
USER_BIN_DIR="${KD1_ANIME_USER_BIN_DIR:-$HOME/.local/bin}"
REQUIRE_CHECKSUM="${KD1_ANIME_REQUIRE_CHECKSUM:-0}"
case "$REQUIRE_CHECKSUM" in
    0|1) ;;
    *)
        err "KD1_ANIME_REQUIRE_CHECKSUM 只能是 0 或 1"
        exit 1
        ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE=""
TEX_BIN=""
CONDA_ENV_DIR=""

cleanup_dirs=()
cleanup() {
    for directory in "${cleanup_dirs[@]:-}"; do
        if [ -n "$directory" ]; then
            rm -rf "$directory"
        fi
    done
    return 0
}
trap cleanup EXIT

find_conda() {
    unset PYTHONHOME
    if command -v module >/dev/null 2>&1; then
        # 按集群要求优先加载模块；失败时继续尝试已有 conda 和已知路径。
        module load python3.12/3.12 2>/dev/null || true
        module load miniconda/py312 2>/dev/null || true
        unset PYTHONHOME
    fi
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    fi
    if [ -z "$CONDA_BASE" ] && [ -n "$KNOWN_CONDA_BASE" ] && [ -x "$KNOWN_CONDA_BASE/bin/conda" ]; then
        export PATH="$KNOWN_CONDA_BASE/bin:$PATH"
        CONDA_BASE="$KNOWN_CONDA_BASE"
    fi
    if [ -z "$CONDA_BASE" ] || [ ! -x "$CONDA_BASE/bin/conda" ]; then
        err "无法定位 conda；请设置 KD1_ANIME_CONDA_BASE 或联系管理员。"
        exit 1
    fi
    export PATH="$CONDA_BASE/bin:$PATH"
    unset PYTHONHOME
    log "conda: $CONDA_BASE"
}

conda_run() { PYTHONHOME= command conda "$@"; }
env_python() { PYTHONHOME= command conda run -n "$ENV_NAME" --no-capture-output python "$@"; }
env_pip() { PYTHONHOME= command conda run -n "$ENV_NAME" --no-capture-output pip "$@"; }

tex_candidate_bins() {
    local candidate path_tool root seen=":"

    # Respect an administrator-provided module/PATH before inspecting common roots.
    path_tool="$(type -P kpsewhich 2>/dev/null || true)"
    if [ -n "$path_tool" ]; then
        candidate="$(dirname "$path_tool")"
        printf '%s\n' "$candidate"
        seen="${seen}${candidate}:"
    fi

    for root in /usr/local/texlive "$TEXLIVE_HOME"; do
        [ -d "$root" ] || continue
        while IFS= read -r candidate; do
            case "$seen" in
                *":$candidate:"*) continue ;;
            esac
            [ -x "$candidate/kpsewhich" ] || continue
            printf '%s\n' "$candidate"
            seen="${seen}${candidate}:"
        done < <(
            find "$root" -mindepth 3 -maxdepth 3 -type d \
                -path "*/bin/$TEXLIVE_PLATFORM" 2>/dev/null | sort -V -r
        )
    done
}

tex_environment_complete() {
    local tex_bin="${1:-$TEX_BIN}" report="${2:-0}" missing=0 command_name file
    local required_files=(
        standalone.cls preview.sty amsmath.sty amssymb.sty amsfonts.sty
        mathtools.sty xcolor.sty tikz.sty physics.sty cancel.sty ulem.sty
        ctex.sty xeCJK.sty fontspec.sty
    )

    for command_name in xelatex kpsewhich dvisvgm; do
        if [ ! -x "$tex_bin/$command_name" ]; then
            [ "$report" = 1 ] && err "缺少 TeX 命令: $command_name"
            missing=1
        fi
    done
    if [ ! -x "$tex_bin/kpsewhich" ]; then
        return 1
    fi
    for file in "${required_files[@]}"; do
        if ! "$tex_bin/kpsewhich" "$file" 2>/dev/null | grep -q .; then
            [ "$report" = 1 ] && err "缺少 LaTeX 文件: $file"
            missing=1
        fi
    done
    return "$missing"
}

find_complete_tex_bin() {
    local candidate
    while IFS= read -r candidate; do
        if tex_environment_complete "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <(tex_candidate_bins)
    return 1
}

tex_bin_is_manageable() {
    local tex_bin="$1" tex_root
    [ -x "$tex_bin/tlmgr" ] && [ -x "$tex_bin/kpsewhich" ] || return 1
    tex_root="$("$tex_bin/kpsewhich" -var-value=TEXMFROOT 2>/dev/null || true)"
    [ -n "$tex_root" ] && [ -d "$tex_root" ] && [ -w "$tex_root" ]
}

find_manageable_tex_bin() {
    local candidate
    while IFS= read -r candidate; do
        if tex_bin_is_manageable "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <(tex_candidate_bins)
    return 1
}

download() {
    local url="$1" output="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --connect-timeout 30 "$url" -o "$output"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --timeout=30 -O "$output" "$url"
    else
        err "需要 curl 或 wget"
        return 1
    fi
}

install_texlive() {
    local tmp installer release root expected actual
    tmp="$(mktemp -d)"
    cleanup_dirs+=("$tmp")
    info "从 USTC 镜像下载 TeX Live 安装器..."
    download "$LIVE_REPO/install-tl-unx.tar.gz" "$tmp/install-tl.tar.gz"
    expected="${KD1_ANIME_TEXLIVE_INSTALLER_SHA256:-}"
    if [ -n "$expected" ]; then
        expected="${expected,,}"
        [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || {
            err "KD1_ANIME_TEXLIVE_INSTALLER_SHA256 必须是 64 位十六进制 SHA-256"
            exit 1
        }
        actual="$(sha256sum "$tmp/install-tl.tar.gz" | awk '{print $1}')"
        [ "$actual" = "$expected" ] || {
            err "TeX Live 安装器 SHA-256 不匹配"
            err "expected: $expected"
            err "actual:   $actual"
            exit 1
        }
        log "TeX Live 安装器 SHA-256 验证通过"
    elif [ "$REQUIRE_CHECKSUM" = 1 ]; then
        err "安全模式要求设置 KD1_ANIME_TEXLIVE_INSTALLER_SHA256"
        exit 1
    else
        warn "未设置 KD1_ANIME_TEXLIVE_INSTALLER_SHA256；建议固定安装器摘要"
    fi
    tar xzf "$tmp/install-tl.tar.gz" -C "$tmp"
    installer="$(find "$tmp" -maxdepth 1 -type d -name 'install-tl-*' | head -1 || true)"
    [ -n "$installer" ] || { err "解压后未找到 install-tl"; exit 1; }

    # install-tl-YYYYMMDD 的前四位就是当前网络发行年份。
    release="$(basename "$installer" | sed -n 's/^install-tl-\([0-9]\{4\}\).*/\1/p')"
    [ -n "$release" ] || release="$(date +%Y)"
    root="$TEXLIVE_HOME/$release"
    info "安装 TeX Live $release 到 $root"

    cat > "$tmp/texlive.profile" <<PROFILE
selected_scheme scheme-basic
TEXDIR $root
TEXMFCONFIG $root/texmf-config
TEXMFHOME $TEXLIVE_HOME/texmf
TEXMFLOCAL $root/texmf-local
TEXMFSYSCONFIG $root/texmf-config
TEXMFSYSVAR $root/texmf-var
TEXMFVAR $root/texmf-var
option_doc 0
option_src 0
binary_$TEXLIVE_PLATFORM 1
PROFILE
    (cd "$installer" && ./install-tl --profile="$tmp/texlive.profile" --repository="$LIVE_REPO")
    TEX_BIN="$root/bin/$TEXLIVE_PLATFORM"
}

configure_tlmgr_repo() {
    local release live historic
    release="$(
        "$TEX_BIN/tlmgr" --version 2>/dev/null |
            sed -n -e 's/.*TeX Live .*version \([0-9]\{4\}\).*/\1/p' -e 's/.*TeX Live \([0-9]\{4\}\).*/\1/p' |
            head -1
    )"
    if [[ ! "$release" =~ ^[0-9]{4}$ ]]; then
        return 1
    fi
    live="$LIVE_REPO"
    historic="https://ftp.tug.org/historic/systems/texlive/${release}/tlnet-final"

    "$TEX_BIN/tlmgr" option repository "$live" >/dev/null 2>&1 || true
    if "$TEX_BIN/tlmgr" info --only-remote texlive.infra >/dev/null 2>&1; then
        echo "$live"
        return 0
    fi
    "$TEX_BIN/tlmgr" option repository "$historic" >/dev/null 2>&1 || true
    if "$TEX_BIN/tlmgr" info --only-remote texlive.infra >/dev/null 2>&1; then
        echo "$historic"
        return 0
    fi
    return 1
}

missing_tex_packages() {
    local tex_bin="${1:-$TEX_BIN}" package file

    [ -x "$tex_bin/kpsewhich" ] || return 1
    {
        [ -x "$tex_bin/xelatex" ] || printf '%s\n' xetex
        [ -x "$tex_bin/dvisvgm" ] || printf '%s\n' dvisvgm
        while read -r package file; do
            if ! "$tex_bin/kpsewhich" "$file" 2>/dev/null | grep -q .; then
                printf '%s\n' "$package"
            fi
        done <<'PACKAGES'
standalone standalone.cls
preview preview.sty
amsmath amsmath.sty
amsfonts amssymb.sty
amsfonts amsfonts.sty
mathtools mathtools.sty
xcolor xcolor.sty
pgf tikz.sty
physics physics.sty
cancel cancel.sty
ulem ulem.sty
ctex ctex.sty
xecjk xeCJK.sty
fontspec fontspec.sty
PACKAGES
    } | sort -u
}

install_tex_dependencies() {
    local tex_repo
    local packages=()

    [ -x "$TEX_BIN/kpsewhich" ] || return 1
    mapfile -t packages < <(missing_tex_packages "$TEX_BIN")
    if [ "${#packages[@]}" -eq 0 ]; then
        return 0
    fi
    tex_bin_is_manageable "$TEX_BIN" || return 1

    info "配置 TeX Live 仓库并安装缺少的最小依赖"
    tex_repo="$(configure_tlmgr_repo || true)"
    [ -n "$tex_repo" ] || return 1
    log "TeX 仓库: $tex_repo"
    info "待安装 TeX 包: ${packages[*]}"
    "$TEX_BIN/tlmgr" install "${packages[@]}"
}

verify_tex_packages() {
    tex_environment_complete "$TEX_BIN" 1
}

ensure_texlive() {
    local candidate

    candidate="$(find_complete_tex_bin || true)"
    if [ -n "$candidate" ]; then
        TEX_BIN="$candidate"
        log "使用现有 TeX Live (XeLaTeX): $TEX_BIN"
        return 0
    fi

    candidate="$(find_manageable_tex_bin || true)"
    if [ -n "$candidate" ]; then
        TEX_BIN="$candidate"
        warn "现有 TeX Live 缺少 XeLaTeX/Manim 依赖，尝试仅补齐缺包: $TEX_BIN"
        if install_tex_dependencies && tex_environment_complete "$TEX_BIN"; then
            log "现有 TeX Live 依赖已补齐"
            return 0
        fi
        warn "现有 TeX Live 无法补齐依赖，改用用户目录安装"
    else
        info "未找到完整且可用的 XeLaTeX 环境，将安装用户目录版 TeX Live"
    fi

    install_texlive
    install_tex_dependencies || {
        err "无法安装 XeLaTeX/Manim 所需的最小 TeX 依赖"
        return 1
    }
    verify_tex_packages
}

install_python_package() {
    local source ref kind tmp archive expected actual
    if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
        source="$SCRIPT_DIR"
        kind="local"
    else
        ref="${KD1_ANIME_REF:-main}"
        if [[ ! "$ref" =~ ^[A-Za-z0-9._/-]+$ ]] || [[ "$ref" == *".."* ]] || [[ "$ref" == /* ]]; then
            err "KD1_ANIME_REF 包含不安全字符: $ref"
            exit 1
        fi
        if [[ "$ref" == v* ]]; then
            source="https://github.com/Enthusjast/kd1-anime/archive/refs/tags/${ref}.zip"
        else
            source="https://github.com/Enthusjast/kd1-anime/archive/refs/heads/${ref}.zip"
        fi
        kind="remote"
    fi
    if [ "$kind" = local ]; then
        env_pip install -e "$source"
        log "kd1-anime 已以可编辑模式安装"
    else
        tmp="$(mktemp -d)"
        cleanup_dirs+=("$tmp")
        archive="$tmp/kd1-anime.zip"
        info "下载临时源码归档（不会保留源码目录）"
        download "$source" "$archive"
        expected="${KD1_ANIME_ARCHIVE_SHA256:-}"
        if [ -n "$expected" ]; then
            expected="${expected,,}"
            if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
                err "KD1_ANIME_ARCHIVE_SHA256 必须是 64 位十六进制 SHA-256"
                exit 1
            fi
            actual="$(sha256sum "$archive" | awk '{print $1}')"
            if [ "$actual" != "$expected" ]; then
                err "源码归档 SHA-256 不匹配"
                err "expected: $expected"
                err "actual:   $actual"
                exit 1
            fi
            log "源码归档 SHA-256 验证通过"
        else
            if [ "$REQUIRE_CHECKSUM" = 1 ]; then
                err "安全模式要求设置 KD1_ANIME_ARCHIVE_SHA256"
                exit 1
            fi
            warn "未设置 KD1_ANIME_ARCHIVE_SHA256；建议发布安装时固定 tag 并提供摘要"
        fi
        env_pip install --upgrade "$archive"
        log "kd1-anime 已安装"
    fi
}

write_user_config() {
    mkdir -p "$CONFIG_DIR"
    chmod 700 "$CONFIG_DIR"
    if [ -f "$CONFIG_FILE" ]; then
        warn "用户配置已存在，未覆盖: $CONFIG_FILE"
        return
    fi
    cat > "$CONFIG_DIR/.env.example" <<EOF
LLM_API_KEY=sk-your-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=your-model-name
LLM_SEND_MAX_TOKENS=true
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=32768
LLM_MAX_RETRIES=3
LLM_RETRY_BASE_DELAY=2.0
LLM_TIMEOUT_CONNECT=30.0
LLM_TIMEOUT_READ=600.0
LLM_HEALTHCHECK_TIMEOUT=15.0
LLM_SILENT_STREAM=true
LLM_EMPTY_RETRY_MAX_TOKENS=16384
LLM_JSON_REPAIR_ATTEMPTS=2
LLM_USE_JSON_MODE=true
LLM_DEBUG=false
VISUAL_LLM_API_KEY=
VISUAL_LLM_BASE_URL=
VISUAL_LLM_MODEL=
VISUAL_LLM_SEND_MAX_TOKENS=true
VISUAL_LLM_TEMPERATURE=0.0
VISUAL_LLM_MAX_TOKENS=3000
VISUAL_LLM_MAX_RETRIES=3
VISUAL_LLM_RETRY_BASE_DELAY=2.0
VISUAL_LLM_TIMEOUT_CONNECT=30.0
VISUAL_LLM_TIMEOUT_READ=300.0
VISUAL_LLM_HEALTHCHECK_TIMEOUT=20.0
VISUAL_LLM_JSON_REPAIR_ATTEMPTS=1
VISUAL_LLM_USE_JSON_MODE=true
VISUAL_LLM_PARALLEL_WORKERS=2
VISUAL_LLM_DEBUG=false
RAG_ENABLED=false
RAG_INDEX_PATH=~/.cache/kd1-anime/rag/index.sqlite3
RAG_DOCS_DIR=
RAG_EXAMPLES_DIR=
RAG_EMBEDDING_API_KEY=
RAG_EMBEDDING_BASE_URL=
RAG_EMBEDDING_MODEL=
RAG_EMBEDDING_TIMEOUT=60.0
RAG_EMBEDDING_BATCH_SIZE=32
RAG_RERANK_API_KEY=
RAG_RERANK_BASE_URL=
RAG_RERANK_MODEL=
RAG_RERANK_TIMEOUT=60.0
RAG_TOP_K=8
RAG_RERANK_TOP_N=4
RAG_MAX_CONTEXT_CHARS=12000
RAG_CHUNK_SIZE=1800
RAG_CHUNK_OVERLAP=200
RAG_PARALLEL_WORKERS=2
SLURM_PARTITION=
SLURM_QOS=
SLURM_ACCOUNT=
SLURM_CONDA_ENV=$ENV_NAME
SLURM_CONDA_BASE=$CONDA_BASE
SLURM_TIME_LIMIT=01:00:00
SLURM_CPUS_PER_TASK=4
SLURM_MEM_GB=
SLURM_GPU_TYPE=
SLURM_GPU_COUNT=1
SLURM_MAX_IN_FLIGHT=0
SLURM_SUBMIT_RETRIES=3
SLURM_SUBMIT_RETRY_DELAY=2.0
SLURM_CONTAINER_IMAGE=
SLURM_REQUIRE_CONTAINER=false
SLURM_CONTAINER_DISABLE_NETWORK=false
MAX_INFRA_RETRIES=2
MANIM_RENDERER=cairo
MANIM_QUALITY=h
MANIM_PIXEL_WIDTH=1920
MANIM_PIXEL_HEIGHT=1080
MANIM_FRAME_RATE=60
MANIM_OPENGL_PLATFORM=egl
ALLOW_PARTIAL_OUTPUT=false
OVERWRITE_OUTPUT=false
TRANSITION_TYPE=fade
TRANSITION_DURATION=0.5
MERGE_VIDEO_CODEC=libx264
MERGE_VIDEO_PRESET=medium
MERGE_VIDEO_CRF=18
MERGE_AUDIO_SAMPLE_RATE=48000
MERGE_AUDIO_CHANNEL_LAYOUT=stereo
LLM_PARALLEL_WORKERS=4
MAX_REVIEW_ROUNDS=5
MAX_CONTINUITY_FIX_ROUNDS=2
SKIP_REVIEW=false
MAX_FIX_ATTEMPTS=5
MAX_FIX_IDENTICAL_ERRORS=3
MAX_CLARIFY_ROUNDS=12
MAX_SCENES=12
MAX_PROMPT_CHARS=50000
MAX_CLARIFY_CONTEXT_CHARS=40000
MAX_LOG_CHARS=30000
CODE_VALIDATION_ATTEMPTS=3
MONITOR_POLL_INTERVAL=10
MONITOR_QUEUE_TIMEOUT=3600
MONITOR_RUN_TIMEOUT=3600
MONITOR_UNKNOWN_TIMEOUT=300
MONITOR_ARTIFACT_GRACE=60
MONITOR_MAX_UNKNOWN=5
LOG_TAIL_LINES=80
ENABLE_AUTO_EVAL=false
ENABLE_VISUAL_EVAL=false
EVAL_THRESHOLD=3.5
MAX_EVAL_ROUNDS=2
VISUAL_EVAL_FRAME_COUNT=6
VISUAL_EVAL_THRESHOLD=3.5
MAX_VISUAL_FIX_ATTEMPTS=2
WORKSPACE_DIR=workspace
OUTPUT_FILE=output_final.mp4
EOF
    cp "$CONFIG_DIR/.env.example" "$CONFIG_FILE"
    chmod 600 "$CONFIG_FILE"
    log "已创建用户配置: $CONFIG_FILE"
    warn "首次运行前请编辑该文件并填写 LLM_API_KEY、LLM_BASE_URL 和 LLM_MODEL"
}

install_command_wrappers() {
    local cli_target env_target cli_tmp env_tmp env_dir_q tex_bin_q conda_sh_q env_name_q
    if [[ "$USER_BIN_DIR" != /* ]]; then
        err "用户命令目录必须是绝对路径: $USER_BIN_DIR"
        return 1
    fi
    if [ -z "${CONDA_ENV_DIR:-}" ] || [ ! -x "$CONDA_ENV_DIR/bin/kd1-anime" ]; then
        err "conda 环境中缺少 kd1-anime 入口: ${CONDA_ENV_DIR:-<unknown>}/bin/kd1-anime"
        return 1
    fi
    if [ ! -r "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        err "缺少 conda 激活脚本: $CONDA_BASE/etc/profile.d/conda.sh"
        return 1
    fi
    if [ -z "$TEX_BIN" ] || [ ! -d "$TEX_BIN" ]; then
        err "TeX Live bin 目录无效: ${TEX_BIN:-<empty>}"
        return 1
    fi

    mkdir -p "$USER_BIN_DIR"
    cli_target="$USER_BIN_DIR/kd1-anime"
    env_target="$USER_BIN_DIR/manim-env"
    cli_tmp="$(mktemp "$USER_BIN_DIR/.kd1-anime.XXXXXX")"
    env_tmp="$(mktemp "$USER_BIN_DIR/.manim-env.XXXXXX")"
    cleanup_dirs+=("$cli_tmp" "$env_tmp")
    printf -v env_dir_q '%q' "$CONDA_ENV_DIR"
    printf -v tex_bin_q '%q' "$TEX_BIN"
    printf -v conda_sh_q '%q' "$CONDA_BASE/etc/profile.d/conda.sh"
    printf -v env_name_q '%q' "$ENV_NAME"

    cat > "$cli_tmp" <<EOF
#!/usr/bin/env bash
# Generated by kd1-anime install.sh
set -euo pipefail
ENV_DIR=$env_dir_q
TEX_BIN=$tex_bin_q
unset PYTHONHOME
export PATH="\$TEX_BIN:\$ENV_DIR/bin:\${PATH:-}"
exec "\$ENV_DIR/bin/kd1-anime" "\$@"
EOF

    cat > "$env_tmp" <<EOF
#!/usr/bin/env bash
# Generated by kd1-anime install.sh
set -eo pipefail
CONDA_SH=$conda_sh_q
ENV_NAME=$env_name_q
TEX_BIN=$tex_bin_q
unset PYTHONHOME
source "\$CONDA_SH"
conda activate "\$ENV_NAME"
export PATH="\$TEX_BIN:\$PATH"
if [ "\$#" -gt 0 ]; then
    exec "\$@"
fi
target_shell="\${SHELL:-/bin/bash}"
if [ ! -x "\$target_shell" ]; then
    target_shell=/bin/bash
fi
exec "\$target_shell" -i
EOF

    chmod 700 "$cli_tmp" "$env_tmp"
    bash -n "$cli_tmp"
    bash -n "$env_tmp"
    mv -f "$cli_tmp" "$cli_target"
    mv -f "$env_tmp" "$env_target"
    log "用户命令已安装到 $USER_BIN_DIR"
}

configure_shell_rc() {
    local rc_file="$1" conda_sh_q env_name_q user_bin_q
    printf -v conda_sh_q '%q' "$CONDA_BASE/etc/profile.d/conda.sh"
    printf -v env_name_q '%q' "$ENV_NAME"
    printf -v user_bin_q '%q' "$USER_BIN_DIR"
    touch "$rc_file"
    sed -i '/^# >>> kd1-anime/,/^# <<< kd1-anime/d' "$rc_file" || true
    cat >> "$rc_file" <<EOF
# >>> kd1-anime install.sh >>>
unset PYTHONHOME
_KD1_ANIME_USER_BIN=$user_bin_q
case ":\$PATH:" in
    *":\$_KD1_ANIME_USER_BIN:"*) ;;
    *) export PATH="\$_KD1_ANIME_USER_BIN:\$PATH" ;;
esac
unset _KD1_ANIME_USER_BIN
manim-env() {
    unset PYTHONHOME
    source $conda_sh_q
    conda activate $env_name_q
}
# 只在用户没有显式设置语言时提供 C.UTF-8 默认值，不覆盖用户已有的
# 中文或其他 UTF-8 locale 配置。
if [ -z "\${LANG:-}" ]; then
    export LANG=C.UTF-8
fi
# <<< kd1-anime install.sh <<<
EOF
}

print_completion() {
    printf '\n%b安装完成%b\n' "$GREEN" "$NC"
    printf '%s\n' \
        "1. 进入 Manim 环境: manim-env" \
        "2. 启动程序: kd1-anime" \
        "3. 编辑配置: $CONFIG_FILE" \
        "   命令目录: $USER_BIN_DIR"
}

main() {
echo -e "${CYAN}=== kd1-anime 环境安装 ===${NC}"
find_conda

info "创建/检查 conda 环境 $ENV_NAME"
if ! conda_run env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda_run create -n "$ENV_NAME" python=3.12 -y
fi

info "安装 Manim 和 FFmpeg"
conda_run install -n "$ENV_NAME" -c conda-forge "$MANIM_SPEC" ffmpeg -y
# 中文字体是 Coder 默认输出的一部分；字体包失败不应阻断其余安装。
conda_run install -n "$ENV_NAME" -c conda-forge fonts-noto-cjk -y \
    || warn "未能安装 Noto CJK 字体，中文 Text 可能无法正确显示"

info "验证 Manim / FFmpeg"
env_python -c 'import manim; print("Manim", manim.__version__)'
env_python -c 'import shutil, subprocess; p=shutil.which("ffmpeg"); assert p; print(subprocess.run([p,"-version"],capture_output=True,text=True).stdout.splitlines()[0])'

ensure_texlive || { err "TeX Live 配置失败"; exit 1; }
export PATH="$TEX_BIN:$PATH"
log "XeLaTeX / CJK 依赖验证通过"

info "配置 conda 激活钩子"
CONDA_ENV_DIR="$(conda_run run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
ACTIVATE_DIR="$CONDA_ENV_DIR/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
cat > "$ACTIVATE_DIR/kd1-anime.sh" <<EOF
#!/usr/bin/env bash
export PATH="$TEX_BIN:\$PATH"
unset PYTHONHOME
EOF
chmod +x "$ACTIVATE_DIR/kd1-anime.sh"

install_python_package

info "安装用户命令"
install_command_wrappers

info "配置 shell 激活命令"
configure_shell_rc "$HOME/.bashrc"
if [ -f "$HOME/.zshrc" ] || command -v zsh >/dev/null 2>&1; then
    configure_shell_rc "$HOME/.zshrc"
fi

write_user_config

info "最终验证"
env_python -c 'from manim import *; print("Python + Manim: OK")'
"$TEX_BIN/xelatex" --version >/dev/null 2>&1 \
    && log "XeLaTeX: OK" || { err "XeLaTeX: FAIL"; exit 1; }
env_python -c 'import shutil; assert shutil.which("ffmpeg"); print("FFmpeg: OK")'
"$USER_BIN_DIR/kd1-anime" version

print_completion
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
