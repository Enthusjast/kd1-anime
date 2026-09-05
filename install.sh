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
# 文档/示例包与 Manim 运行时保持同一补丁版本；确需其它版本时仍可通过
# KD1_ANIME_MANIM_SPEC 显式覆盖，并应同步更新知识库。
MANIM_SPEC="${KD1_ANIME_MANIM_SPEC:-manim==0.20.1}"
MANIM_KNOWLEDGE_VERSION="0.20.1"
MANIM_KNOWLEDGE_ARCHIVE="resources/manim-0.20.1-knowledge.tar.gz"
MANIM_KNOWLEDGE_ARCHIVE_SHA256="e202d20612f443a40c54b50e5a1e0d27b142e93f606939b04d0db38022c02372"
MANIM_RECIPE_ARCHIVE="resources/manim-0.20.1-recipes.tar.gz"
MANIM_RECIPE_ARCHIVE_SHA256="a1dbcdb0358d5b2f26347e3cc9b4fa77af1171645e720e10d3b04241e99d6142"
CONFIGURE_MODE="${KD1_ANIME_CONFIGURE_MODE:-auto}"
case "$CONFIGURE_MODE" in
    auto|interactive|never) ;;
    *)
        err "KD1_ANIME_CONFIGURE_MODE 只能是 auto、interactive 或 never"
        exit 1
        ;;
esac
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
CONFIG_DIR="$HOME/.kd1-anime"
CONFIG_FILE="$CONFIG_DIR/config.toml"
LEGACY_CONFIG_DIR="$HOME/.config/kd1-anime"
LEGACY_CONFIG_HOME_FILE="$LEGACY_CONFIG_DIR/.env"
LEGACY_CONFIG_EXAMPLE_FILE="$LEGACY_CONFIG_DIR/.env.example"
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

rewrite_legacy_storage_defaults() {
    local file="$1"
    [ -f "$file" ] || return 0
    sed -i \
        -e 's|^RAG_INDEX_PATH=~/.cache/kd1-anime/rag/index.sqlite3$|RAG_INDEX_PATH=~/.kd1-anime/rag/index.sqlite3|' \
        -e 's|^RAG_DOCS_DIR=$|RAG_DOCS_DIR=~/.kd1-anime/knowledge/docs|' \
        -e 's|^RAG_EXAMPLES_DIR=$|RAG_EXAMPLES_DIR=~/.kd1-anime/knowledge/examples|' \
        -e 's|^RAG_RECIPES_DIR=$|RAG_RECIPES_DIR=~/.kd1-anime/knowledge/recipes|' \
        -e 's|^WORKSPACE_DIR=workspace$|WORKSPACE_DIR=~/.kd1-anime/workspace|' \
        -e 's|^SCENES_DIR=workspace/scenes$|SCENES_DIR=~/.kd1-anime/workspace/scenes|' \
        -e 's|^LOGS_DIR=workspace/logs$|LOGS_DIR=~/.kd1-anime/workspace/logs|' \
        -e 's|^VIDEOS_DIR=workspace/videos$|VIDEOS_DIR=~/.kd1-anime/workspace/videos|' \
        "$file"
}

write_empty_toml_config() {
    cat > "$CONFIG_FILE" <<'EOF'
# kd1-anime runtime configuration.
# Environment variables take precedence over this file.
EOF
    chmod 600 "$CONFIG_FILE"
}

write_toml_template() {
    if [ -f "$SCRIPT_DIR/config.toml.example" ]; then
        cp "$SCRIPT_DIR/config.toml.example" "$CONFIG_FILE"
    else
        cat > "$CONFIG_FILE" <<'EOF'
# kd1-anime 运行时配置；文件可能包含 API Key，权限应为 0600。
# 优先级：进程环境变量 > 此文件 > 当前目录 .env > ~/.kd1-anime/.env

[llm]
api_key = "sk-your-key-here"
base_url = "https://api.openai.com/v1"
model = "your-model-name"

[visual_llm]
api_key = ""
base_url = ""
model = ""

[rag]
enabled = false
index_path = "~/.kd1-anime/rag/index.sqlite3"
docs_dir = "~/.kd1-anime/knowledge/docs"
examples_dir = "~/.kd1-anime/knowledge/examples"
recipes_dir = "~/.kd1-anime/knowledge/recipes"
embedding_api_key = ""
embedding_base_url = ""
embedding_model = ""
rerank_api_key = ""
rerank_base_url = ""
rerank_model = ""

[render]
backend = "slurm"
manim_renderer = "cairo"
manim_quality = "h"

[slurm]
conda_env = "manim_env"
time_limit = "01:00:00"
cpus_per_task = 4
gpu_count = 1

[pipeline]
max_review_rounds = 8
max_fix_attempts = 8
max_plan_review_rounds = 2
max_plan_replan_attempts = 3

[evaluation]
enable_auto_eval = false
enable_visual_eval = false

[paths]
workspace_dir = "~/.kd1-anime/workspace"
output_file = "output_final.mp4"
EOF
    fi
    chmod 600 "$CONFIG_FILE"
}

migrate_installed_config() {
    local python_bin="${CONDA_ENV_DIR:-}/bin/python" local_legacy_file="$CONFIG_DIR/.env"
    [ -n "${CONDA_ENV_DIR:-}" ] && [ -x "$python_bin" ] || return 1
    CONFIG_MIGRATION_TARGET="$CONFIG_FILE" \
        CONFIG_MIGRATION_LOCAL="$local_legacy_file" \
        CONFIG_MIGRATION_LEGACY="$LEGACY_CONFIG_HOME_FILE" \
        "$python_bin" - <<'PY'
import os
from pathlib import Path

from kd1_anime.config import migrate_legacy_user_config, migrate_user_env_to_toml

target = Path(os.environ["CONFIG_MIGRATION_TARGET"])
local = Path(os.environ["CONFIG_MIGRATION_LOCAL"])
legacy = Path(os.environ["CONFIG_MIGRATION_LEGACY"])
if local.is_file():
    migrate_user_env_to_toml(local, target)
elif legacy.is_file():
    migrate_legacy_user_config(legacy.parent, target.parent)
    migrate_user_env_to_toml(target.parent / ".env", target)
PY
}

write_user_config() {
    local local_legacy_file="$CONFIG_DIR/.env"
    mkdir -p \
        "$CONFIG_DIR" \
        "$CONFIG_DIR/knowledge/docs" \
        "$CONFIG_DIR/knowledge/examples" \
        "$CONFIG_DIR/knowledge/recipes" \
        "$CONFIG_DIR/rag" \
        "$CONFIG_DIR/workspace"
    chmod 700 "$CONFIG_DIR"
    if [ -f "$CONFIG_FILE" ]; then
        chmod 600 "$CONFIG_FILE"
        warn "用户配置已存在，未覆盖: $CONFIG_FILE"
        return
    fi

    if [ -f "$local_legacy_file" ] || [ -f "$LEGACY_CONFIG_HOME_FILE" ]; then
        if migrate_installed_config && [ -f "$CONFIG_FILE" ]; then
            chmod 600 "$CONFIG_FILE"
            warn "已将旧用户 .env 迁移到: $CONFIG_FILE（旧文件未删除）"
        else
            # 没有可用的应用 Python 时不写默认值，确保旧 .env 仍能完整回退。
            write_empty_toml_config
            warn "暂时无法迁移旧 .env；保留兼容读取，首次启动时会再次尝试迁移"
        fi
        return
    fi

    write_toml_template
    write_config_value SLURM_CONDA_ENV "$ENV_NAME"
    write_config_value SLURM_CONDA_BASE "$CONDA_BASE"
    log "已创建用户配置: $CONFIG_FILE"
    warn "首次运行前请编辑该文件并填写 [llm] 的 api_key、base_url 和 model"
}

install_manim_knowledge() {
    local archive ref archive_url expected actual tmp extract source_root source relative target temporary_target
    archive="$SCRIPT_DIR/$MANIM_KNOWLEDGE_ARCHIVE"
    if [ ! -f "$archive" ]; then
        ref="${KD1_ANIME_REF:-main}"
        if [[ ! "$ref" =~ ^[A-Za-z0-9._/-]+$ ]] || [[ "$ref" == *".."* ]] || [[ "$ref" == /* ]]; then
            err "KD1_ANIME_REF 包含不安全字符: $ref"
            return 1
        fi
        tmp="$(mktemp -d)"
        cleanup_dirs+=("$tmp")
        archive="$tmp/manim-knowledge.tar.gz"
        archive_url="https://raw.githubusercontent.com/Enthusjast/kd1-anime/${ref}/${MANIM_KNOWLEDGE_ARCHIVE}"
        info "下载 Manim ${MANIM_KNOWLEDGE_VERSION} 文档和示例包"
        download "$archive_url" "$archive"
    fi

    expected="${KD1_ANIME_KNOWLEDGE_ARCHIVE_SHA256:-$MANIM_KNOWLEDGE_ARCHIVE_SHA256}"
    expected="${expected,,}"
    if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
        err "KD1_ANIME_KNOWLEDGE_ARCHIVE_SHA256 必须是 64 位十六进制 SHA-256"
        return 1
    fi
    actual="$(sha256sum "$archive" | awk '{print $1}')"
    if [ "$actual" != "$expected" ]; then
        err "Manim 知识包 SHA-256 不匹配"
        err "expected: $expected"
        err "actual:   $actual"
        return 1
    fi

    local entries
    entries="$(tar -tzf "$archive")" || {
        err "无法读取 Manim 知识包: $archive"
        return 1
    }
    while IFS= read -r relative; do
        [ -n "$relative" ] || continue
        if [[ "$relative" == /* || "$relative" == *".."* ]]; then
            err "Manim 知识包包含不安全路径: $relative"
            return 1
        fi
        case "$relative" in
            knowledge/|knowledge/docs/|knowledge/docs/manim-${MANIM_KNOWLEDGE_VERSION}/|\
            knowledge/docs/manim-${MANIM_KNOWLEDGE_VERSION}/*|knowledge/examples/|\
            knowledge/examples/manim-${MANIM_KNOWLEDGE_VERSION}/|\
            knowledge/examples/manim-${MANIM_KNOWLEDGE_VERSION}/*.py) ;;
            *)
                err "Manim 知识包包含未允许的路径: $relative"
                return 1
                ;;
        esac
    done <<< "$entries"

    extract="$(mktemp -d)"
    cleanup_dirs+=("$extract")
    tar -xzf "$archive" -C "$extract" --no-same-owner --no-same-permissions
    source_root="$extract/knowledge"
    [ -d "$source_root/docs/manim-${MANIM_KNOWLEDGE_VERSION}" ] || {
        err "Manim 知识包缺少文档目录"
        return 1
    }
    [ -d "$source_root/examples/manim-${MANIM_KNOWLEDGE_VERSION}" ] || {
        err "Manim 知识包缺少示例目录"
        return 1
    }
    if find "$source_root" -type l -print -quit | grep -q .; then
        err "Manim 知识包不允许包含符号链接"
        return 1
    fi

    mkdir -p "$CONFIG_DIR/knowledge"
    chmod 700 "$CONFIG_DIR/knowledge"
    while IFS= read -r -d '' source; do
        relative="${source#"$source_root/"}"
        case "$relative" in
            docs/manim-${MANIM_KNOWLEDGE_VERSION}/*.md|\
            docs/manim-${MANIM_KNOWLEDGE_VERSION}/*.rst|\
            examples/manim-${MANIM_KNOWLEDGE_VERSION}/*.py) ;;
            *)
                err "Manim 知识包包含未允许的文件: $relative"
                return 1
                ;;
        esac
        target="$CONFIG_DIR/knowledge/$relative"
        mkdir -p "$(dirname "$target")"
        chmod 700 "$(dirname "$target")"
        if [ -e "$target" ] || [ -L "$target" ]; then
            if [ -f "$target" ] && cmp -s "$source" "$target"; then
                continue
            fi
            warn "知识库文件已存在，保留用户版本: $target"
            continue
        fi
        # 逐文件原子安装；若安装过程被中断，下一次运行不能把残缺文件
        # 当作用户版本永久保留下来。
        temporary_target="$(mktemp "$CONFIG_DIR/.knowledge.XXXXXX")"
        cleanup_dirs+=("$temporary_target")
        cp "$source" "$temporary_target"
        chmod 600 "$temporary_target"
        # 用硬链接“仅当目标不存在时创建”，避免两个安装进程并发时用
        # -f 覆盖用户刚放入知识库的文件；临时文件与目标位于同一目录。
        if ln "$temporary_target" "$target" 2>/dev/null; then
            rm -f "$temporary_target"
        elif [ -e "$target" ] || [ -L "$target" ]; then
            warn "知识库文件在安装期间已出现，保留现有版本: $target"
            rm -f "$temporary_target"
        else
            err "无法安全安装知识库文件: $target"
            return 1
        fi
    done < <(find "$source_root" -type f -print0 | sort -z)
    log "Manim ${MANIM_KNOWLEDGE_VERSION} 文档和示例已安装到 $CONFIG_DIR/knowledge"
}

install_manim_recipes() {
    local archive ref archive_url expected actual tmp extract source_root source relative target temporary_target
    archive="$SCRIPT_DIR/$MANIM_RECIPE_ARCHIVE"
    if [ ! -f "$archive" ]; then
        ref="${KD1_ANIME_REF:-main}"
        if [[ ! "$ref" =~ ^[A-Za-z0-9._/-]+$ ]] || [[ "$ref" == *".."* ]] || [[ "$ref" == /* ]]; then
            err "KD1_ANIME_REF 包含不安全字符: $ref"
            return 1
        fi
        tmp="$(mktemp -d)"
        cleanup_dirs+=("$tmp")
        archive="$tmp/manim-recipes.tar.gz"
        archive_url="https://raw.githubusercontent.com/Enthusjast/kd1-anime/${ref}/${MANIM_RECIPE_ARCHIVE}"
        info "下载 Manim ${MANIM_KNOWLEDGE_VERSION} Recipe 包"
        download "$archive_url" "$archive"
    fi

    expected="${KD1_ANIME_RECIPE_ARCHIVE_SHA256:-$MANIM_RECIPE_ARCHIVE_SHA256}"
    expected="${expected,,}"
    if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
        err "KD1_ANIME_RECIPE_ARCHIVE_SHA256 必须是 64 位十六进制 SHA-256"
        return 1
    fi
    actual="$(sha256sum "$archive" | awk '{print $1}')"
    if [ "$actual" != "$expected" ]; then
        err "Manim Recipe 包 SHA-256 不匹配"
        err "expected: $expected"
        err "actual:   $actual"
        return 1
    fi

    local entries
    entries="$(tar -tzf "$archive")" || {
        err "无法读取 Manim Recipe 包: $archive"
        return 1
    }
    while IFS= read -r relative; do
        [ -n "$relative" ] || continue
        if [[ "$relative" == /* || "$relative" == *".."* ]]; then
            err "Manim Recipe 包包含不安全路径: $relative"
            return 1
        fi
        case "$relative" in
            knowledge/|knowledge/recipes/|knowledge/recipes/manim-${MANIM_KNOWLEDGE_VERSION}/|\
            knowledge/recipes/manim-${MANIM_KNOWLEDGE_VERSION}/*.md) ;;
            *)
                err "Manim Recipe 包包含未允许的路径: $relative"
                return 1
                ;;
        esac
    done <<< "$entries"

    extract="$(mktemp -d)"
    cleanup_dirs+=("$extract")
    tar -xzf "$archive" -C "$extract" --no-same-owner --no-same-permissions
    source_root="$extract/knowledge"
    [ -d "$source_root/recipes/manim-${MANIM_KNOWLEDGE_VERSION}" ] || {
        err "Manim Recipe 包缺少 Recipe 目录"
        return 1
    }
    if find "$source_root" -type l -print -quit | grep -q .; then
        err "Manim Recipe 包不允许包含符号链接"
        return 1
    fi

    mkdir -p "$CONFIG_DIR/knowledge/recipes/manim-${MANIM_KNOWLEDGE_VERSION}"
    chmod 700 "$CONFIG_DIR/knowledge/recipes" \
        "$CONFIG_DIR/knowledge/recipes/manim-${MANIM_KNOWLEDGE_VERSION}"
    while IFS= read -r -d '' source; do
        relative="${source#"$source_root/"}"
        case "$relative" in
            recipes/manim-${MANIM_KNOWLEDGE_VERSION}/*.md) ;;
            *)
                err "Manim Recipe 包包含未允许的文件: $relative"
                return 1
                ;;
        esac
        target="$CONFIG_DIR/knowledge/$relative"
        if [ -e "$target" ] || [ -L "$target" ]; then
            if [ -f "$target" ] && cmp -s "$source" "$target"; then
                continue
            fi
            warn "Recipe 文件已存在，保留用户版本: $target"
            continue
        fi
        temporary_target="$(mktemp "$CONFIG_DIR/.recipe.XXXXXX")"
        cleanup_dirs+=("$temporary_target")
        cp "$source" "$temporary_target"
        chmod 600 "$temporary_target"
        if ln "$temporary_target" "$target" 2>/dev/null; then
            rm -f "$temporary_target"
        elif [ -e "$target" ] || [ -L "$target" ]; then
            warn "Recipe 文件在安装期间已出现，保留现有版本: $target"
            rm -f "$temporary_target"
        else
            err "无法安全安装 Recipe 文件: $target"
            return 1
        fi
    done < <(find "$source_root" -type f -print0 | sort -z)
    log "Manim ${MANIM_KNOWLEDGE_VERSION} Recipe 已安装到 $CONFIG_DIR/knowledge/recipes"
}

toml_config_tool() {
    local operation="$1" field="${2:-}" value="${3:-}" python_bin
    if [ -n "${CONDA_ENV_DIR:-}" ] && [ -x "$CONDA_ENV_DIR/bin/python" ]; then
        python_bin="$CONDA_ENV_DIR/bin/python"
    else
        python_bin="$(command -v python3 || true)"
    fi
    if [ -z "$python_bin" ]; then
        err "找不到用于读写 TOML 配置的 Python"
        return 1
    fi
    CONFIG_PATH="$CONFIG_FILE" CONFIG_FIELD="$field" CONFIG_VALUE="$value" \
        "$python_bin" - "$operation" <<'PY'
import json
import os
import re
import sys
import tempfile
from pathlib import Path


def location(field):
    if field == "ENABLE_VISUAL_EVAL":
        return "evaluation", "enable_visual_eval"
    if field == "RAG_ENABLED":
        return "rag", "enabled"
    if field.startswith("VISUAL_LLM_"):
        return "visual_llm", field[11:].lower()
    if field.startswith("LLM_"):
        return "llm", field[4:].lower()
    if field.startswith("RAG_"):
        return "rag", field[4:].lower()
    if field.startswith("SLURM_"):
        return "slurm", field[6:].lower()
    if field == "RENDER_BACKEND":
        return "render", "backend"
    if (
        field.startswith("LOCAL_RENDER_")
        or field.startswith("LOCAL_SMOKE_RENDER_")
        or field.startswith("SMOKE_RENDER_")
        or field.startswith("MANIM_")
        or field in {"ALLOW_PARTIAL_OUTPUT", "OVERWRITE_OUTPUT"}
    ):
        return "render", field.lower()
    if field.startswith("MERGE_") or field.startswith("TRANSITION_"):
        return "merge", field.lower()
    if field in {"WORKSPACE_DIR", "OUTPUT_FILE", "SCENES_DIR", "LOGS_DIR", "VIDEOS_DIR"}:
        return "paths", field.lower()
    if field.startswith("MONITOR_") or field == "LOG_TAIL_LINES":
        return "monitor", field.lower()
    if field in {
        "ENABLE_AUTO_EVAL",
        "EVAL_THRESHOLD",
        "MAX_EVAL_ROUNDS",
        "EVAL_VISUAL_MODEL",
        "VISUAL_EVAL_FRAME_COUNT",
        "VISUAL_EVAL_THRESHOLD",
        "MAX_VISUAL_FIX_ATTEMPTS",
    }:
        return "evaluation", field.lower()
    return "pipeline", field.lower()


path = Path(os.environ["CONFIG_PATH"])
field = os.environ["CONFIG_FIELD"]
operation = sys.argv[1]
section, key = location(field)

if path.is_file():
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
else:
    lines = []

current_section = ""
found = None
section_start = None
section_end = None
for index, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        if current_section == section and section_end is None:
            section_end = index
        current_section = stripped[1:-1].strip().lower()
        if current_section == section and section_start is None:
            section_start = index
        continue
    if current_section != section:
        continue
    match = re.match(r"^(\s*)" + re.escape(key) + r"\s*=\s*(.*?)(?:\r?\n)?$", line)
    if match:
        found = index

if operation == "get":
    if found is None:
        raise SystemExit(0)
    raw = lines[found].split("=", 1)[1].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw[1:-1] if len(raw) >= 2 and raw[0] == raw[-1] == "'" else raw
    print(parsed)
    raise SystemExit(0)

if operation != "set":
    raise SystemExit(f"unsupported TOML operation: {operation}")

boolean_fields = {"ENABLE_VISUAL_EVAL", "RAG_ENABLED"}
if field in boolean_fields:
    literal = os.environ["CONFIG_VALUE"].lower()
    if literal not in {"true", "false"}:
        raise SystemExit(f"布尔配置值无效: {field}")
else:
    literal = json.dumps(os.environ["CONFIG_VALUE"], ensure_ascii=False)

replacement = f"{key} = {literal}\n"
if found is not None:
    lines[found] = replacement
elif section_start is not None:
    insert_at = section_end if section_end is not None else len(lines)
    lines.insert(insert_at, replacement)
else:
    if lines and lines[-1].strip():
        lines.append("\n")
    lines.extend([f"[{section}]\n", replacement])

path.parent.mkdir(parents=True, exist_ok=True)
path.parent.chmod(0o700)
fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
temporary = Path(temporary_name)
try:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        fd = -1
        handle.writelines(lines)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
finally:
    if fd >= 0:
        os.close(fd)
    if temporary.exists():
        temporary.unlink()
PY
}

config_value() {
    local key="$1" line value
    if [[ "$CONFIG_FILE" == *.env ]]; then
        line="$(grep -E "^${key}=" "$CONFIG_FILE" | tail -n 1 || true)"
        value="${line#*=}"
        value="${value%$'\r'}"
        if [ "${#value}" -ge 2 ] && [ "${value:0:1}" = '"' ] && [ "${value: -1}" = '"' ]; then
            value="${value:1:${#value}-2}"
        elif [ "${#value}" -ge 2 ] && [ "${value:0:1}" = "'" ] && [ "${value: -1}" = "'" ]; then
            value="${value:1:${#value}-2}"
        fi
        CONFIG_VALUE="$value"
        return
    fi
    CONFIG_VALUE="$(toml_config_tool get "$key" "")"
}

write_config_value() {
    local key="$1" value="$2" temporary line
    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || [[ "$value" == *$'\n'* ]] || [[ "$value" == *$'\r'* ]]; then
        err "配置项或配置值包含不安全字符: $key"
        return 1
    fi
    if [[ "$CONFIG_FILE" == *.env ]]; then
        temporary="$(mktemp "$CONFIG_DIR/.env.configure.XXXXXX")"
        cleanup_dirs+=("$temporary")
        if grep -qE "^${key}=" "$CONFIG_FILE"; then
            while IFS= read -r line || [ -n "$line" ]; do
                if [[ "$line" == "$key="* ]]; then
                    printf '%s=%s\n' "$key" "$value"
                else
                    printf '%s\n' "$line"
                fi
            done < "$CONFIG_FILE" > "$temporary"
        else
            cat "$CONFIG_FILE" > "$temporary"
            printf '%s=%s\n' "$key" "$value" >> "$temporary"
        fi
        chmod 600 "$temporary"
        mv -f "$temporary" "$CONFIG_FILE"
        return
    fi
    toml_config_tool set "$key" "$value"
}

prompt_yes_no() {
    local prompt="$1" default="${2:-n}" answer
    while true; do
        if ! IFS= read -r -p "$prompt" answer; then
            printf '\n'
            return 1
        fi
        answer="${answer,,}"
        if [ -z "$answer" ]; then
            answer="$default"
        fi
        case "$answer" in
            y|yes) return 0 ;;
            n|no) return 1 ;;
            *) warn "请输入 y 或 n" ;;
        esac
    done
}

prompt_url() {
    local label="$1" default="$2" value
    while true; do
        if ! IFS= read -r -p "$label Base URL [${default:-必填}]: " value; then
            printf '\n'
            return 1
        fi
        value="${value:-$default}"
        if [[ "$value" =~ ^https?://[^[:space:]]+$ ]]; then
            CONFIG_VALUE="$value"
            return 0
        fi
        warn "Base URL 必须以 http:// 或 https:// 开头，且不能包含空格"
    done
}

prompt_api_key() {
    local label="$1" default="$2" value prompt
    if [ -n "$default" ]; then
        prompt="${label} API Key（回车保留当前值，输入 - 清空）: "
    else
        prompt="${label} API Key（无鉴权可直接回车）: "
    fi
    if ! IFS= read -r -s -p "$prompt" value; then
        printf '\n'
        return 1
    fi
    printf '\n'
    if [ "$value" = "-" ]; then
        CONFIG_VALUE=""
    elif [ -z "$value" ]; then
        CONFIG_VALUE="$default"
    else
        CONFIG_VALUE="$value"
    fi
}

prompt_model() {
    local label="$1" default="$2" value
    while true; do
        if ! IFS= read -r -p "$label 模型名 [${default:-必填}]: " value; then
            printf '\n'
            return 1
        fi
        value="${value:-$default}"
        if [ -n "$value" ] && [ "$value" != "your-model-name" ]; then
            CONFIG_VALUE="$value"
            return 0
        fi
        warn "模型名不能为空"
    done
}

configure_model_profile() {
    local prefix="$1" label="$2" api_required="${3:-false}" default_url default_key default_model
    config_value "${prefix}_BASE_URL"
    default_url="$CONFIG_VALUE"
    config_value "${prefix}_API_KEY"
    default_key="$CONFIG_VALUE"
    config_value "${prefix}_MODEL"
    default_model="$CONFIG_VALUE"
    case "$default_key" in
        sk-your-key-here|your-api-key) default_key="" ;;
    esac

    prompt_url "$label" "$default_url" || return 1
    write_config_value "${prefix}_BASE_URL" "$CONFIG_VALUE" || return 1
    prompt_api_key "$label" "$default_key" || return 1
    if [ "$api_required" = true ] && [ -z "$CONFIG_VALUE" ]; then
        warn "$label API Key 不能为空"
        return 1
    fi
    write_config_value "${prefix}_API_KEY" "$CONFIG_VALUE" || return 1
    prompt_model "$label" "$default_model" || return 1
    write_config_value "${prefix}_MODEL" "$CONFIG_VALUE" || return 1
}

build_rag_index_from_installer() {
    if [ -z "${CONDA_ENV_DIR:-}" ] || [ ! -x "$CONDA_ENV_DIR/bin/kd1-anime" ]; then
        warn "找不到已安装的 kd1-anime 命令，稍后请手动执行: kd1-anime rag index"
        return 0
    fi
    info "正在构建 Manim 0.20.1 RAG 索引（可能产生 Embedding 请求）"
    if "$CONDA_ENV_DIR/bin/kd1-anime" rag index; then
        log "RAG 索引构建完成"
    else
        warn "RAG 索引构建失败；安装仍继续，稍后可执行: kd1-anime rag index"
    fi
}

configure_user_models() {
    if [ "$CONFIGURE_MODE" = never ] || { [ "$CONFIGURE_MODE" = auto ] && { [ ! -t 0 ] || [ ! -t 1 ]; }; }; then
        info "非交互安装，跳过模型配置；需要引导配置时请在终端运行: KD1_ANIME_CONFIGURE_MODE=interactive bash install.sh"
        return 0
    fi

    printf '\n'
    printf '%b模型配置向导%b\n' "$CYAN" "$NC"
    printf '%s\n' "按顺序配置 Base URL、API Key 和模型名；API Key 输入时不会回显。"
    printf '%s\n\n' "可选服务选择“否”时会保留已有凭据，但不会启用该功能。"

    info "配置主模型"
    configure_model_profile "LLM" "主模型" true || return 1

    config_value ENABLE_VISUAL_EVAL
    local visual_default="n"
    case "${CONFIG_VALUE,,}" in
        true|1|yes|y|on) visual_default="y" ;;
    esac
    if prompt_yes_no "是否启用视觉模型/视觉评估？[y/N] " "$visual_default"; then
        write_config_value ENABLE_VISUAL_EVAL true || return 1
        info "配置视觉模型"
        configure_model_profile "VISUAL_LLM" "视觉模型" true || return 1
    else
        write_config_value ENABLE_VISUAL_EVAL false || return 1
        log "已跳过视觉模型"
    fi

    config_value RAG_ENABLED
    local rag_default="n"
    case "${CONFIG_VALUE,,}" in
        true|1|yes|y|on) rag_default="y" ;;
    esac
    local embedding_enabled=0 reranker_enabled=0
    if prompt_yes_no "是否启用 Embedding 模型？[y/N] " "$rag_default"; then
        embedding_enabled=1
        info "配置 Embedding 模型"
        configure_model_profile "RAG_EMBEDDING" "Embedding 模型" false || return 1
        if prompt_yes_no "是否现在构建 RAG 索引？[y/N] " "n"; then
            build_rag_index_from_installer
        else
            info "已跳过索引构建；稍后可执行: kd1-anime rag index"
        fi
    else
        log "已跳过 Embedding 模型"
    fi

    if prompt_yes_no "是否启用 Reranker 模型？[y/N] " "$rag_default"; then
        reranker_enabled=1
        info "配置 Reranker 模型"
        configure_model_profile "RAG_RERANK" "Reranker 模型" false || return 1
    else
        log "已跳过 Reranker 模型"
    fi

    if [ "$embedding_enabled" -eq 1 ] && [ "$reranker_enabled" -eq 1 ]; then
        write_config_value RAG_ENABLED true || return 1
        log "RAG 已启用"
    else
        write_config_value RAG_ENABLED false || return 1
        warn "Embedding 和 Reranker 未同时启用，RAG 已保持关闭"
    fi
    log "模型配置已保存: $CONFIG_FILE"
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
install_manim_knowledge
install_manim_recipes
configure_user_models

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
