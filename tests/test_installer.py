import hashlib
import os
import shlex
import subprocess
from pathlib import Path

INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"
ARCHIVE_CONTENT = b"archive-content"
ARCHIVE_SHA256 = hashlib.sha256(ARCHIVE_CONTENT).hexdigest()


def make_fake_tex_bin(path: Path, *, complete: bool) -> Path:
    path.mkdir(parents=True)
    kpsewhich = path / "kpsewhich"
    kpsewhich.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = -var-value=TEXMFROOT ]; then\n'
        "    printf '%s\\n' \"${FAKE_TEX_ROOT:-/nonexistent}\"\n"
        'elif [ "${FAKE_TEX_COMPLETE:-0}" = 1 ]; then\n'
        "    printf '/fake/texmf/%s\\n' \"${1:-unknown}\"\n"
        "fi\n",
        encoding="utf-8",
    )
    kpsewhich.chmod(0o700)
    if complete:
        for command in ("xelatex", "dvisvgm"):
            executable = path / command
            executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            executable.chmod(0o700)
    return path


def run_install_function(tmp_path: Path, *, ref: str, checksum: str) -> subprocess.CompletedProcess:
    installer = shlex.quote(str(INSTALLER))
    fake_source = shlex.quote(str(tmp_path / "no-source-tree"))
    script = f"""
source {installer}
SCRIPT_DIR={fake_source}
download() {{ printf 'archive-content' > "$2"; }}
env_pip() {{ printf 'PIP:%s\\n' "$*"; }}
KD1_ANIME_REF={shlex.quote(ref)}
KD1_ANIME_ARCHIVE_SHA256={shlex.quote(checksum)}
install_python_package
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_installer_can_be_sourced_without_running_main(tmp_path):
    result = subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(INSTALLER))}; echo sourced"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "sourced"


def test_installer_pins_documented_manim_version(tmp_path):
    script = f"source {shlex.quote(str(INSTALLER))}; printf '%s\\n' \"$MANIM_SPEC\""

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "manim==0.20.1"


def test_remote_archive_checksum_is_verified_before_pip(tmp_path):
    result = run_install_function(tmp_path, ref="v0.3.0", checksum=ARCHIVE_SHA256)

    assert result.returncode == 0, result.stderr
    assert "SHA-256 验证通过" in result.stdout
    assert "PIP:install --upgrade" in result.stdout


def test_remote_archive_checksum_mismatch_stops_before_pip(tmp_path):
    result = run_install_function(tmp_path, ref="v0.3.0", checksum="0" * 64)

    assert result.returncode != 0
    assert "SHA-256 不匹配" in result.stderr
    assert "PIP:" not in result.stdout


def test_unsafe_remote_ref_stops_before_download_or_pip(tmp_path):
    result = run_install_function(
        tmp_path,
        ref="../../main",
        checksum=ARCHIVE_SHA256,
    )

    assert result.returncode != 0
    assert "不安全字符" in result.stderr


def test_tlmgr_falls_back_to_release_matched_repository(tmp_path):
    tex_bin = tmp_path / "tex-bin"
    tex_bin.mkdir()
    state_file = tmp_path / "repository"
    calls_file = tmp_path / "calls"
    fake_tlmgr = tex_bin / "tlmgr"
    fake_tlmgr.write_text(
        r"""#!/usr/bin/env bash
set -eu
case "${1:-}" in
    --version)
        printf '%s\n' 'tlmgr revision 1' 'TeX Live (https://tug.org/texlive) version 2024'
        ;;
    option)
        printf '%s\n' "$3" > "$FAKE_TLMGR_STATE"
        ;;
    info)
        printf '%s\n' "$*" >> "$FAKE_TLMGR_CALLS"
        repo="$(cat "$FAKE_TLMGR_STATE")"
        [[ "$*" == *"--only-remote"* ]]
        [[ "$repo" == */2024/tlnet-final ]]
        ;;
    *)
        exit 2
        ;;
esac
""",
        encoding="utf-8",
    )
    fake_tlmgr.chmod(0o700)

    env = os.environ.copy()
    env.update(
        {
            "FAKE_TLMGR_STATE": str(state_file),
            "FAKE_TLMGR_CALLS": str(calls_file),
        }
    )
    script = f"""
source {shlex.quote(str(INSTALLER))}
TEX_BIN={shlex.quote(str(tex_bin))}
configure_tlmgr_repo
"""
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=env,
    )

    expected = "https://ftp.tug.org/historic/systems/texlive/2024/tlnet-final"
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected
    assert calls_file.read_text(encoding="utf-8").splitlines() == [
        "info --only-remote texlive.infra",
        "info --only-remote texlive.infra",
    ]


def test_existing_path_xelatex_is_preferred_and_skips_install(tmp_path):
    tex_bin = make_fake_tex_bin(tmp_path / "path-tex" / "bin", complete=True)
    calls = tmp_path / "calls"
    script = f"""
source {shlex.quote(str(INSTALLER))}
tex_candidate_bins() {{ printf '%s\\n' {shlex.quote(str(tex_bin))}; }}
install_texlive() {{ printf 'install\\n' >> {shlex.quote(str(calls))}; return 1; }}
install_tex_dependencies() {{ printf 'tlmgr\\n' >> {shlex.quote(str(calls))}; return 1; }}
ensure_texlive
printf 'TEX_BIN=%s\\n' "$TEX_BIN"
"""
    env = os.environ.copy()
    env["FAKE_TEX_COMPLETE"] = "1"
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert f"TEX_BIN={tex_bin}" in result.stdout
    assert "使用现有 TeX Live" in result.stdout
    assert not calls.exists()


def test_path_candidate_precedes_common_texlive_roots(tmp_path):
    tex_bin = make_fake_tex_bin(tmp_path / "path-tex" / "bin", complete=True)
    env = os.environ.copy()
    env["PATH"] = f"{tex_bin}:/usr/bin:/bin"
    env["FAKE_TEX_COMPLETE"] = "1"
    script = f"""
source {shlex.quote(str(INSTALLER))}
mapfile -t candidates < <(tex_candidate_bins)
printf '%s\\n' "${{candidates[0]}}"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tex_bin)


def test_unusable_system_tex_falls_back_to_private_install(tmp_path):
    incomplete = make_fake_tex_bin(tmp_path / "system-tex" / "bin", complete=False)
    private = make_fake_tex_bin(tmp_path / "private-tex" / "bin", complete=True)
    calls = tmp_path / "calls"
    script = f"""
source {shlex.quote(str(INSTALLER))}
tex_candidate_bins() {{ printf '%s\\n' {shlex.quote(str(incomplete))}; }}
install_texlive() {{ printf 'install\\n' >> {shlex.quote(str(calls))}; TEX_BIN={shlex.quote(str(private))}; }}
install_tex_dependencies() {{ printf 'dependencies\\n' >> {shlex.quote(str(calls))}; return 0; }}
ensure_texlive
printf 'TEX_BIN=%s\\n' "$TEX_BIN"
"""
    env = os.environ.copy()
    env["FAKE_TEX_COMPLETE"] = "1"
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == ["install", "dependencies"]
    assert f"TEX_BIN={private}" in result.stdout


def test_minimal_tex_packages_include_xelatex_and_cjk_support(tmp_path):
    tex_bin = make_fake_tex_bin(tmp_path / "incomplete-tex" / "bin", complete=False)
    script = f"""
source {shlex.quote(str(INSTALLER))}
missing_tex_packages {shlex.quote(str(tex_bin))}
"""
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    packages = set(result.stdout.splitlines())
    assert {"xetex", "dvisvgm", "ctex", "xecjk", "fontspec"} <= packages
    assert "collection-latexextra" not in packages
    assert "scheme-full" not in packages


def test_completion_message_renders_ansi_escape_sequences(tmp_path):
    config_file = tmp_path / "config" / ".env"
    user_bin = tmp_path / "bin"
    script = f"""
source {shlex.quote(str(INSTALLER))}
CONFIG_FILE={shlex.quote(str(config_file))}
USER_BIN_DIR={shlex.quote(str(user_bin))}
print_completion
"""
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "\x1b[0;32m安装完成\x1b[0m" in result.stdout
    assert r"\033" not in result.stdout
    assert f"3. 编辑配置: {config_file}" in result.stdout
    assert f"命令目录: {user_bin}" in result.stdout


def test_installer_uses_private_application_home_for_user_storage(tmp_path):
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    result = subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(INSTALLER))}; write_user_config"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    config_dir = tmp_path / ".kd1-anime"
    config_file = config_dir / ".env"
    assert config_file.is_file()
    assert config_file.stat().st_mode & 0o777 == 0o600
    content = config_file.read_text(encoding="utf-8")
    assert "RAG_INDEX_PATH=~/.kd1-anime/rag/index.sqlite3" in content
    assert "RAG_DOCS_DIR=~/.kd1-anime/knowledge/docs" in content
    assert "RAG_EXAMPLES_DIR=~/.kd1-anime/knowledge/examples" in content
    assert "WORKSPACE_DIR=~/.kd1-anime/workspace" in content
    assert "MAX_PLAN_REVIEW_ROUNDS=2" in content
    assert "SAFE_FALLBACK_ENABLED=true" in content
    assert "MAX_IDENTICAL_REVIEW_ATTEMPTS=2" in content
    assert (config_dir / "knowledge" / "docs").is_dir()
    assert (config_dir / "knowledge" / "examples").is_dir()
    assert not (tmp_path / ".config" / "kd1-anime").exists()


def test_installer_migrates_legacy_user_config_without_overwriting_it(tmp_path):
    legacy_dir = tmp_path / ".config" / "kd1-anime"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / ".env"
    legacy_file.write_text(
        "RAG_INDEX_PATH=~/.cache/kd1-anime/rag/index.sqlite3\n"
        "RAG_DOCS_DIR=\n"
        "RAG_EXAMPLES_DIR=\n"
        "WORKSPACE_DIR=workspace\n"
        "LLM_MODEL=legacy-model\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    result = subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(INSTALLER))}; write_user_config"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    migrated = tmp_path / ".kd1-anime" / ".env"
    assert migrated.read_text(encoding="utf-8") == (
        "RAG_INDEX_PATH=~/.kd1-anime/rag/index.sqlite3\n"
        "RAG_DOCS_DIR=~/.kd1-anime/knowledge/docs\n"
        "RAG_EXAMPLES_DIR=~/.kd1-anime/knowledge/examples\n"
        "WORKSPACE_DIR=~/.kd1-anime/workspace\n"
        "LLM_MODEL=legacy-model\n"
    )
    assert legacy_file.read_text(encoding="utf-8") == (
        "RAG_INDEX_PATH=~/.cache/kd1-anime/rag/index.sqlite3\n"
        "RAG_DOCS_DIR=\n"
        "RAG_EXAMPLES_DIR=\n"
        "WORKSPACE_DIR=workspace\n"
        "LLM_MODEL=legacy-model\n"
    )


def test_installer_extracts_bundled_manim_knowledge(tmp_path):
    script = f"""
source {shlex.quote(str(INSTALLER))}
CONFIG_DIR={shlex.quote(str(tmp_path / ".kd1-anime"))}
SCRIPT_DIR={shlex.quote(str(INSTALLER.parent))}
install_manim_knowledge
"""
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    docs = tmp_path / ".kd1-anime" / "knowledge" / "docs" / "manim-0.20.1"
    examples = tmp_path / ".kd1-anime" / "knowledge" / "examples" / "manim-0.20.1"
    assert (docs / "SOURCE.md").is_file()
    assert (docs / "guides" / "configuration.rst").is_file()
    assert (examples / "basic.py").is_file()
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in docs.rglob("*") if path.is_file())
    assert all(
        path.stat().st_mode & 0o777 == 0o600 for path in examples.rglob("*") if path.is_file()
    )
    assert all(path.stat().st_mode & 0o777 == 0o700 for path in docs.rglob("*") if path.is_dir())


def test_interactive_model_configuration_wizard_writes_all_profiles(tmp_path):
    marker = tmp_path / "index-built"
    script = f"""
source {shlex.quote(str(INSTALLER))}
CONFIG_DIR={shlex.quote(str(tmp_path / ".kd1-anime"))}
CONFIG_FILE="$CONFIG_DIR/.env"
CONFIGURE_MODE=interactive
write_user_config
build_rag_index_from_installer() {{ printf 'built\\n' > {shlex.quote(str(marker))}; }}
configure_user_models
"""
    answers = (
        "\n".join(
            [
                "https://main.example/v1",
                "main-secret",
                "main-model",
                "y",
                "https://visual.example/v1",
                "visual-secret",
                "visual-model",
                "y",
                "https://embedding.example/v1",
                "embedding-secret",
                "embedding-model",
                "y",
                "y",
                "https://rerank.example/v1",
                "rerank-secret",
                "rerank-model",
            ]
        )
        + "\n"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        input=answers,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "main-secret" not in result.stdout + result.stderr
    assert "visual-secret" not in result.stdout + result.stderr
    assert "embedding-secret" not in result.stdout + result.stderr
    assert "rerank-secret" not in result.stdout + result.stderr
    content = (tmp_path / ".kd1-anime" / ".env").read_text(encoding="utf-8")
    assert "LLM_BASE_URL=https://main.example/v1" in content
    assert "LLM_API_KEY=main-secret" in content
    assert "LLM_MODEL=main-model" in content
    assert "ENABLE_VISUAL_EVAL=true" in content
    assert "VISUAL_LLM_BASE_URL=https://visual.example/v1" in content
    assert "VISUAL_LLM_API_KEY=visual-secret" in content
    assert "VISUAL_LLM_MODEL=visual-model" in content
    assert "RAG_ENABLED=true" in content
    assert "RAG_EMBEDDING_BASE_URL=https://embedding.example/v1" in content
    assert "RAG_EMBEDDING_API_KEY=embedding-secret" in content
    assert "RAG_EMBEDDING_MODEL=embedding-model" in content
    assert "RAG_RERANK_BASE_URL=https://rerank.example/v1" in content
    assert "RAG_RERANK_API_KEY=rerank-secret" in content
    assert "RAG_RERANK_MODEL=rerank-model" in content
    assert marker.read_text(encoding="utf-8") == "built\n"


def test_model_configuration_wizard_skips_non_tty_by_default(tmp_path):
    script = f"""
source {shlex.quote(str(INSTALLER))}
CONFIG_DIR={shlex.quote(str(tmp_path / ".kd1-anime"))}
CONFIG_FILE="$CONFIG_DIR/.env"
write_user_config
configure_user_models
"""
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        input="",
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "非交互安装，跳过模型配置" in result.stdout
    assert "模型配置向导" not in result.stdout


def test_installer_creates_runnable_wrappers_and_idempotent_shell_config(tmp_path):
    conda_base = tmp_path / "conda"
    conda_sh = conda_base / "etc" / "profile.d" / "conda.sh"
    conda_sh.parent.mkdir(parents=True)
    conda_sh.write_text(
        "conda() {\n"
        '    [ "${1:-}" = activate ] || return 1\n'
        '    export CONDA_DEFAULT_ENV="$2"\n'
        "}\n",
        encoding="utf-8",
    )
    conda_env = conda_base / "envs" / "manim_env"
    cli_entry = conda_env / "bin" / "kd1-anime"
    cli_entry.parent.mkdir(parents=True)
    cli_entry.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'CLI:%s\\n' \"$*\"\n"
        "printf 'PYTHONHOME:%s\\n' \"${PYTHONHOME-unset}\"\n"
        "printf 'PATH:%s\\n' \"$PATH\"\n",
        encoding="utf-8",
    )
    cli_entry.chmod(0o700)
    tex_bin = tmp_path / "texlive" / "bin"
    tex_bin.mkdir(parents=True)
    user_bin = tmp_path / "user-bin"
    rc_file = tmp_path / ".zshrc"
    script = f"""
source {shlex.quote(str(INSTALLER))}
USER_BIN_DIR={shlex.quote(str(user_bin))}
CONDA_BASE={shlex.quote(str(conda_base))}
CONDA_ENV_DIR={shlex.quote(str(conda_env))}
TEX_BIN={shlex.quote(str(tex_bin))}
ENV_NAME=manim_env
install_command_wrappers
configure_shell_rc {shlex.quote(str(rc_file))}
configure_shell_rc {shlex.quote(str(rc_file))}
"""
    install_result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert install_result.returncode == 0, install_result.stderr
    cli_wrapper = user_bin / "kd1-anime"
    env_wrapper = user_bin / "manim-env"
    assert cli_wrapper.stat().st_mode & 0o777 == 0o700
    assert env_wrapper.stat().st_mode & 0o777 == 0o700

    clean_env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONHOME": "broken",
        "SHELL": "/bin/bash",
    }
    cli_result = subprocess.run(
        [str(cli_wrapper), "version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=clean_env,
    )
    assert cli_result.returncode == 0, cli_result.stderr
    assert "CLI:version" in cli_result.stdout
    assert "PYTHONHOME:unset" in cli_result.stdout
    assert f"PATH:{tex_bin}:{conda_env}/bin:" in cli_result.stdout

    env_result = subprocess.run(
        [
            str(env_wrapper),
            "/bin/bash",
            "-c",
            'printf "ENV:%s\\nPYTHONHOME:%s\\n" "$CONDA_DEFAULT_ENV" "${PYTHONHOME-unset}"',
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=clean_env,
    )
    assert env_result.returncode == 0, env_result.stderr
    assert env_result.stdout == "ENV:manim_env\nPYTHONHOME:unset\n"

    rc_content = rc_file.read_text(encoding="utf-8")
    assert rc_content.count("# >>> kd1-anime install.sh >>>") == 1
    assert "manim-env()" in rc_content
    shell_result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            f"source {shlex.quote(str(rc_file))}; manim-env; "
            'printf "ENV:%s\\n" "$CONDA_DEFAULT_ENV"; command -v kd1-anime',
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=clean_env,
    )
    assert shell_result.returncode == 0, shell_result.stderr
    assert "ENV:manim_env" in shell_result.stdout
    assert str(cli_wrapper) in shell_result.stdout
