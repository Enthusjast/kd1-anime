"""安装脚本的用户级默认配置测试。"""

import os
import shlex
import subprocess
from pathlib import Path

INSTALL_SCRIPT = Path(__file__).parents[1] / "install.sh"


def run_install_function(home: Path, function: str, *, backend: str = "local"):
    command = f"source {shlex.quote(str(INSTALL_SCRIPT))}; {function}"
    environment = os.environ.copy()
    environment["HOME"] = str(home)
    environment["KD1_ANIME_RENDER_BACKEND"] = backend
    return subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_minimal_config_defaults_to_local_backend(tmp_path: Path):
    result = run_install_function(
        tmp_path,
        'write_minimal_toml_config; cat "$CONFIG_FILE"',
    )

    assert result.returncode == 0, result.stderr
    assert 'backend = "local"' in result.stdout
    assert "[slurm]" not in result.stdout
    config = tmp_path / ".kd1-anime" / "config.toml"
    assert config.stat().st_mode & 0o777 == 0o600


def test_minimal_config_can_opt_into_slurm(tmp_path: Path):
    result = run_install_function(
        tmp_path,
        'write_minimal_toml_config; grep \'^backend = "slurm"$\' "$CONFIG_FILE"',
        backend="slurm",
    )

    assert result.returncode == 0, result.stderr
    assert 'backend = "slurm"' in result.stdout


def test_check_host_accepts_linux_with_basic_tools(tmp_path: Path):
    result = run_install_function(tmp_path, "check_host")

    assert result.returncode == 0, result.stderr
