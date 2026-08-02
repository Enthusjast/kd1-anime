import pytest

import kd1_anime.config as config_module
from kd1_anime.config import Settings
from kd1_anime.orchestrator import RunPaths


def test_llm_config_rejects_placeholder_model():
    config = Settings(
        _env_file=None,
        LLM_API_KEY="valid-key",
        LLM_BASE_URL="https://example.invalid/v1",
        LLM_MODEL="your-model-name",
    )

    with pytest.raises(ValueError, match="LLM_MODEL"):
        config.require_llm_key()


def test_llm_config_accepts_generic_openai_compatible_provider():
    config = Settings(
        _env_file=None,
        LLM_API_KEY="valid-key",
        LLM_BASE_URL="https://example.invalid/v1",
        LLM_MODEL="provider-model",
    )

    config.require_llm_key()


def test_slurm_identifier_rejects_newline_injection():
    with pytest.raises(ValueError, match="SLURM_PARTITION"):
        Settings(
            _env_file=None,
            SLURM_PARTITION="normal\n#SBATCH --mail-user=attacker@example.com",
        )


def test_settings_assignment_is_validated():
    config = Settings(_env_file=None)
    with pytest.raises(ValueError):
        config.SLURM_QOS = "normal\n#SBATCH --exclusive"


def test_relative_runtime_paths_fall_back_to_home_when_cwd_is_missing(monkeypatch):
    def missing_cwd():
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(config_module, "_current_working_directory", missing_cwd)
    monkeypatch.setattr(config_module.settings, "WORKSPACE_DIR", config_module.Path("workspace"))
    monkeypatch.setattr(
        config_module.settings,
        "OUTPUT_FILE",
        config_module.Path("output_final.mp4"),
    )

    paths = RunPaths.create()

    expected_runs = (config_module.Path.home() / "workspace" / "runs").resolve()
    assert paths.root.parent == expected_runs
    assert paths.output == paths.root / "output_final.mp4"
