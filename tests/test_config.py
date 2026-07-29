import pytest

from kd1_anime.config import Settings


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
