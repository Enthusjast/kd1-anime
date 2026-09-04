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


def test_llm_base_url_is_normalized_and_rejects_unsafe_or_invalid_urls():
    config = Settings(_env_file=None, LLM_BASE_URL="https://example.invalid/v1/")
    assert config.LLM_BASE_URL == "https://example.invalid/v1"

    for value in (
        "example.invalid/v1",
        "ftp://example.invalid/v1",
        "https://example.invalid/v1\nX-Injected: yes",
    ):
        with pytest.raises(ValueError, match="LLM_BASE_URL"):
            Settings(_env_file=None, LLM_BASE_URL=value)


def test_slurm_identifier_rejects_newline_injection():
    with pytest.raises(ValueError, match="SLURM_PARTITION"):
        Settings(
            _env_file=None,
            SLURM_PARTITION="normal\n#SBATCH --mail-user=attacker@example.com",
        )


def test_slurm_time_limit_rejects_invalid_minute_or_second():
    with pytest.raises(ValueError, match="MM 和 SS"):
        Settings(_env_file=None, SLURM_TIME_LIMIT="01:60:00")
    with pytest.raises(ValueError, match="MM 和 SS"):
        Settings(_env_file=None, SLURM_TIME_LIMIT="01:00:60")


def test_empty_container_image_is_normalized_to_none():
    """.env 中 SLURM_CONTAINER_IMAGE= 不应被解析成 truthy 的 Path(".")。"""
    config = Settings(
        _env_file=None,
        SLURM_CONTAINER_IMAGE="",
        SLURM_REQUIRE_CONTAINER=False,
    )
    assert config.SLURM_CONTAINER_IMAGE is None

    config2 = Settings(
        _env_file=None,
        SLURM_CONTAINER_IMAGE="   ",
        SLURM_REQUIRE_CONTAINER=False,
    )
    assert config2.SLURM_CONTAINER_IMAGE is None


def test_empty_conda_base_is_normalized_to_none():
    config = Settings(_env_file=None, SLURM_CONDA_BASE="")
    assert config.SLURM_CONDA_BASE is None

    config2 = Settings(_env_file=None, SLURM_CONDA_BASE="   ")
    assert config2.SLURM_CONDA_BASE is None


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


def test_llm_timeout_and_silent_stream_defaults():
    config = Settings(_env_file=None)
    assert config.LLM_TIMEOUT_CONNECT == 30.0
    assert config.LLM_TIMEOUT_READ == 600.0
    assert config.LLM_SILENT_STREAM is True
    assert config.LLM_HEALTHCHECK_TIMEOUT == 15.0
    assert config.LLM_MAX_TOKENS == 32768
    assert config.LLM_EMPTY_RETRY_MAX_TOKENS == 16384


def test_llm_timeout_and_silent_stream_validation():
    with pytest.raises(ValueError):
        Settings(_env_file=None, LLM_TIMEOUT_READ=5.0)  # 低于下限 10s
    with pytest.raises(ValueError):
        Settings(_env_file=None, LLM_HEALTHCHECK_TIMEOUT=0.5)
    with pytest.raises(ValueError):
        Settings(_env_file=None, LLM_EMPTY_RETRY_MAX_TOKENS=100)  # 低于下限


def test_max_fix_attempts_default_and_upper_bound():
    config = Settings(_env_file=None)
    assert config.MAX_FIX_ATTEMPTS == 5
    # 超过上限 le=20 会被拒绝
    with pytest.raises(ValueError):
        Settings(_env_file=None, MAX_FIX_ATTEMPTS=21)
    # 赋值校验同样生效
    with pytest.raises(ValueError):
        config.MAX_FIX_ATTEMPTS = 21


def test_max_fix_identical_errors_default():
    config = Settings(_env_file=None)
    assert config.MAX_FIX_IDENTICAL_ERRORS == 3
    with pytest.raises(ValueError):
        Settings(_env_file=None, MAX_FIX_IDENTICAL_ERRORS=0)


def test_visual_llm_profile_is_independent_from_main_endpoint():
    config = Settings(
        _env_file=None,
        LLM_API_KEY="main-key",
        LLM_BASE_URL="https://main.invalid/v1",
        LLM_MODEL="main-model",
    )

    profile = config.visual_llm_profile()

    assert profile.api_key == ""
    assert profile.base_url == ""
    assert profile.model == ""
    with pytest.raises(ValueError, match="VISUAL_LLM_API_KEY"):
        config.require_visual_llm()


def test_visual_llm_profile_uses_only_visual_endpoint_and_model_override():
    config = Settings(
        _env_file=None,
        LLM_API_KEY="main-key",
        LLM_BASE_URL="https://main.invalid/v1",
        LLM_MODEL="main-model",
        VISUAL_LLM_API_KEY="visual-key",
        VISUAL_LLM_BASE_URL="https://visual.invalid/v1/",
        VISUAL_LLM_MODEL="visual-default",
    )

    profile = config.visual_llm_profile(model_override="visual-override")

    profile.require()
    assert profile.api_key == "visual-key"
    assert profile.base_url == "https://visual.invalid/v1"
    assert profile.model == "visual-override"
    assert profile.debug is False
