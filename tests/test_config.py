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


@pytest.mark.parametrize(
    "field",
    ["LLM_BASE_URL", "VISUAL_LLM_BASE_URL", "RAG_EMBEDDING_BASE_URL"],
)
def test_service_urls_reject_embedded_credentials_and_fragments(field):
    with pytest.raises(ValueError, match=r"凭据|fragment"):
        Settings(_env_file=None, **{field: "https://user:secret@example.invalid/v1#frag"})


def test_default_storage_is_under_private_application_home():
    config = Settings(_env_file=None)

    assert config_module.Path.home() / ".kd1-anime" == config_module.APP_HOME
    assert config_module.USER_CONFIG_DIR == config_module.APP_HOME
    assert config_module.USER_ENV_FILE == config_module.APP_HOME / ".env"
    assert config.RAG_INDEX_PATH == config_module.DEFAULT_RAG_INDEX_PATH
    assert config.RAG_DOCS_DIR == config_module.DEFAULT_RAG_DOCS_DIR
    assert config.RAG_EXAMPLES_DIR == config_module.DEFAULT_RAG_EXAMPLES_DIR
    assert config.WORKSPACE_DIR == config_module.DEFAULT_WORKSPACE_DIR
    assert config.SCENES_DIR == config_module.DEFAULT_SCENES_DIR
    assert config.LOGS_DIR == config_module.DEFAULT_LOGS_DIR
    assert config.VIDEOS_DIR == config_module.DEFAULT_VIDEOS_DIR


def test_legacy_user_config_is_migrated_without_overwriting_custom_paths(tmp_path):
    legacy_dir = tmp_path / "legacy"
    target_dir = tmp_path / "new"
    legacy_dir.mkdir()
    (legacy_dir / ".env").write_text(
        "RAG_INDEX_PATH=~/.cache/kd1-anime/rag/index.sqlite3\n"
        "RAG_DOCS_DIR=\n"
        "RAG_EXAMPLES_DIR=\n"
        "WORKSPACE_DIR=workspace\n"
        "SCENES_DIR=/srv/custom-scenes\n"
        "LLM_MODEL=keep-this-model\n",
        encoding="utf-8",
    )
    (legacy_dir / ".env.example").write_text(
        "WORKSPACE_DIR=workspace\n",
        encoding="utf-8",
    )

    migrated = config_module.migrate_legacy_user_config(legacy_dir, target_dir)

    assert set(migrated) == {target_dir / ".env", target_dir / ".env.example"}
    assert (target_dir / ".env").read_text(encoding="utf-8") == (
        f"RAG_INDEX_PATH={config_module.DEFAULT_RAG_INDEX_PATH}\n"
        f"RAG_DOCS_DIR={config_module.DEFAULT_RAG_DOCS_DIR}\n"
        f"RAG_EXAMPLES_DIR={config_module.DEFAULT_RAG_EXAMPLES_DIR}\n"
        f"WORKSPACE_DIR={config_module.DEFAULT_WORKSPACE_DIR}\n"
        "SCENES_DIR=/srv/custom-scenes\n"
        "LLM_MODEL=keep-this-model\n"
    )
    assert (target_dir / ".env").stat().st_mode & 0o777 == 0o600
    assert (target_dir / ".env.example").read_text(encoding="utf-8") == (
        f"WORKSPACE_DIR={config_module.DEFAULT_WORKSPACE_DIR}\n"
    )

    # 迁移是幂等且非覆盖的。
    (target_dir / ".env").write_text("LLM_MODEL=edited\n", encoding="utf-8")
    assert config_module.migrate_legacy_user_config(legacy_dir, target_dir) == ()
    assert (target_dir / ".env").read_text(encoding="utf-8") == "LLM_MODEL=edited\n"


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
    assert config.LLM_PLANNING_TEMPERATURE == 0.2
    assert config.LLM_TECHNICAL_TEMPERATURE == 0.0
    assert config.LLM_CODE_TEMPERATURE == 0.2
    assert config.LLM_REVIEW_TEMPERATURE == 0.0
    assert config.LLM_FIX_TEMPERATURE == 0.1
    assert config.LLM_PLANNING_MAX_TOKENS == 16384
    assert config.LLM_TECHNICAL_MAX_TOKENS == 16384
    assert config.LLM_CODE_MAX_TOKENS == 24576
    assert config.LLM_REVIEW_MAX_TOKENS == 8192
    assert config.LLM_EMPTY_RETRY_MAX_TOKENS == 16384
    assert config.LLM_CACHE_ENABLED is True
    assert config.LLM_CACHE_MAX_ENTRIES == 512
    assert config.LLM_MAX_CONTEXT_CHARS == 120_000
    assert config.LLM_MAX_CODE_CONTEXT_CHARS == 60_000
    assert config.LLM_MAX_REVIEW_CONTEXT_CHARS == 90_000
    assert config.LLM_MAX_TECHNICAL_SPEC_CHARS == 30_000
    assert config.MAX_TECHNICAL_SPEC_ATTEMPTS == 3
    assert config.MAX_PLAN_REPLAN_ATTEMPTS == 3
    assert config.SMOKE_RENDER_ENABLED is True
    assert config.SMOKE_RENDER_QUALITY == "l"
    assert config.LOCAL_SMOKE_RENDER_ENABLED is False


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


def test_rag_settings_normalize_empty_source_dirs_and_validate_urls(tmp_path):
    config = Settings(
        _env_file=None,
        RAG_INDEX_PATH=tmp_path / "index.sqlite3",
        RAG_DOCS_DIR="",
        RAG_EXAMPLES_DIR="   ",
        RAG_EMBEDDING_BASE_URL="https://embedding.invalid/v1/",
        RAG_RERANK_BASE_URL="https://rerank.invalid/v1/",
    )

    assert config.RAG_DOCS_DIR is None
    assert config.RAG_EXAMPLES_DIR is None
    assert config.RAG_EMBEDDING_BASE_URL == "https://embedding.invalid/v1"
    assert config.RAG_RERANK_BASE_URL == "https://rerank.invalid/v1"

    default_path = Settings(_env_file=None, RAG_INDEX_PATH="").RAG_INDEX_PATH
    assert default_path == config_module.DEFAULT_RAG_INDEX_PATH


def test_rag_settings_reject_invalid_url_and_chunk_overlap():
    with pytest.raises(ValueError, match="RAG 服务 URL"):
        Settings(_env_file=None, RAG_EMBEDDING_BASE_URL="ftp://embedding.invalid")
    with pytest.raises(ValueError, match="RAG_CHUNK_OVERLAP"):
        Settings(_env_file=None, RAG_CHUNK_SIZE=100, RAG_CHUNK_OVERLAP=100)
    config = Settings(_env_file=None)
    with pytest.raises(ValueError, match="RAG_CHUNK_OVERLAP"):
        config.RAG_CHUNK_SIZE = 100
