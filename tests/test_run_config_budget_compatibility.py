import json

import pytest
from pydantic import ValidationError

from kd1_anime.domain.runtime import RunConfig


def test_pre_budget_v10_settings_load_without_fabricating_limits_or_prices():
    current = json.loads(RunConfig.capture().values_json)
    for key in (
        "MAX_RUN_LLM_TOKENS",
        "MAX_RUN_LLM_COST_USD",
        "LLM_MODEL_PRICES_USD",
        "LLM_CONTEXT_SAFETY_TOKENS",
    ):
        current.pop(key)
    restored = RunConfig(values_json=json.dumps(current))
    settings = restored.resolve()
    assert settings.MAX_RUN_LLM_TOKENS is None and settings.MAX_RUN_LLM_COST_USD is None
    assert settings.LLM_MODEL_PRICES_USD == {}
    assert settings.LLM_CONTEXT_SAFETY_TOKENS == 1024


def test_preexisting_required_run_settings_are_not_fabricated():
    current = json.loads(RunConfig.capture().values_json)
    current.pop("LLM_MODEL")
    with pytest.raises(ValidationError):
        RunConfig(values_json=json.dumps(current))
