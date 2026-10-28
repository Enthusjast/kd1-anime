import json

import pytest
from pydantic import ValidationError
from test_application import application_boundary as application_boundary

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


def test_completed_pre_budget_v10_run_resumes_without_new_requests(application_boundary):
    boundary = application_boundary
    video = boundary.app.generate("Show a one-second pause")
    root = next((boundary.workspace / "runs").iterdir())
    manifest_path = root / "manifest.json"
    raw = json.loads(manifest_path.read_text())
    assert raw["schema_version"] == 10 and raw["status"] == "completed"
    values = json.loads(raw["run_config"]["values_json"])
    for key in (
        "MAX_RUN_LLM_TOKENS",
        "MAX_RUN_LLM_COST_USD",
        "LLM_MODEL_PRICES_USD",
        "LLM_CONTEXT_SAFETY_TOKENS",
    ):
        values.pop(key)
    raw["run_config"]["values_json"] = json.dumps(values)
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    before = manifest_path.read_bytes()
    calls, submissions = list(boundary.calls), list(boundary.backend.sources)
    assert boundary.app.resume(root.name) == video
    assert manifest_path.read_bytes() == before
    assert boundary.calls == calls and boundary.backend.sources == submissions
