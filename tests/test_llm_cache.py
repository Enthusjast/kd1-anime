from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from kd1_anime.agents.base import BaseAgent
from kd1_anime.config import LLMRuntimeProfile
from kd1_anime.llm_cache import LLMResponseCache, make_cache_key


def profile() -> LLMRuntimeProfile:
    return LLMRuntimeProfile(
        label="测试",
        env_prefix="LLM",
        api_key="secret",
        base_url="https://example.invalid/v1",
        model="model-a",
        send_max_tokens=True,
        temperature=0.3,
        max_tokens=100,
        max_retries=2,
        retry_base_delay=0.1,
        timeout_connect=1,
        timeout_read=10,
        healthcheck_timeout=1,
        silent_stream=True,
        empty_retry_max_tokens=100,
        json_repair_attempts=1,
        use_json_mode=True,
        debug=False,
    )


def test_cache_round_trip_is_private_and_does_not_depend_on_api_key(tmp_path: Path):
    first = profile()
    second = replace(first, api_key="other-secret")
    messages = [{"role": "user", "content": "hello"}]
    key1 = make_cache_key(
        first,
        messages,
        temperature=0.3,
        max_tokens=100,
        json_mode=False,
        allow_truncated=False,
    )
    key2 = make_cache_key(
        second,
        messages,
        temperature=0.3,
        max_tokens=100,
        json_mode=False,
        allow_truncated=False,
    )
    assert key1 == key2

    cache = LLMResponseCache(tmp_path / "cache" / "llm.sqlite3", max_entries=2)
    cache.set(key1, "response")
    assert cache.get(key2) == "response"
    assert cache.path.stat().st_mode & 0o777 == 0o600
    assert cache.path.parent.stat().st_mode & 0o777 == 0o700
    assert cache.stats.hits == 1


def test_cache_is_bounded_and_can_be_cleared(tmp_path: Path):
    cache = LLMResponseCache(tmp_path / "llm.sqlite3", max_entries=1)
    for index in range(3):
        cache.set(str(index), f"response-{index}")

    assert cache.get("0") is None
    assert cache.get("2") == "response-2"
    assert cache.clear() == 1
    assert cache.get("2") is None


def test_base_agent_reuses_complete_non_stream_response(monkeypatch, tmp_path: Path):
    from kd1_anime.config import settings

    monkeypatch.setattr(settings, "LLM_API_KEY", "key")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(settings, "LLM_MODEL", "model")
    monkeypatch.setattr(settings, "LLM_SILENT_STREAM", False)
    monkeypatch.setattr(settings, "LLM_CACHE_ENABLED", True)
    monkeypatch.setattr(settings, "LLM_CACHE_PATH", tmp_path / "llm.sqlite3")
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="cached response"),
                        finish_reason="stop",
                    )
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(BaseAgent, "client", property(lambda _self: fake_client))

    assert BaseAgent().call_llm(user_message="same") == "cached response"
    assert BaseAgent().call_llm(user_message="same") == "cached response"
    assert len(calls) == 1
