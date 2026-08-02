from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from kd1_anime.agents.base import BaseAgent
from kd1_anime.config import settings


class TextResult(BaseModel):
    text: str


class StubAgent(BaseAgent):
    def call_llm(self, **kwargs):
        return r'{"text":"line1\nnext"}'


def test_valid_json_escape_is_not_rewritten():
    result = StubAgent().call_llm_json("system", "user", TextResult)
    assert result.text == "line1\nnext"


class PositiveResult(BaseModel):
    value: int


class StubListAgent(BaseAgent):
    def call_llm(self, **kwargs):
        return '{"items":[{"value":1},{"value":"invalid"}]}'


def test_json_list_rejects_partially_invalid_output():
    with pytest.raises(RuntimeError, match="拒绝使用残缺结果"):
        StubListAgent().call_llm_json_list("system", "user", PositiveResult)


def test_stream_is_closed_when_iteration_fails():
    class BrokenStream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            raise RuntimeError("stream failed")

        def close(self):
            self.closed = True

    stream = BrokenStream()

    class FakeCompletions:
        def create(self, **kwargs):
            return stream

    class StreamAgent(BaseAgent):
        @property
        def client(self):
            return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    agent = StreamAgent()
    try:
        agent._stream_llm({"model": "x", "messages": []})
    except RuntimeError as exc:
        assert "stream failed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert stream.closed is True


def _chunk(content=None, reasoning=None, finish=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, reasoning_content=reasoning),
                finish_reason=finish,
            )
        ]
    )


def test_empty_json_mode_response_falls_back_to_prompt_only(monkeypatch):
    monkeypatch.setattr(settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://test.local/v1")
    monkeypatch.setattr(settings, "LLM_MODEL", "test-model")
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs.copy())
            if "response_format" in kwargs:
                # 推理模型在 json 模式下返回空内容
                return iter([_chunk(finish="stop")])
            return iter([_chunk('{"items": [{"value": 1}]}', finish="stop")])

    class CompatibleAgent(BaseAgent):
        @property
        def client(self):
            return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    result = CompatibleAgent().call_llm(json_mode=True)

    assert result == '{"items": [{"value": 1}]}'
    assert len(calls) == 2
    assert calls[0]["stream"] is True
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]


def test_empty_stream_response_gets_max_tokens_boost(monkeypatch):
    # 默认 LLM_SILENT_STREAM=True: stream=False 也走静默流式。
    # 空响应(推理模型耗尽预算) → 补充 max_tokens 后重试。
    monkeypatch.setattr(settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://test.local/v1")
    monkeypatch.setattr(settings, "LLM_MODEL", "test-model")
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs.copy())
            if "max_tokens" not in kwargs:
                return iter([_chunk(finish="stop")])
            return iter([_chunk('{"ok": true}', finish="stop")])

    class CompatibleAgent(BaseAgent):
        @property
        def client(self):
            return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    result = CompatibleAgent().call_llm()

    assert result == '{"ok": true}'
    assert len(calls) == 2
    assert calls[0]["stream"] is True
    assert "max_tokens" not in calls[0]
    assert calls[1]["max_tokens"] == settings.LLM_EMPTY_RETRY_MAX_TOKENS


def test_non_stream_empty_falls_back_to_silent_stream_when_silent_disabled(monkeypatch):
    # LLM_SILENT_STREAM=False 时保持旧行为: 非流式空响应 → 静默流式重试。
    monkeypatch.setattr(settings, "LLM_SILENT_STREAM", False)
    monkeypatch.setattr(settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://test.local/v1")
    monkeypatch.setattr(settings, "LLM_MODEL", "test-model")
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs.copy())
            if not kwargs.get("stream"):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=None),
                            finish_reason="stop",
                        )
                    ]
                )
            return iter([_chunk('{"ok": true}', finish="stop")])

    class CompatibleAgent(BaseAgent):
        @property
        def client(self):
            return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    result = CompatibleAgent().call_llm()

    assert result == '{"ok": true}'
    assert len(calls) == 2
    assert "stream" not in calls[0]
    assert calls[1]["stream"] is True


def test_length_stop_reports_token_limit_without_compatibility_retries(monkeypatch):
    monkeypatch.setattr(settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://test.local/v1")
    monkeypatch.setattr(settings, "LLM_MODEL", "test-model")
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs.copy())
            return iter([_chunk(finish="length")])

    class LimitedAgent(BaseAgent):
        @property
        def client(self):
            return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    with pytest.raises(RuntimeError, match="max_tokens"):
        LimitedAgent().call_llm(json_mode=True, max_tokens=4096)

    assert len(calls) == 1
