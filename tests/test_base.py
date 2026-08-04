from types import SimpleNamespace
from typing import Literal

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


def test_length_empty_boosts_max_tokens_then_recovers(monkeypatch):
    # 空内容 + finish=length: 推理模型耗尽输出预算。
    # 不应立即判死, 而应补充 max_tokens 后重试, 重试成功则返回内容。
    monkeypatch.setattr(settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://test.local/v1")
    monkeypatch.setattr(settings, "LLM_MODEL", "test-model")
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs.copy())
            if len(calls) == 1:
                return iter([_chunk(finish="length")])
            return iter([_chunk('{"ok": true}', finish="stop")])

    class CompatibleAgent(BaseAgent):
        @property
        def client(self):
            return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    result = CompatibleAgent().call_llm(json_mode=True, max_tokens=4096)

    assert result == '{"ok": true}'
    assert calls[0]["max_tokens"] == 4096
    assert calls[1]["max_tokens"] == max(4096 * 2, settings.LLM_EMPTY_RETRY_MAX_TOKENS)


def test_length_empty_raises_only_after_retries_exhausted(monkeypatch):
    # 补充 max_tokens 后仍持续空 + length: 走正常重试, 重试耗尽才抛错,
    # 而不是第一次空响应就直接把整个场景判死。
    monkeypatch.setattr(settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://test.local/v1")
    monkeypatch.setattr(settings, "LLM_MODEL", "test-model")
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 1)
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs.copy())
            return iter([_chunk(finish="length")])

    class CompatibleAgent(BaseAgent):
        @property
        def client(self):
            return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    with pytest.raises(RuntimeError, match="仍然失败"):
        CompatibleAgent().call_llm(json_mode=True, max_tokens=4096)

    # 1 次补充 max_tokens (不消耗业务重试) + LLM_MAX_RETRIES 次业务重试
    assert len(calls) == 2
    assert calls[1]["max_tokens"] == max(4096 * 2, settings.LLM_EMPTY_RETRY_MAX_TOKENS)


def test_json_schema_error_retries_then_recovers():
    """结构化输出不合规 (如 severity='none') → 带错误反馈重试, 重试成功返回解析结果。"""

    class SevModel(BaseModel):
        is_valid: bool
        severity: Literal["info", "minor", "major"]

    class RepairAgent(BaseAgent):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.last_user_message = ""

        def call_llm(self, **kwargs):
            self.calls += 1
            self.last_user_message = kwargs.get("user_message", "")
            if self.calls == 1:
                return '{"is_valid": true, "severity": "none"}'
            return '{"is_valid": true, "severity": "info"}'

    agent = RepairAgent()
    result = agent.call_llm_json("system", "user", SevModel)

    assert result.is_valid is True
    assert result.severity == "info"
    assert agent.calls == 2
    assert "未通过结构校验" in agent.last_user_message


def test_json_schema_error_raises_after_repair_attempts(monkeypatch):
    """持续不合规 → 修复重试耗尽后仍抛错 (不无限循环, 不零重试判死)。"""
    monkeypatch.setattr(settings, "LLM_JSON_REPAIR_ATTEMPTS", 1)

    class SevModel(BaseModel):
        is_valid: bool
        severity: Literal["info", "minor", "major"]

    class RepairAgent(BaseAgent):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def call_llm(self, **kwargs):
            self.calls += 1
            return '{"is_valid": true, "severity": "none"}'

    agent = RepairAgent()
    with pytest.raises(RuntimeError, match="输出不符合预期结构"):
        agent.call_llm_json("system", "user", SevModel)

    assert agent.calls == 2  # 1 次初始 + LLM_JSON_REPAIR_ATTEMPTS 次修复


def test_json_list_schema_error_retries_then_recovers():
    """列表项不合规 → 带错误反馈重试后成功。"""

    class ItemModel(BaseModel):
        value: int

    class RepairListAgent(BaseAgent):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def call_llm(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return '{"items": [{"value": 1}, {"value": "bad"}]}'
            return '{"items": [{"value": 1}, {"value": 2}]}'

    agent = RepairListAgent()
    result = agent.call_llm_json_list("system", "user", ItemModel)

    assert [r.value for r in result] == [1, 2]
    assert agent.calls == 2
