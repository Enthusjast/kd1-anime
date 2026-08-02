from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prompt_toolkit.keys import Keys
from rich.console import Console

import kd1_anime.orchestrator as orchestrator_module
import kd1_anime.tui as tui_module
from kd1_anime.config import settings
from kd1_anime.tui import ChatSession, Clarifier, _input_bindings, _insert_newline, _submit_input


def test_clarifier_fallback_keeps_all_user_answers():
    clarifier = Clarifier()
    clarifier.history.extend(
        [
            {"role": "user", "content": "解释傅里叶级数"},
            {"role": "assistant", "content": "目标受众是谁？"},
            {"role": "user", "content": "面向高中生"},
            {"role": "assistant", "content": "视频多长？"},
            {"role": "user", "content": "三分钟"},
        ]
    )

    fallback = clarifier.build_fallback_prompt("解释傅里叶级数")

    assert "面向高中生" in fallback
    assert "三分钟" in fallback
    assert "目标受众是谁" not in fallback


def test_enter_submits_input():
    buffer = Mock()

    _submit_input(SimpleNamespace(current_buffer=buffer, data="\r"))

    buffer.validate_and_handle.assert_called_once_with()
    buffer.insert_text.assert_not_called()


def test_insert_newline_handler_inserts_newline():
    buffer = Mock()

    _insert_newline(SimpleNamespace(current_buffer=buffer))

    buffer.insert_text.assert_called_once_with("\n")


@pytest.mark.parametrize(
    "keys",
    [
        (Keys.ControlJ,),
        (Keys.Escape, Keys.ControlM),
        (Keys.Escape, "[", "1", "3", ";", "2", "u"),
        (Keys.Escape, "[", "1", "3", ";", "5", "u"),
    ],
)
def test_shift_and_ctrl_enter_bindings_insert_newline(keys):
    bindings = _input_bindings.get_bindings_for_keys(keys)

    assert any(binding.handler is _insert_newline for binding in bindings)


@pytest.mark.parametrize(
    "data",
    ["\x1b[27;2;13~", "\x1b[27;5;13~"],
)
def test_modify_other_enter_sequences_insert_newline(data):
    buffer = Mock()

    _submit_input(SimpleNamespace(current_buffer=buffer, data=data))

    buffer.insert_text.assert_called_once_with("\n")
    buffer.validate_and_handle.assert_not_called()


def test_clarifier_accepts_only_strict_ready_payload():
    clarifier = Clarifier()

    assert (
        clarifier.extract_ready('```json\n{"READY": true, "prompt": "  解释傅里叶级数  "}\n```')
        == "解释傅里叶级数"
    )


@pytest.mark.parametrize(
    "response",
    [
        '{"READY": "true", "prompt": "需求"}',
        '{"READY": 1, "prompt": "需求"}',
        '{"READY": true, "prompt": 123}',
        '{"READY": true, "prompt": "   "}',
        '{"READY": false, "prompt": "需求"}',
    ],
)
def test_clarifier_rejects_invalid_ready_payload(response):
    assert Clarifier().extract_ready(response) is None


def test_clarifier_rejects_oversized_ready_prompt(monkeypatch):
    monkeypatch.setattr(settings, "MAX_PROMPT_CHARS", 100)
    prompt = "x" * 101
    response = f'{{"READY": true, "prompt": "{prompt}"}}'

    assert Clarifier().extract_ready(response) is None


def test_clarifier_does_not_display_internal_ready_json(monkeypatch):
    clarifier = Clarifier()
    payload = '{"READY": true, "prompt": "面向高中生解释勾股定理"}'
    captured_call = {}
    output = StringIO()

    def fake_call_llm(**kwargs):
        captured_call.update(kwargs)
        return payload

    monkeypatch.setattr(clarifier.agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(tui_module, "console", Console(file=output, force_terminal=False))

    assert clarifier.ask("高中生") == payload
    assert captured_call["stream"] is False
    assert payload not in output.getvalue()
    assert "AI:" not in output.getvalue()


def test_clarifier_displays_question_after_buffering(monkeypatch):
    clarifier = Clarifier()
    output = StringIO()

    monkeypatch.setattr(
        clarifier.agent,
        "call_llm",
        lambda **kwargs: "你希望视频时长是多少？",
    )
    monkeypatch.setattr(tui_module, "console", Console(file=output, force_terminal=False))

    clarifier.ask("解释勾股定理")

    rendered = output.getvalue()
    assert "AI:" in rendered
    assert "你希望视频时长是多少？" in rendered


def test_pipeline_error_is_concise_and_does_not_render_markup(monkeypatch):
    class BrokenOrchestrator:
        def run(self, *args, **kwargs):
            raise FileNotFoundError(2, "No such file or directory")

    output = StringIO()
    monkeypatch.setattr(orchestrator_module, "Orchestrator", BrokenOrchestrator)
    monkeypatch.setattr(tui_module, "console", Console(file=output, force_terminal=False))
    monkeypatch.setattr(settings, "LLM_DEBUG", False)

    ChatSession()._run_pipeline("test prompt")

    rendered = output.getvalue()
    assert "生成失败: [Errno 2] No such file or directory" in rendered
    assert "[bold red]" not in rendered
    assert "Traceback" not in rendered
