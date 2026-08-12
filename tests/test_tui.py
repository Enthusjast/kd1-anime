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


def test_clarifier_accepts_ready_payload_with_raw_newlines():
    """LLM 在长中文 prompt 里输出未转义的原始换行时也应识别为 READY。

    此前 json.loads 会因 "Invalid control character" 失败, READY 被误判为
    普通提问打印 "AI:" 并卡在澄清循环。
    """
    clarifier = Clarifier()
    response = '{"READY": true, "prompt": "## 动画目标\n制作一个数学动画视频，面向初中生演示两个代数公式。\n\n## 核心内容\n### 公式一：(a+b)² = a² + 2ab + b²"}'
    refined = clarifier.extract_ready(response)
    assert refined is not None
    assert refined.startswith("## 动画目标")
    assert "### 公式一" in refined


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

    session = ChatSession()
    assert session._run_pipeline("test prompt") is False
    assert session.exit_code == 1

    rendered = output.getvalue()
    assert "生成失败: [Errno 2] No such file or directory" in rendered
    assert "[bold red]" not in rendered
    assert "Traceback" not in rendered


def test_show_banner_returns_true_after_resume_selection(monkeypatch):
    """用户选择恢复运行后 _show_banner 返回 True, 会话应结束。"""
    session = ChatSession()
    monkeypatch.setattr(session, "_check_interrupted_runs", lambda: True)
    assert session._show_banner() is True


def test_show_banner_returns_false_when_no_resume(monkeypatch):
    """没有选择恢复运行 → 返回 False, 继续进入新需求输入。"""
    session = ChatSession()
    monkeypatch.setattr(session, "_check_interrupted_runs", lambda: False)
    assert session._show_banner() is False


def test_run_exits_after_resume_instead_of_new_prompt(monkeypatch):
    """选择恢复运行后 run() 应直接返回, 不再落入"描述你的需求"新提示。"""
    session = ChatSession()
    monkeypatch.setattr(session, "_show_banner", lambda: True)
    calls = []
    monkeypatch.setattr(session, "_get_initial_prompt", lambda: calls.append("prompt"))
    session.run()
    assert calls == []


def test_check_interrupted_runs_triggers_resume_and_returns_true(monkeypatch, tmp_path):
    """在真实 manifest 下选择恢复运行: 返回 True 并调用 _resume_run。"""
    from datetime import datetime, timedelta, timezone

    from kd1_anime.agents.planner import ScenePlan
    from kd1_anime.run_store import RunManifest, StoredSceneState

    def make_plan(sid):
        return ScenePlan(
            scene_id=sid,
            title=f"scene {sid}",
            duration_seconds=10,
            purpose="test",
            math_concept="circle",
            visual_design="dark",
            camera_movement="fixed",
            visual_flow=["show"],
            key_moments=["pause"],
            computation="radius=1",
        )

    workspace = tmp_path / "workspace"
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)

    run_id = "20260804-223002-337db553"
    root = workspace / "runs" / run_id
    root.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    manifest = RunManifest(
        run_id=run_id,
        status="failed",
        state="MONITORING",
        user_prompt="p",
        output_path=str(tmp_path / "out.mp4"),
        created_at=now - timedelta(hours=1),
        updated_at=now,
        scenes={
            1: StoredSceneState(plan=make_plan(1), rendered=True),
            2: StoredSceneState(plan=make_plan(2), rendered=True),
            3: StoredSceneState(plan=make_plan(3), rendered=True),
            4: StoredSceneState(plan=make_plan(4), give_up=True),
        },
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")

    session = ChatSession()
    resumed: list[str] = []
    monkeypatch.setattr("builtins.input", lambda _: "1")
    monkeypatch.setattr(session, "_resume_run", resumed.append)

    assert session._check_interrupted_runs() is True
    assert resumed == [run_id]
