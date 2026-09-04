"""
TUI 交互模块
提供类似 Claude Code 的终端交互体验:
  用户输入 → 需求澄清 → 确认 → 流式生成 (带状态指示)

输入方式:
  使用 prompt_toolkit (multiline=True):
  - Enter 提交, Shift+Enter 或 Ctrl+Enter 换行
  - 粘贴多行文本不会被截断
"""

import locale
import signal
import sys
from contextlib import suppress

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from kd1_anime.config import settings

console = Console()

# ---------------------------------------------------------------------------
# 多行输入 — prompt_toolkit (multiline=True)
# Enter 提交, Shift+Enter 或 Ctrl+Enter 换行, 粘贴多行不截断
# ---------------------------------------------------------------------------

_MODIFIED_ENTER_DATA = {
    "\x1b[13;2u",
    "\x1b[13;5u",
    "\x1b[27;2;13~",
    "\x1b[27;5;13~",
}

_input_bindings = KeyBindings()


@_input_bindings.add("enter")
def _submit_input(event) -> None:
    if getattr(event, "data", "") in _MODIFIED_ENTER_DATA:
        _insert_newline(event)
        return
    event.current_buffer.validate_and_handle()


@_input_bindings.add("c-j")
@_input_bindings.add("escape", "enter")
@_input_bindings.add("escape", "[", "1", "3", ";", "2", "u")
@_input_bindings.add("escape", "[", "1", "3", ";", "5", "u")
def _insert_newline(event) -> None:
    event.current_buffer.insert_text("\n")


_prompt_session = PromptSession(multiline=True, key_bindings=_input_bindings)


def _read_multiline(prompt_text: str = "") -> str:
    try:
        return _prompt_session.prompt(prompt_text).strip()
    except EOFError:
        return ""
    except KeyboardInterrupt:
        raise


CLARIFIER_SYSTEM_PROMPT = """你是一个专业的数学动画需求分析师.用户想用 Manim 制作数学动画,你的任务是通过对话明确他们的需求.

## 工作流程 — 每轮对话遵循以下步骤

### 第一步: 在内部评估信息缺口
阅读用户的初始描述和所有历史对话,在内部判断:
- 已明确的信息有哪些
- 还缺什么关键信息来生成高质量动画规划
- 不要问用户已经明确说过的事情
- 不要向用户展示分析过程、已明确列表或信息缺口列表

### 第二步: 提一个问题
从信息缺口中选择当前最关键的一个提问.
问题必须具体、可回答,不要像问卷调查一样笼统.

### 第三步 (收到回答后): 更新评估
将新信息加入"已明确"列表,重新执行第一步.
如果新回答引出了需要进一步明确的细节,将其加入缺口列表.

## 你需要收集的信息维度 (按需,不必全问)
- 动画目标: 严格跟随文稿 vs 仅表现核心概念 vs 混合 (有画面描述处还原,无描述处自行策划)
- 核心数学/物理概念及其深度
- 目标受众和前置知识水平
- 视频总时长预期
- 对特定动画风格的偏好
- 是否需要特定的颜色方案或视觉主题

## 动画风格参考 (仅在用户询问或需要二选一时使用,不要主动描述风格细节)
- 3Blue1Brown 风: 深色纯色背景, 清晰几何图形, 数学变换动画, 颜色意义编码
- 现代教学风: 简洁浅色背景, 清晰图形, 适度文字
- 炫酷展示风: 强视觉冲击, 粒子光效, 动态镜头

## 结束条件
当你判断信息足够生成完整的动画规划时,输出:
{"READY": true, "prompt": "整合后的完整需求描述,包含所有已确认的细节"}
prompt 字段必须使用清晰的 Markdown 结构组织 (用 ## 分小节、- 列表罗列要点), 让后续规划阶段能直接按小节解析使用。

还需要信息时直接提问,不要输出 JSON.
"""


class Clarifier:
    """需求澄清对话管理"""

    def __init__(self):
        from kd1_anime.agents.base import BaseAgent

        class _Agent(BaseAgent):
            name = "Clarifier"

        self.agent = _Agent()
        self.history: list[dict] = [
            {"role": "system", "content": CLARIFIER_SYSTEM_PROMPT},
        ]

    def ask(self, user_input: str) -> str:
        """发送用户输入；内部 READY 载荷缓冲解析，不展示给用户。"""
        self.history.append({"role": "user", "content": user_input})

        while True:
            try:
                # 非流式接收，以便区分用户问题和内部 READY JSON
                response = self.agent.call_llm(messages=self._bounded_history(), stream=False)
                self.history.append({"role": "assistant", "content": response})
                if self.extract_ready(response) is None:
                    # 非 READY 响应按 Markdown 渲染。READY JSON 是内部协议，
                    # 已在上面拦截，不应直接展示给用户。
                    console.print("[dim cyan]AI:[/]")
                    console.print(Markdown(response))
                return response
            except Exception as e:
                console.print()
                console.print(f"Clarifier LLM 错误: {e}", style="yellow", markup=False)
                try:
                    answer = Confirm.ask(
                        "[bold]再试一次?[/] (y = 重试, n = 退出)",
                        default=True,
                        console=console,
                    )
                except (EOFError, KeyboardInterrupt):
                    answer = False
                if not answer:
                    # 移除刚才加入但未得到回复的 user message
                    self.history.pop()
                    raise RuntimeError("Clarifier 失败, 用户选择退出") from e

    @staticmethod
    def _clip_text(content: str, limit: int) -> str:
        """按字符裁剪长消息，同时保留开头和结尾的关键信息。"""

        if len(content) <= limit:
            return content
        if limit <= 80:
            return content[:limit]
        marker = "\n...[内容因上下文预算被裁剪]...\n"
        available = max(1, limit - len(marker))
        head = (available + 1) // 2
        tail = available - head
        return content[:head] + marker + (content[-tail:] if tail else "")

    def _bounded_history(self) -> list[dict]:
        """构造有界的澄清上下文，保留系统提示、初始需求和最近回答。"""

        if not self.history:
            return []
        budget = settings.MAX_CLARIFY_CONTEXT_CHARS
        system = dict(self.history[0])
        system_content = str(system.get("content", ""))
        system_limit = min(len(system_content), max(1000, budget // 4))
        system["content"] = self._clip_text(system_content, system_limit)
        if len(self.history) == 1:
            return [system]

        # 初始需求是后续澄清的锚点，始终保留；单条超长消息也不能独占整个预算。
        first = dict(self.history[1])
        first_limit = max(500, min(budget // 3, budget - len(system["content"]) - 500))
        first["content"] = self._clip_text(str(first.get("content", "")), first_limit)
        used = len(str(system.get("content", ""))) + len(str(first.get("content", "")))
        recent: list[dict] = []
        for original in reversed(self.history[2:]):
            message = dict(original)
            content = str(message.get("content", ""))
            available = budget - used - 2
            if available <= 0:
                break
            clipped = self._clip_text(content, available)
            message["content"] = clipped
            recent.append(message)
            used += len(clipped) + 2
            if len(clipped) < len(content):
                break
        return [system, first, *reversed(recent)]

    def extract_ready(self, response: str) -> str | None:
        """
        如果 LLM 表示需求已明确, 返回精炼后的 prompt; 否则返回 None.

        健壮解析: 先去 markdown fence, 再用括号配平提取 JSON, 容忍
        散文包裹 / 换行格式化 / 嵌套花括号.
        """
        import json

        text = response.strip()
        # 快速短路: 完全没有大括号, 不可能是 READY 信号
        if "{" not in text:
            return None

        # Clarifier 的协议是 JSON 对象. 不能让前置说明中的数学区间
        # (例如 ``[-5, 5]``) 抢先成为待解析的 JSON。
        json_str = self.agent._extract_json(text, expected_type="object")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # LLM 常在长中文 JSON 里输出未转义的原始换行 (Invalid control character),
            # 先修复再解析一次, 否则 READY 会被误判为普通提问而卡在澄清循环。
            repaired = self.agent._escape_unescaped_quotes_in_json(json_str)
            repaired = self.agent._escape_control_chars_in_json(repaired)
            repaired = self.agent._fix_latex_escapes_in_json(repaired)
            repaired = self.agent._close_truncated_json(repaired)
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError:
                return None

        if not isinstance(data, dict) or data.get("READY") is not True:
            return None

        prompt = data.get("prompt")
        if not isinstance(prompt, str):
            return None
        prompt = prompt.strip()
        if not prompt or len(prompt) > settings.MAX_PROMPT_CHARS:
            return None

        return prompt

    def build_fallback_prompt(self, initial_prompt: str) -> str:
        """在模型未输出 READY 时保留所有用户实际提供的信息。"""

        additions: list[str] = []
        initial_normalized = initial_prompt.strip()
        for message in self.history:
            if message.get("role") != "user":
                continue
            content = str(message.get("content", "")).strip()
            if content and content != initial_normalized and content not in additions:
                additions.append(content)
        if not additions:
            return self._clip_text(initial_normalized, settings.MAX_PROMPT_CHARS)
        prefix = f"{initial_normalized}\n\n用户在澄清过程中补充的信息：\n"
        result = prefix
        for content in additions:
            item = f"- {content}\n"
            available = settings.MAX_PROMPT_CHARS - len(result)
            if available <= 0:
                break
            result += self._clip_text(item, available)
            if len(item) > available:
                break
        return result[: settings.MAX_PROMPT_CHARS]


def _setup_terminal() -> None:
    """配置终端以正确处理多字节字符输入."""
    # 1. 确保 locale 正确 (Python I/O 据此选择编码)
    with suppress(locale.Error):
        locale.setlocale(locale.LC_ALL, "")

    # 2. 确保 stdin 使用 UTF-8 (修复损坏的 surrogate 字节)
    if hasattr(sys.stdin, "buffer"):
        with suppress(AttributeError, OSError):
            sys.stdin.reconfigure(encoding="utf-8", errors="surrogateescape")


def _clean_input(text: str) -> str:
    """清理终端输入中的损坏字符.

    浏览器终端 backspace 中文时可能产生三种垃圾:
    1. 被插入字面的控制字符 (\\x08 BS, \\x7f DEL, \\x00 NUL)
    2. 无法解码的孤立字节 → Python surrogateescape 保留为 \\udcXX
    3. 残缺字节与相邻字符重组为错误但合法的 Unicode
    """
    import re

    # 删除 ANSI 转义序列
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)

    # 删除低值控制字符 (BS, DEL, NUL 等, 保留 \\t \\n)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # surrogateescape 产生的孤立 surrogate → 删除
    text = re.sub(r"[\udc80-\udcff]", "", text)

    # 最后一道: encode→decode 清理残留的无效序列
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    text = re.sub(r"�+", "", text)

    return text


class ChatSession:
    """交互式会话管理"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.clarifier: Clarifier | None = None
        # 交互模式也必须向 shell 传递流水线失败/中断语义，不能只在终端
        # 打印红字后以退出码 0 结束。
        self.exit_code = 0

    def run(self) -> int:
        """启动完整的交互会话"""

        # 设置信号处理，避免退出时的 threading 警告
        def _signal_handler(signum, frame):
            raise KeyboardInterrupt()

        signal.signal(signal.SIGINT, _signal_handler)

        _setup_terminal()
        try:
            # 恢复运行是显式的 `kd1-anime resume <run-id>` 命令；交互启动
            # 只显示横幅，不扫描历史运行。
            self._show_banner()

            # 在构造 Agent 前检查完整的 OpenAI-compatible 配置。
            try:
                settings.require_llm_key()
            except ValueError as exc:
                console.print(
                    Panel(
                        str(exc),
                        title="配置错误",
                        border_style="red",
                    )
                )
                self.exit_code = 1
                return self.exit_code

            # 获取初始需求
            user_prompt = self._get_initial_prompt()
            if not user_prompt:
                return self.exit_code

            # 此刻才构造 Clarifier (Key 已确认存在)
            self.clarifier = Clarifier()

            # 需求澄清 + 确认 (可循环: 不满意就继续讨论)
            while True:
                refined_prompt = self._run_clarification(user_prompt)
                if not refined_prompt:
                    return self.exit_code

                if self._confirm_prompt(refined_prompt):
                    break

                # 用户选 n → 让用户说出哪里不满意, 再喂给 Clarifier
                console.print("[dim]哪里需要调整? (输入后回车)[/]")
                feedback = _read_multiline(">>> ")
                if not feedback or feedback.lower() in ("quit", "exit", "q"):
                    console.print("[dim]已退出[/]")
                    return self.exit_code
                user_prompt = (
                    f"用户对当前需求描述不满意: {feedback}\n\n"
                    f"当前的需求描述是: {refined_prompt}\n\n"
                    f"请根据用户的反馈进一步澄清和完善."
                )

            # 执行生成流水线
            self._run_pipeline(refined_prompt)
            return self.exit_code

        except KeyboardInterrupt:
            console.print("\n[dim]已取消[/]")
            self.exit_code = 130
            return self.exit_code

    def _show_banner(self) -> None:
        """显示欢迎横幅并进入新会话。

        恢复运行是显式操作，使用 ``kd1-anime resume <run-id>``；启动交互
        会话时不扫描历史 manifest，避免旧运行阻塞当前需求输入。
        """
        banner = Text()
        banner.append("  ╔═══════════════════════════════════════╗\n", style="bold cyan")
        banner.append("  ║       ", style="bold cyan")
        banner.append("kd1-anime", style="bold white")
        banner.append(" · ", style="dim cyan")
        banner.append("Manim 动画生成器", style="bold cyan")
        banner.append("    ║\n", style="bold cyan")
        banner.append("  ╚═══════════════════════════════════════╝", style="bold cyan")
        console.print(banner)

        # 启动时展示所有会参与流水线的模型；即使某个可选能力未启用，
        # 也显示已配置的模型名，便于确认当前配置是否生效。
        model_name = settings.LLM_MODEL or "未配置"
        visual_model_name = settings.visual_llm_profile().model or "未配置"
        rag_state = "已启用" if settings.RAG_ENABLED else "未启用"
        embedding_model_name = settings.RAG_EMBEDDING_MODEL or "未配置"
        reranker_model_name = settings.RAG_RERANK_MODEL or "未配置"
        visual_state = "已启用" if settings.ENABLE_VISUAL_EVAL else "未启用"
        console.print(f"  主模型: [dim]{model_name}[/]")
        console.print(f"  视觉模型: [dim]{visual_model_name}[/] ([dim]{visual_state}[/])")
        console.print(f"  Embedding 模型: [dim]{embedding_model_name}[/] ([dim]{rag_state}[/])")
        console.print(f"  Reranker 模型: [dim]{reranker_model_name}[/] ([dim]{rag_state}[/])")

        console.print(
            "\n  输入你想制作的数学动画描述,我会帮你生成.\n  输入 [bold cyan]quit[/] 退出.\n",
        )

    def _get_initial_prompt(self) -> str | None:
        """获取用户的初始需求描述

        使用 prompt_toolkit, 支持多行粘贴和 Alt+Enter 换行.
        """
        console.print(Rule("描述你的需求", style="dim"))
        console.print()
        while True:
            try:
                raw = _read_multiline(">>> ")
            except EOFError:
                console.print("\n[dim]已退出[/]")
                return None
            except KeyboardInterrupt:
                console.print("\n[dim]已取消[/]")
                return None
            user_input = _clean_input(raw).strip()

            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[dim]已退出[/]")
                return None

            if not user_input:
                # 空输入不退出, 重新提示
                continue
            if len(user_input) > settings.MAX_PROMPT_CHARS:
                console.print(
                    f"[yellow]输入过长：{len(user_input)} 字符，"
                    f"最大允许 {settings.MAX_PROMPT_CHARS} 字符。[/]"
                )
                continue

            return user_input

    def _run_clarification(self, initial_prompt: str) -> str | None:
        """运行需求澄清对话"""
        max_rounds = settings.MAX_CLARIFY_ROUNDS

        console.print(Rule("需求澄清", style="dim"))
        console.print()

        try:
            # 第一轮: LLM 根据初始描述提问
            response = self.clarifier.ask(initial_prompt)
        except RuntimeError:
            # Clarifier 失败, 用户选择退出 → 直接用初始 prompt 走 pipeline
            console.print("[yellow]跳过需求澄清, 使用原始描述继续.[/]")
            self._show_refined(initial_prompt)
            return initial_prompt

        # 检查是否已经足够明确
        refined = self.clarifier.extract_ready(response)
        if refined:
            self._show_refined(refined)
            return refined

        # 开始多轮对话
        console.print()

        for _round_num in range(1, max_rounds + 1):
            try:
                raw = _read_multiline(">>> ")
            except EOFError:
                console.print("\n[dim]已退出[/]")
                return None
            except KeyboardInterrupt:
                console.print("\n[dim]已取消[/]")
                return None
            user_input = _clean_input(raw).strip()

            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[dim]已退出[/]")
                return None

            if not user_input:
                continue
            if len(user_input) > settings.MAX_PROMPT_CHARS:
                console.print(
                    f"[yellow]本轮输入过长，最大允许 {settings.MAX_PROMPT_CHARS} 字符。[/]"
                )
                continue

            try:
                response = self.clarifier.ask(user_input)
            except RuntimeError:
                fallback = self.clarifier.build_fallback_prompt(initial_prompt)
                console.print(
                    "\nClarifier 失败, 使用已收集到的信息继续.",
                    style="yellow",
                )
                self._show_refined(fallback)
                return fallback

            refined = self.clarifier.extract_ready(response)
            if refined:
                self._show_refined(refined)
                return refined

            console.print()

        # 达到最大轮次仍未 READY
        console.print(f"[yellow]已达到最大澄清轮次 ({max_rounds}), 使用当前收集到的信息继续.[/]")
        fallback = self.clarifier.build_fallback_prompt(initial_prompt)
        self._show_refined(fallback)
        return fallback

    @staticmethod
    def _show_refined(refined: str) -> None:
        """统一展示精炼后的需求"""
        console.print()
        console.print(
            Panel(
                Markdown(refined),
                title="[bold green]已明确的需求[/]",
                border_style="green",
            )
        )

    def _confirm_prompt(self, refined_prompt: str) -> bool:
        """让用户确认需求"""
        console.print()
        answer = Confirm.ask(
            "[bold]开始生成?[/]",
            default=True,
            console=console,
        )
        if not answer:
            console.print("[dim]已取消. 重新描述需求或输入 quit 退出.[/]")
        return answer

    def _run_pipeline(self, prompt: str) -> bool:
        """执行生成流水线,带进度指示"""
        console.print()
        console.print(Rule("开始生成", style="bold magenta"))
        console.print()

        from kd1_anime.dashboard import SceneDashboard
        from kd1_anime.orchestrator import Orchestrator

        # 并行阶段使用 Rich Live 仪表盘；非 TTY 自动降级为普通输出
        dashboard = SceneDashboard()
        dashboard_active = dashboard.start()

        def callback(event: str, data: dict) -> None:
            if dashboard_active:
                dashboard.on_event(event, data)
            else:
                self._pipeline_callback(event, data)

        try:
            orchestrator = Orchestrator()
            final_video = orchestrator.run(
                prompt,
                callback=callback,
                dry_run=self.dry_run,
                interactive=True,
            )
            if dashboard_active:
                dashboard.stop()
            self._show_completion(final_video)
            self.exit_code = 0
            return True
        except KeyboardInterrupt:
            if dashboard_active:
                dashboard.stop()
            console.print("\n[bold yellow]用户中断,正在清理...[/]")
            self.exit_code = 130
            raise
        except Exception as e:
            if dashboard_active:
                dashboard.stop()
            console.print()
            console.print(f"生成失败: {e}", style="bold red", markup=False)
            if settings.LLM_DEBUG:
                console.print_exception()
            self.exit_code = 1
            return False

    @staticmethod
    def _escape_markup(text: str) -> str:
        """转义 Rich markup 标记, 防止 LLM/系统输出中的 [] 被误解析"""
        return str(text).replace("[", "\\[")

    @staticmethod
    def _pipeline_callback(event: str, data: dict) -> None:
        """流水线状态回调 — 简洁输出当前步骤"""
        esc = ChatSession._escape_markup
        match event:
            case "run_started":
                console.print(
                    f"[dim]Run {esc(data.get('run_id', '?'))} · {esc(data.get('run_dir', ''))}[/]"
                )

            case "stage_start":
                stage = data.get("stage", "")
                match stage:
                    case "planning":
                        console.print(Rule("[bold magenta]场景概要[/]", style="magenta"))
                    case "detailing":
                        console.print(Rule("[bold magenta]导演分镜[/]", style="magenta"))
                    case "technical":
                        console.print(Rule("[bold magenta]技术实现设计[/]", style="magenta"))
                    case "coding":
                        console.print(Rule("[bold magenta]代码生成[/]", style="magenta"))
                    case "reviewing":
                        console.print(Rule("[bold magenta]代码审查[/]", style="magenta"))
                    case "dispatching":
                        console.print(Rule("[bold magenta]提交渲染[/]", style="magenta"))
                    case "monitoring":
                        console.print(Rule("[bold magenta]监控渲染[/]", style="magenta"))
                    case "fixing":
                        console.print(Rule("[bold magenta]自动修复[/]", style="magenta"))
                    case "visual_evaluating":
                        console.print(Rule("[bold magenta]视觉评估[/]", style="magenta"))
                    case "merging":
                        console.print(Rule("[bold magenta]视频拼接[/]", style="magenta"))
                    case "evaluating":
                        console.print(Rule("[bold magenta]质量评估[/]", style="magenta"))

            case "security_warning":
                console.print(f"[bold yellow]安全警告:[/] {esc(data.get('message', ''))}")

            case "rag_status":
                status = esc(data.get("status", "disabled"))
                warning = esc(data.get("warning", ""))
                console.print(f"  [dim]RAG:[/] {status}" + (f" — {warning}" if warning else ""))

            case "plan_complete":
                scenes = data.get("scenes", [])
                table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
                table.add_column("#", style="cyan", justify="center")
                table.add_column("场景", style="green")
                table.add_column("时长", style="magenta", justify="center")
                table.add_column("概念", style="yellow")
                table.add_column("叙事", style="dim")
                for scene in scenes:
                    table.add_row(
                        str(scene.scene_id),
                        Text(scene.title),
                        f"{scene.duration_seconds}s",
                        Text(scene.math_concept),
                        Text(scene.purpose),
                    )
                console.print(table)
                console.print()

            case "plan_reviewing":
                console.print(Rule("[bold magenta]计划正确性审查[/]", style="magenta"))

            case "continuity_contract_repaired":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [yellow]连续性合同已自动修复[/]")

            case "scene_detailing":
                scene_id = data.get("scene_id", "?")
                title = data.get("title", "")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [cyan]{esc(title)}[/] 开始生成分镜")

            case "scene_detailed":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [bold green]分镜完成 ✓[/]")

            case "scene_technical_planning":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [cyan]技术实现设计中[/]")

            case "scene_technical_ready":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [bold green]技术合同完成 ✓[/]")

            case "scene_technical_failed":
                scene_id = data.get("scene_id", "?")
                reason = esc(data.get("reason", ""))
                suffix = f"：{reason}" if reason else ""
                console.print(f"  [dim]▸[/] Scene {scene_id}: [bold red]技术合同失败 ✗[/]{suffix}")

            case "scene_plan_reviewing":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [cyan]计划正确性审查中[/]")

            case "scene_plan_review_pass":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [bold green]计划审查通过 ✓[/]")

            case "scene_plan_replanned":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [yellow]计划已重规划[/]")

            case "scene_plan_review_fail":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [bold red]计划审查失败 ✗[/]")

            case "scene_safe_fallback":
                scene_id = data.get("scene_id", "?")
                reason = esc(data.get("reason", ""))
                suffix = f"：{reason}" if reason else ""
                console.print(
                    f"  [dim]▸[/] Scene {scene_id}: [yellow]切换为保守教学方案[/]{suffix}"
                )

            case "scene_coding":
                scene_id = data.get("scene_id", "?")
                title = data.get("title", "")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [green]{esc(title)}[/] 开始生成代码")

            case "scene_coded":
                file_path = data.get("file_path", "")
                console.print(f"  [bold green]✓[/] [dim]{esc(file_path)}[/]")

            case "scene_review_pass":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [bold green]审查通过 ✓[/]")

            case "scene_reviewing":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [cyan]代码审查中[/]")

            case "scene_review_skipped":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [yellow]已跳过审查[/]")

            case "scene_rewriting":
                scene_id = data.get("scene_id", "?")
                reason = esc(data.get("reason", ""))
                console.print(f"  [dim]▸[/] Scene {scene_id}: [cyan]Coder 正在修正...[/]")
                if reason:
                    console.print(f"    [dim]原因: {reason}[/]")

            case "scene_review_fail":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [yellow]需修正[/]")

            case "scene_smoke_rendering":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [cyan]Smoke Render 检查中[/]")

            case "scene_smoke_rendered":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [bold green]Smoke Render 通过 ✓[/]")

            case "scene_submitted":
                scene_id = data.get("scene_id", "?")
                job_id = data.get("job_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id} → Job {job_id}")

            case "scene_artifact_invalid":
                scene_id = data.get("scene_id", "?")
                reason = esc(data.get("reason", ""))
                suffix = f"：{reason}" if reason else ""
                console.print(
                    f"  [dim]▸[/] Scene {scene_id}: [yellow]渲染产物不可用，将重新处理[/]{suffix}"
                )

            case "scene_rendered":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [bold green]渲染成功 ✓[/]")

            case "scene_visual_evaluating":
                scene_id = data.get("scene_id", "?")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [cyan]视觉评估中[/]")

            case "scene_visual_pass":
                scene_id = data.get("scene_id", "?")
                score = data.get("score")
                suffix = f" ({score:.2f}/5)" if isinstance(score, (int, float)) else ""
                console.print(
                    f"  [dim]▸[/] Scene {scene_id}: [bold green]视觉评估通过 ✓[/]{suffix}"
                )

            case "scene_visual_fixing":
                scene_id = data.get("scene_id", "?")
                attempt = data.get("attempt", 0)
                maximum = data.get("max_attempts", 0)
                console.print(
                    f"  [dim]▸[/] Scene {scene_id}: [yellow]安排视觉修复 {attempt}/{maximum}[/]"
                )

            case "scene_visual_plan_fixing":
                scene_id = data.get("scene_id", "?")
                target = esc(data.get("target", "计划"))
                attempt = data.get("attempt", 0)
                maximum = data.get("max_attempts", 0)
                console.print(
                    f"  [dim]▸[/] Scene {scene_id}: [yellow]视觉反馈回到{target}层 "
                    f"{attempt}/{maximum}[/]"
                )

            case "scene_plan_repair_requested":
                scene_id = data.get("scene_id", "?")
                target = esc(data.get("target", "计划"))
                console.print(
                    f"  [dim]▸[/] Scene {scene_id}: [yellow]代码审查反馈回到{target}层[/]"
                )

            case "scene_waiting_for_dependency":
                scene_id = data.get("scene_id", "?")
                dependency = data.get("dependency_scene_id", "?")
                console.print(
                    f"  [dim]▸[/] Scene {scene_id}: [yellow]等待 Scene {dependency} 完成后继续[/]"
                )

            case "boundary_visual_pass":
                score = data.get("score")
                suffix = f" ({score:.2f}/5)" if isinstance(score, (int, float)) else ""
                console.print(f"  [dim]▸[/] [green]场景边界视觉审查通过 ✓[/]{suffix}")

            case "boundary_visual_warning":
                target = esc(data.get("target", "unknown"))
                console.print(f"  [dim]▸[/] [yellow]场景边界视觉提示，路由: {target}[/]")

            case "boundary_visual_unknown":
                reason = esc(data.get("reason", "视觉端点不可用"))
                console.print(f"  [dim]▸[/] [yellow]场景边界视觉结果 unknown: {reason}[/]")

            case "scene_visual_warning":
                scene_id = data.get("scene_id", "?")
                reason = esc(data.get("reason", "视觉问题已记录"))
                console.print(f"  [dim]▸[/] Scene {scene_id}: [yellow]视觉提示: {reason}[/]")

            case "scene_visual_unknown":
                scene_id = data.get("scene_id", "?")
                reason = esc(data.get("reason", "视觉端点不可用"))
                console.print(
                    f"  [dim]▸[/] Scene {scene_id}: [yellow]视觉结果 unknown，继续流水线: "
                    f"{reason}[/]"
                )

            case "scene_failed":
                scene_id = data.get("scene_id", "?")
                category = {
                    "planning": "计划",
                    "continuity": "连续性",
                    "coding": "编码",
                    "review": "代码审查",
                    "render": "渲染",
                    "infrastructure": "环境",
                    "llm": "模型",
                    "system": "系统",
                }.get(data.get("category", ""), "流水线")
                console.print(f"  [dim]▸[/] Scene {scene_id}: [bold red]{category}失败 ✗[/]")

            case "scene_fixing":
                scene_id = data.get("scene_id", "?")
                attempt = data.get("attempt", 0)
                console.print(f"  [dim]▸[/] Scene {scene_id}: [yellow]修复尝试 {attempt}[/]")

            case "scene_give_up":
                scene_id = data.get("scene_id", "?")
                reason = esc(data.get("reason", "已放弃"))
                console.print(f"  [dim]▸[/] Scene {scene_id}: [bold red]已放弃[/] {reason}")

            case "partial_output_blocked":
                console.print(
                    f"[bold red]部分场景未完成，已阻止生成不完整视频: {data.get('incomplete', [])}[/]"
                )

            case "dry_run_complete":
                console.print(
                    f"[bold green]Dry-run 完成[/]，文件位于 {esc(data.get('run_dir', ''))}"
                )

            case "merge_complete":
                path = data.get("path", "")
                size_mb = data.get("size_mb", 0)
                partial = " [yellow](部分输出)[/]" if data.get("partial") else ""
                console.print(f"\n  [bold]输出:[/] {esc(path)} [dim]({size_mb:.1f} MB)[/]{partial}")

            case "final_visual_complete":
                score = data.get("score")
                suffix = f"{score:.2f}/5" if isinstance(score, (int, float)) else "已生成"
                console.print(f"  [green]成片视觉报告:[/] {suffix}")

            case "final_visual_unknown":
                console.print(
                    f"  [yellow]成片视觉报告不可用:[/] {esc(data.get('reason', 'unknown'))}"
                )

    @staticmethod
    def _show_completion(output_path) -> None:
        """显示完成信息"""
        if output_path is None:
            return  # 流水线失败, orchestrator 已输出错误
        console.print()
        console.print(Rule("[bold green]完成[/]", style="green"))
        size_mb = output_path.stat().st_size / (1024 * 1024)
        console.print(
            Panel(
                f"[bold]{output_path}[/]\n[dim]{size_mb:.1f} MB[/]",
                title="[bold green]✓ 最终视频[/]",
                border_style="green",
            )
        )
