"""Bounded generation and deterministic candidate repair, independent of workflow state."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

from kd1_anime.agents.risk import assess_scene_risk
from kd1_anime.agents.technical_planner import TechnicalSpec, compile_technical_spec
from kd1_anime.config import settings
from kd1_anime.domain.planning import (
    ContinuityBible,
    ElementManifest,
    LessonSpec,
    ScenePlan,
    TeachingGraph,
    VisualElementState,
)
from kd1_anime.exceptions import ValidationError
from kd1_anime.models import snapshot_value
from kd1_anime.run_store import sha256_text


@dataclass(frozen=True, slots=True)
class GeneratedCode:
    code: str
    class_name: str
    attempts: int
    provenance: str = "llm"


class GenerationService:
    def __init__(
        self,
        *,
        agent,
        validator: Callable,
        candidate_acceptor,
        generation_mode: str,
        manim_version: str | None = None,
        limits: dict[str, int] | None = None,
        run_config=None,
        emit: Callable | None = None,
    ):
        self.config = (
            run_config.resolve() if run_config is not None else settings.model_copy(deep=True)
        )
        self.agent = agent
        self._validate = validator
        self.candidate_acceptor = candidate_acceptor
        self.generation_mode = generation_mode
        self.manim_version = manim_version
        self.limits = dict(limits or {})
        self.emit = emit

    def _emit(self, event: str, **data):
        if self.emit is not None:
            self.emit(event, **data)

    @staticmethod
    def _supports_keyword(method, keyword: str) -> bool:
        parameters = inspect.signature(method).parameters
        return keyword in parameters or any(
            item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
        )

    @staticmethod
    def validate_technical_contract(plan, technical_spec, *, renderer=None) -> None:
        """Apply the same technical compiler before any source-producing branch."""
        if technical_spec is None:
            return
        result = compile_technical_spec(plan, technical_spec, renderer=renderer)
        if not result.is_valid:
            raise ValidationError(
                "TechnicalSpec 未通过确定性编译：\n"
                + "\n".join(f"- {error}" for error in result.errors),
                hint="请重新生成 TechnicalSpec，不要直接修改代码绕过技术合同",
            )

    def generate(
        self,
        plan: ScenePlan,
        *,
        feedback: str = "",
        previous_code: str = "",
        stream: bool = False,
        renderer: str | None = None,
        continuity_bible: ContinuityBible | None = None,
        inherited_elements_code: str = "",
        inherited_elements: list[VisualElementState] | None = None,
        elements_to_remove: list[VisualElementState] | None = None,
        element_manifest: ElementManifest | None = None,
        technical_spec: TechnicalSpec | None = None,
        rag_context: str = "",
        lesson_spec: LessonSpec | None = None,
        teaching_graph: TeachingGraph | None = None,
        attempts_used: int = 0,
        reserve_attempt: Callable[[], int] | None = None,
    ) -> GeneratedCode:
        plan = snapshot_value(plan)
        technical_spec = snapshot_value(technical_spec)
        continuity_bible = snapshot_value(continuity_bible)
        element_manifest = snapshot_value(element_manifest)
        lesson_spec = snapshot_value(lesson_spec)
        teaching_graph = snapshot_value(teaching_graph)
        agent = self.agent
        scene_risk = assess_scene_risk(plan, technical_spec)
        candidate_key = f"MAX_CODE_CANDIDATES_{scene_risk.level.upper()}"
        candidate_budget = self.limits.get(candidate_key, getattr(self.config, candidate_key))
        current_feedback = feedback
        current_previous = previous_code
        last_feedback = ""
        repeated_candidate_count = 0
        strategy_hint = ""
        temperature_override: float | None = None
        generation_mode = self.generation_mode
        max_validation_attempts = self.limits.get(
            "CODE_VALIDATION_ATTEMPTS", self.config.CODE_VALIDATION_ATTEMPTS
        )
        failed_candidate_signatures: dict[str, int] = {}
        validation_failure_attempts = 0
        # 两种模式都受验证预算约束；持续无进展时提前交还调度器处理。
        relaxed_stagnation_threshold = max(
            4, self.limits.get("MAX_STAGNANT_ATTEMPTS", self.config.MAX_STAGNANT_ATTEMPTS) + 2
        )
        self.validate_technical_contract(plan, technical_spec, renderer=renderer)
        attempt = attempts_used
        if attempt >= max_validation_attempts:
            raise ValidationError("生成代码未通过确定性校验：当前输入的验证尝试预算已耗尽")
        while attempt < max_validation_attempts:
            attempt = reserve_attempt() if reserve_attempt is not None else attempt + 1
            attempt_limit_text = str(max_validation_attempts)
            code_kwargs = {
                "feedback": current_feedback,
                "previous_code": current_previous,
                "stream": stream,
                "renderer": renderer,
            }
            if strategy_hint and self._supports_keyword(agent.generate_code, "strategy_hint"):
                code_kwargs["strategy_hint"] = strategy_hint
            if temperature_override is not None and self._supports_keyword(
                agent.generate_code, "temperature_override"
            ):
                code_kwargs["temperature_override"] = temperature_override
            if self._supports_keyword(agent.generate_code, "candidate_index"):
                # 候选策略编号受风险预算限制，两种模式的总请求也都有上限。
                code_kwargs["candidate_index"] = min(attempt, candidate_budget)
            if self._supports_keyword(agent.generate_code, "candidate_budget"):
                code_kwargs["candidate_budget"] = candidate_budget
            if self._supports_keyword(agent.generate_code, "risk_level"):
                code_kwargs["risk_level"] = scene_risk.level
            if continuity_bible is not None:
                code_kwargs["continuity_bible"] = continuity_bible
            if inherited_elements_code:
                code_kwargs["inherited_elements_code"] = inherited_elements_code
            if inherited_elements:
                code_kwargs["inherited_elements"] = inherited_elements
            if elements_to_remove:
                code_kwargs["elements_to_remove"] = elements_to_remove
            if element_manifest is not None and self._supports_keyword(
                agent.generate_code, "element_manifest"
            ):
                code_kwargs["element_manifest"] = element_manifest
            if technical_spec is not None and self._supports_keyword(
                agent.generate_code, "technical_spec"
            ):
                code_kwargs["technical_spec"] = technical_spec
            if rag_context and self._supports_keyword(agent.generate_code, "rag_context"):
                code_kwargs["rag_context"] = rag_context
            if lesson_spec is not None and self._supports_keyword(
                agent.generate_code, "lesson_spec"
            ):
                code_kwargs["lesson_spec"] = lesson_spec
            if teaching_graph is not None and self._supports_keyword(
                agent.generate_code, "teaching_graph"
            ):
                code_kwargs["teaching_graph"] = teaching_graph
            code = agent.generate_code(
                snapshot_value(plan),
                **snapshot_value(code_kwargs),
            )
            if not isinstance(code, str):
                raise TypeError("Coder did not return source text")
            inspection = self.candidate_acceptor.inspect_candidate(
                code,
                plan,
                technical_spec=technical_spec,
                renderer=renderer,
                manim_version=self.manim_version,
                validator=self._validate,
                allow_repair=True,
            )
            code = inspection.code
            if inspection.repairs:
                log = getattr(agent, "_log", None)
                if callable(log):
                    log("已应用确定性候选修复: " + "；".join(inspection.repairs), style="yellow")
            if inspection.accepted:
                return GeneratedCode(code, inspection.class_name, attempt)
            errors_by_stage: dict[str, list[str]] = {}
            for diagnostic in inspection.diagnostics:
                if diagnostic.severity == "major":
                    errors_by_stage.setdefault(diagnostic.stage, []).append(
                        diagnostic.detail or diagnostic.summary
                    )
            validation_feedback = "\n".join(errors_by_stage.get("ast", ()))
            api_errors = errors_by_stage.get("api", ())
            continuity_error = "\n".join(errors_by_stage.get("continuity", ()))
            lifecycle_error = "\n".join(errors_by_stage.get("lifecycle", ()))
            last_feedback = inspection.feedback
            validation_failure_attempts += 1
            if generation_mode == "relaxed":
                failure_signature = sha256_text(
                    "\n".join(
                        (
                            validation_feedback,
                            continuity_error,
                            lifecycle_error,
                            "\n".join(api_errors),
                        )
                    )
                )
                failed_candidate_signatures[failure_signature] = (
                    failed_candidate_signatures.get(failure_signature, 0) + 1
                )
                repeated_candidate_count = failed_candidate_signatures[failure_signature]
                if validation_failure_attempts >= relaxed_stagnation_threshold:
                    raise ValidationError(
                        "代码候选连续未通过确定性校验，需要显式降级或停止",
                        hint="由调度器决定并记录降级来源",
                    )
            # 提供详细的修复指导
            feedback_parts = [
                f"上一候选是第 {attempt}/{attempt_limit_text} 次尝试，未通过确定性校验；"
                f"现在进行第 {attempt + 1}/{attempt_limit_text} "
                "次修复。不得原样返回上一候选代码，必须针对下面的确定性错误做最小修改：\n"
                f"{validation_feedback}"
            ]
            if api_errors:
                feedback_parts.append(
                    "\nManim API 静态检查未通过，必须先修复：\n- " + "\n- ".join(api_errors)
                )
            if current_previous and code == current_previous:
                feedback_parts.append(
                    "\n上一候选代码与本次输出完全相同，说明上一次修复没有生效；"
                    "请改变实现结构，不要再次复制同一份代码。\n"
                )
            if continuity_error:
                feedback_parts.append(
                    "\n连续性导出合同校验未通过，必须修复以下问题：\n- " + continuity_error
                )
                if "不能包含已移除元素" in continuity_error:
                    feedback_parts.append(
                        "\n移除元素的修复方式：该元素仍需在 construct() 开头定义并用于"
                        " FadeOut，但它的完整定义和 element_id 标记必须移到"
                        " KD1_CONTINUITY_EXPORT_BEGIN 之前；导出区只能保留结束时"
                        "仍存在的 required=true 元素。不要把已移除元素留在 marker 内，"
                        "也不要为了删除 marker 而删除场景中的 FadeOut。\n"
                    )
                feedback_parts.append(
                    "\n导出区修复规则：只允许导出 required=true 的边界元素；"
                    "required=false 的对象只能作为场景内部对象，不能出现在 marker 内。"
                    "如果没有 required=true 元素，保留空的导出区或删除 marker；"
                    "tex_template/COLORS/FONTS 等场景上下文必须放在 marker 之前。\n"
                )
            if lifecycle_error:
                feedback_parts.append(
                    "\n动画生命周期校验未通过，必须修复以下问题：\n- " + lifecycle_error
                )
                if (
                    "动画事件标记重复" in lifecycle_error
                    or "重复使用动画事件标记" in lifecycle_error
                ):
                    feedback_parts.append(
                        "\n事件标记修复规则：introduce 事件只能对应一次 self.play。需要同时展示多个对象时，"
                        "把它们放进同一次 AnimationGroup/LaggedStart；update/camera 在同一语义阶段"
                        "可以拆成连续 self.play 并重复同一 marker，remove 事件也允许按不重叠对象分段。\n"
                    )
                if "未操作合同对象" in lifecycle_error:
                    feedback_parts.append(
                        "\n合同对象修复规则：逐字使用 TechnicalSpec 的 variable_name 作为"
                        "实际动画参数。例如对象变量是 basis_i，就必须在该事件的"
                        " Create/Write/AnimationGroup 中出现 basis_i；basis_i_arrow、"
                        "basis_i_label、basis_i_target 等别名不能替代它。\n"
                    )
                if "marker 未在 TechnicalSpec 中声明" in lifecycle_error:
                    feedback_parts.append(
                        "\n事件范围修复规则：删除 TechnicalSpec 未声明的额外 self.play，"
                        "或改用已有事件；不要自行发明 title_fade_out 等 marker。若合同"
                        "已提供 remove 事件，使用该 exact event_id。\n"
                    )
                if "缺少语义事件标记" in lifecycle_error:
                    feedback_parts.append(
                        "\n事件位置修复规则：目标对象和 copy/apply_matrix 等准备代码可以"
                        "放在 marker 之前，但 marker 必须紧邻实际 self.play；不要在 marker"
                        "之后继续写准备语句，否则校验器无法把事件绑定到动画。\n"
                    )
                if "hold 未操作合同对象" in lifecycle_error:
                    feedback_parts.append(
                        "\nhold 事件修复规则：纯停顿使用 self.wait()，不需要 marker；如果"
                        "使用 self.play，只操作已经 active 的 TechnicalSpec source，不得"
                        "引入或退出对象；hold 的 source 列表是可选状态上下文，不要求全部"
                        "对象都参与同一个强调动画。\n"
                    )
                if "重定义仍处于 active 的对象" in lifecycle_error:
                    feedback_parts.append(
                        "\n生命周期修复规则：导出区只能有一个；继承且需要继续交接的对象只能定义一次，"
                        "并且应放在唯一的 KD1_CONTINUITY_EXPORT_BEGIN/END 区内；"
                        "不要先在 marker 外复制继承代码，再在 marker 内重新定义，"
                        "也不要在动画结束位置再次重建同名对象。需要淡出的继承元素"
                        "才定义在 marker 外，并保留唯一的 FadeOut。\n"
                    )
                    if "必须导出的对象不 active" in lifecycle_error:
                        feedback_parts.append(
                            "\n导出对象激活规则：required=true 的导出变量必须在动画流程中"
                            "使用对应 semantic_action=introduce 的事件实际引入，并在结尾"
                            "保持 active；不要只定义该变量或只让带后缀的临时变量参与动画。\n"
                        )
                if "animate 作用于未 active 对象" in lifecycle_error:
                    feedback_parts.append(
                        "\n本次错误通常表示把 required 导出变量和 *_initial 临时变量混用了。"
                        "请把首次 FadeIn/Create/Write 和后续所有 animate/Transform 的 source"
                        "都改为同一个合同变量；例如直接 FadeIn(v1) 后再执行"
                        "v1.animate.scale(...)。不要用 `v1 = v1_initial` 代替 introducer；"
                        "若必须从临时对象交接到 v1，只能使用 ReplacementTransform(v1_initial, v1)，"
                        "并删除之后对 v1_initial 的动画。\n"
                    )
                    if (
                        "Transform 的 source 未 active" in lifecycle_error
                        or "update source 未 active" in lifecycle_error
                    ):
                        feedback_parts.append(
                            "\nupdate 源对象修复规则：复合 Mobject 本身只有在整体加入或由"
                            "introduce 事件引入后才是 active；单独引入子对象不等于引入 group。"
                            "请把 update 事件的 source 改为此前已 active 的合同对象，或先安排"
                            "一个 introduce 事件，不要用 Python 重绑定代替状态交接。\n"
                        )

            if generation_mode == "relaxed" and repeated_candidate_count >= 2:
                strategies = (
                    "从 construct() 重新组织对象生命周期，先保证每个合同对象只有一个 active 身份",
                    "采用最小可验证实现，删除非必要辅助对象和复杂 Transform，再逐步恢复画面细节",
                    "改用与上一候选完全不同的动画编排，但严格按 TechnicalSpec 事件顺序实现",
                )
                strategy_hint = strategies[(repeated_candidate_count - 2) % len(strategies)]
                temperature_override = min(
                    0.85,
                    max(self.config.LLM_CODE_TEMPERATURE, 0.2)
                    + 0.25 * min(repeated_candidate_count - 1, 3),
                )
                feedback_parts.append(
                    "\n这是 relaxed 模式第 "
                    f"{repeated_candidate_count} 次收到相同的无效候选。不得停止重试，"
                    "也不得复制上一版代码；请切换到完全不同的实现结构，优先采用"
                    "更小、更直接、生命周期清晰的方案，并逐条修复上面的确定性错误。"
                )

            # 如果是 TexTemplate 相关错误，提供正确示例
            if "TexTemplate" in validation_feedback or "tex_template" in validation_feedback:
                feedback_parts.append("""
\n=== 正确的 TexTemplate 配置示例 ===
你必须在 construct() 方法开头添加以下代码：

    tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
    tex_template.add_to_preamble(r"\\usepackage{ctex}")
    config.tex_template = tex_template

然后每个 Tex/MathTex 调用都必须传入 tex_template=tex_template：

    eq = MathTex(r"\\frac{a}{b}", tex_template=tex_template)
    text = Tex(r"中文文本", tex_template=tex_template)
""")

            current_feedback = "".join(feedback_parts)
            # 连续重复时不要继续把同一份大代码作为必需上下文发送；
            # 让 Coder 真正从合同和确定性错误重新组织实现。
            current_previous = (
                "" if generation_mode == "relaxed" and repeated_candidate_count >= 2 else code
            )
        raise ValidationError(
            "生成代码未通过确定性校验：\n" + (last_feedback or "未知错误"),
            hint="尝试简化场景或调整 prompt",
        )
