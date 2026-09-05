"""生成候选代码的统一接纳入口。

Coder、AutoFix、精确替换、回滚和结构化编译都必须经过同一份确定性
检查链。这个模块不负责修改场景状态，只负责把“可接纳的候选”变成
带有完整证据的不可变结果；状态清理和 checkpoint 仍由 Orchestrator
负责，从而避免服务层反向持有整个状态机。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kd1_anime.agents.api_linter import lint_manim_api
from kd1_anime.agents.continuity import extract_scene_continuity_elements
from kd1_anime.agents.lifecycle import detect_unknown_animations, validate_animation_lifecycle
from kd1_anime.agents.planner import ExtractedElement, ScenePlan
from kd1_anime.agents.technical_planner import TechnicalSpec
from kd1_anime.agents.validator import CodeValidationResult, validate_manim_code
from kd1_anime.run_store import atomic_write_text, sha256_text


class CandidateRejected(ValueError):
    """候选代码未通过统一静态接纳检查。"""


@dataclass(frozen=True, slots=True)
class AcceptedCandidate:
    """统一接纳后的代码及其确定性证据。"""

    code: str
    class_name: str
    code_sha256: str
    exported_elements_code: str
    exported_elements: tuple[ExtractedElement, ...]
    unknown_animations: tuple[str, ...]
    validation: CodeValidationResult
    api_warnings: tuple[str, ...] = ()


class CandidateAcceptor:
    """集中执行代码、API、交接和生命周期验证。"""

    def inspect(
        self,
        code: str,
        plan: ScenePlan,
        *,
        technical_spec: TechnicalSpec | None = None,
        renderer: str | None = None,
        expected_class_name: str | None = None,
        validator: Callable[..., CodeValidationResult] | None = None,
    ) -> AcceptedCandidate:
        if not isinstance(code, str) or not code.strip():
            raise CandidateRejected("候选代码为空")
        validation = (validator or validate_manim_code)(code, renderer=renderer)
        if not validation.is_valid:
            raise CandidateRejected("AST/安全校验失败:\n" + validation.feedback)
        if expected_class_name and expected_class_name not in validation.scene_classes:
            raise CandidateRejected(
                f"候选代码不包含期望 Scene 类 {expected_class_name!r}，"
                f"可用类: {validation.scene_classes}"
            )
        api_result = lint_manim_api(code, renderer=renderer, scene_plan=plan)
        if not api_result.is_valid:
            raise CandidateRejected("Manim API 校验失败:\n" + "\n".join(api_result.errors))
        try:
            exported_code, exported_elements = extract_scene_continuity_elements(code, plan)
        except ValueError as exc:
            raise CandidateRejected(f"连续性导出合同失败: {exc}") from exc
        if technical_spec is not None:
            lifecycle = validate_animation_lifecycle(
                code,
                technical_spec,
                renderer=renderer,
            )
            if not lifecycle.is_valid:
                raise CandidateRejected("动画生命周期校验失败:\n" + "\n".join(lifecycle.errors))
            unknown = tuple(lifecycle.unknown_animations)
        else:
            unknown = ()
        return AcceptedCandidate(
            code=code,
            class_name=expected_class_name or validation.scene_classes[0],
            code_sha256=sha256_text(code),
            exported_elements_code=exported_code,
            exported_elements=tuple(exported_elements),
            unknown_animations=unknown,
            validation=validation,
            api_warnings=tuple(api_result.warnings),
        )

    def accept(
        self,
        code: str,
        plan: ScenePlan,
        *,
        technical_spec: TechnicalSpec | None = None,
        renderer: str | None = None,
        expected_class_name: str | None = None,
        destination: Path | None = None,
        validator: Callable[..., CodeValidationResult] | None = None,
    ) -> AcceptedCandidate:
        """验证候选并可选地原子写入场景文件。"""

        accepted = self.inspect(
            code,
            plan,
            technical_spec=technical_spec,
            renderer=renderer,
            expected_class_name=expected_class_name,
            validator=validator,
        )
        if destination is not None:
            atomic_write_text(destination, accepted.code, mode=0o600)
        return accepted

    @staticmethod
    def unknown_animation_details(
        code: str,
        technical_spec: TechnicalSpec | None,
        *,
        renderer: str | None = None,
    ) -> tuple[str, ...]:
        if technical_spec is None:
            return ()
        return tuple(detect_unknown_animations(code, technical_spec, renderer=renderer))


__all__ = ["AcceptedCandidate", "CandidateAcceptor", "CandidateRejected"]
