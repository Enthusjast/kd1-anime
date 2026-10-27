"""生成候选代码的统一、无副作用静态检查与确定性修复入口。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kd1_anime.agents.api_linter import (
    ApiLintResult,
    lint_manim_api,
    repair_manim_api_compatibility,
)
from kd1_anime.agents.continuity import (
    extract_scene_continuity_elements,
    strip_redundant_optional_export_block,
)
from kd1_anime.agents.lifecycle import (
    LifecycleValidationResult,
    repair_initial_active_alias_lifecycle,
    repair_missing_animation_markers,
    repair_removed_active_lifecycle,
    repair_required_export_alias_lifecycle,
    repair_required_export_replacement_lifecycle,
    repair_required_export_transform_alias_lifecycle,
    validate_animation_lifecycle,
)
from kd1_anime.agents.planner import ExtractedElement, ScenePlan
from kd1_anime.agents.technical_planner import TechnicalSpec
from kd1_anime.agents.validator import CodeValidationResult, validate_manim_code
from kd1_anime.domain.diagnostics import (
    Diagnostic,
    diagnostic_from_api,
    diagnostic_from_ast,
    diagnostic_from_continuity_error,
    diagnostic_from_lifecycle,
)
from kd1_anime.run_store import atomic_write_text, sha256_text


@dataclass(frozen=True, slots=True)
class ContinuityValidationResult:
    is_valid: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateInspection:
    """一次完整检查的证据；所有检查均绑定到 code_sha256 对应的最终源码。"""

    accepted: bool
    original_code: str
    original_code_sha256: str
    code: str
    code_sha256: str
    class_name: str
    ast_result: CodeValidationResult
    api_result: ApiLintResult
    continuity_result: ContinuityValidationResult
    lifecycle_result: LifecycleValidationResult | None
    exported_elements_code: str
    exported_elements: tuple[ExtractedElement, ...]
    unknown_animations: tuple[str, ...]
    repairs: tuple[str, ...]
    diagnostics: tuple[Diagnostic, ...]

    @property
    def feedback(self) -> str:
        sections: list[str] = []
        if not self.ast_result.is_valid:
            sections.append("AST/安全校验失败:\n" + self.ast_result.feedback)
        if not self.api_result.is_valid:
            sections.append("Manim API 校验失败:\n" + "\n".join(self.api_result.errors))
        if not self.continuity_result.is_valid:
            sections.append("连续性导出合同失败:\n" + "\n".join(self.continuity_result.errors))
        if self.lifecycle_result is not None and not self.lifecycle_result.is_valid:
            sections.append("动画生命周期校验失败:\n" + "\n".join(self.lifecycle_result.errors))
        return "\n".join(sections)


class CandidateRejected(ValueError):
    """候选未通过静态接纳；保留完整检查证据供调用方决定如何重试。"""

    def __init__(self, inspection: CandidateInspection):
        self.inspection = inspection
        super().__init__(inspection.feedback)


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
    repairs: tuple[str, ...] = ()


class CandidateAcceptor:
    """集中执行代码、API、交接和生命周期验证，不持有工作流状态。"""

    def inspect_candidate(
        self,
        code: str,
        plan: ScenePlan,
        *,
        technical_spec: TechnicalSpec | None = None,
        renderer: str | None = None,
        manim_version: str | None = None,
        expected_class_name: str | None = None,
        validator: Callable[..., CodeValidationResult] | None = None,
        allow_repair: bool = True,
    ) -> CandidateInspection:
        """检查并可选地修复内存中的源码；普通候选错误作为结果返回，不写文件。"""

        original_code = code
        original_hash = sha256_text(code)
        repairs: list[str] = []
        if allow_repair:
            if technical_spec is not None:
                code, notes = repair_manim_api_compatibility(code)
                repairs.extend(notes)
                for repair in (
                    repair_required_export_alias_lifecycle,
                    repair_required_export_transform_alias_lifecycle,
                    repair_required_export_replacement_lifecycle,
                    repair_initial_active_alias_lifecycle,
                ):
                    code, notes = repair(code, technical_spec)
                    repairs.extend(notes)
            stripped = strip_redundant_optional_export_block(code, plan)
            if stripped != code:
                code = stripped
                repairs.append("移除重复的可选连续性导出定义块")

        def inspect_source(source: str) -> CandidateInspection:
            # 自定义 validator 只能补充约束，不能替代安全边界或其它检查。
            ast_result = validate_manim_code(source, renderer=renderer)
            errors = list(ast_result.errors)
            valid = ast_result.is_valid
            if validator is not None and validator is not validate_manim_code:
                custom = validator(source, renderer=renderer)
                errors.extend(error for error in custom.errors if error not in errors)
                valid = valid and custom.is_valid
                if not custom.is_valid and not custom.errors:
                    errors.append("自定义代码校验未通过")
            if expected_class_name and expected_class_name not in ast_result.scene_classes:
                errors.append(
                    f"候选代码不包含期望 Scene 类 {expected_class_name!r}，"
                    f"可用类: {ast_result.scene_classes}"
                )
                valid = False
            ast_result = CodeValidationResult(valid, errors, list(ast_result.scene_classes))
            api_result = lint_manim_api(
                source, renderer=renderer, scene_plan=plan, manim_version=manim_version
            )
            exported_code = ""
            exported_elements: tuple[ExtractedElement, ...] = ()
            try:
                exported_code, elements = extract_scene_continuity_elements(source, plan)
                exported_elements = tuple(elements)
                continuity = ContinuityValidationResult(True)
            except ValueError as exc:
                continuity = ContinuityValidationResult(False, (str(exc),))
            lifecycle = (
                validate_animation_lifecycle(source, technical_spec, renderer=renderer)
                if technical_spec is not None
                else None
            )
            code_hash = sha256_text(source)
            location = {"scene_id": plan.scene_id, "code_sha256": code_hash}
            diagnostics = [
                *diagnostic_from_ast(ast_result, **location),
                *diagnostic_from_api(api_result, **location),
            ]
            diagnostics.extend(
                diagnostic_from_continuity_error(error, **location) for error in continuity.errors
            )
            if lifecycle is not None:
                diagnostics.extend(diagnostic_from_lifecycle(lifecycle, **location))
            return CandidateInspection(
                accepted=(
                    ast_result.is_valid
                    and api_result.is_valid
                    and continuity.is_valid
                    and (lifecycle is None or lifecycle.is_valid)
                ),
                original_code=original_code,
                original_code_sha256=original_hash,
                code=source,
                code_sha256=code_hash,
                class_name=(expected_class_name or next(iter(ast_result.scene_classes), "")),
                ast_result=ast_result,
                api_result=api_result,
                continuity_result=continuity,
                lifecycle_result=lifecycle,
                exported_elements_code=exported_code,
                exported_elements=exported_elements,
                unknown_animations=tuple(lifecycle.unknown_animations) if lifecycle else (),
                repairs=tuple(repairs),
                diagnostics=tuple(diagnostics),
            )

        inspection = inspect_source(code)
        if (
            allow_repair
            and technical_spec is not None
            and inspection.ast_result.is_valid
            and inspection.continuity_result.is_valid
            and inspection.lifecycle_result is not None
            and not inspection.lifecycle_result.is_valid
        ):
            repaired, notes = repair_removed_active_lifecycle(
                code, technical_spec, inspection.lifecycle_result.errors
            )
            if repaired != code:
                code = repaired
                repairs.extend(notes)
                # 任何源码补丁都会使旧证据失效，必须重跑整条检查链。
                inspection = inspect_source(code)
            if (
                inspection.ast_result.is_valid
                and inspection.continuity_result.is_valid
                and inspection.lifecycle_result is not None
                and any("缺少语义事件标记" in error for error in inspection.lifecycle_result.errors)
            ):
                repaired, notes = repair_missing_animation_markers(code, technical_spec)
                if repaired != code:
                    code = repaired
                    repairs.extend(notes)
                    inspection = inspect_source(code)
        return inspection

    def accept_candidate(
        self,
        code: str,
        plan: ScenePlan,
        *,
        technical_spec: TechnicalSpec | None = None,
        renderer: str | None = None,
        manim_version: str | None = None,
        expected_class_name: str | None = None,
        destination: Path | None = None,
        validator: Callable[..., CodeValidationResult] | None = None,
        allow_repair: bool = True,
    ) -> AcceptedCandidate:
        """接纳检查成功的源码，并可选地原子写入权限为 0600 的目标文件。"""

        inspection = self.inspect_candidate(
            code,
            plan,
            technical_spec=technical_spec,
            renderer=renderer,
            manim_version=manim_version,
            expected_class_name=expected_class_name,
            validator=validator,
            allow_repair=allow_repair,
        )
        if not inspection.accepted:
            raise CandidateRejected(inspection)
        accepted = AcceptedCandidate(
            code=inspection.code,
            class_name=inspection.class_name,
            code_sha256=inspection.code_sha256,
            exported_elements_code=inspection.exported_elements_code,
            exported_elements=inspection.exported_elements,
            unknown_animations=inspection.unknown_animations,
            validation=inspection.ast_result,
            api_warnings=inspection.api_result.warnings,
            repairs=inspection.repairs,
        )
        if destination is not None:
            atomic_write_text(destination, accepted.code, mode=0o600)
        return accepted


__all__ = [
    "AcceptedCandidate",
    "CandidateAcceptor",
    "CandidateInspection",
    "CandidateRejected",
    "ContinuityValidationResult",
]
