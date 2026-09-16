"""场景运行能力合同与 renderer/API 兼容性检查。"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kd1_anime.agents.planner import ScenePlan
from kd1_anime.agents.technical_planner import TechnicalSpec
from kd1_anime.rendering import RenderProfile

SceneParent = Literal["Scene", "ThreeDScene", "MovingCameraScene"]
Renderer = Literal["cairo", "opengl"]


_CAMERA_API_TERMS = ("camera.frame", "movingcamerascene")
_CAMERA_NEGATION_RE = re.compile(
    r"(?:禁止|严禁|不得|不能|不要|不应|无需|避免|勿|免于|do\s+not|don't|never|without|not)"
    r"(?:\s|使用|调用|要求|访问|通过|依赖|use|call|require|access|with)?",
    re.IGNORECASE,
)


def _contains_positive_camera_api_reference(text: str) -> bool:
    """只把实际要求 frame API 的文字视为 MovingCameraScene 需求。

    OpenGL 的安全指导会明确写出“禁止 camera.frame”。如果仅使用
    ``"camera.frame" in text``，这类否定说明会被误判为实际需求。这里
    保留少量上下文检查否定词，并将其与普通“镜头平移/旋转”区分开。
    """

    normalized = str(text or "")
    for term in _CAMERA_API_TERMS:
        offset = 0
        while True:
            index = normalized.lower().find(term, offset)
            if index < 0:
                break
            prefix = re.split(r"[\n。；，,、;!?]", normalized[:index])[-1]
            if not _CAMERA_NEGATION_RE.search(prefix):
                return True
            offset = index + len(term)
    return False


class CapabilityContract(BaseModel):
    """一个场景在代码生成/提交前必须满足的运行能力集合。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    scene_id: int = Field(ge=1)
    renderer: Renderer
    scene_parent: SceneParent = "Scene"
    requires_3d: bool = False
    requires_moving_camera: bool = False
    requires_tex: bool = False
    requires_numpy: bool = False
    requires_gpu: bool = False
    required_gpu_type: str = ""
    required_apis: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


class CapabilityValidationResult(BaseModel):
    """能力合同校验结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    is_valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _combined_text(scene_plan: ScenePlan, technical_spec: TechnicalSpec | None) -> str:
    values = [
        scene_plan.visual_design,
        scene_plan.camera_movement,
        scene_plan.computation,
        *scene_plan.visual_flow,
        *scene_plan.key_moments,
    ]
    if technical_spec is not None:
        values.extend(
            [
                item.constructor
                + " "
                + item.visual_role
                + " "
                + item.initial_state
                + " "
                + item.final_state
                for item in technical_spec.objects
            ]
        )
        values.extend(item.api_notes for item in technical_spec.animations)
        values.extend(technical_spec.implementation_notes)
    return " ".join(str(value) for value in values).lower()


def build_capability_contract(
    scene_plan: ScenePlan,
    technical_spec: TechnicalSpec | None = None,
    *,
    renderer: Renderer | None = None,
    render_profile: RenderProfile | None = None,
    gpu_type: str = "",
) -> CapabilityContract:
    """从计划和技术合同确定性推断场景能力。"""

    effective_renderer: Renderer = renderer or (
        technical_spec.renderer if technical_spec is not None else "cairo"
    )
    text = _combined_text(scene_plan, technical_spec)
    constructors = {
        item.constructor for item in (technical_spec.objects if technical_spec is not None else ())
    }
    requires_3d = bool(
        constructors & {"Surface", "ThreeDAxes", "Sphere", "Cube", "ParametricSurface"}
    ) or any(token in text for token in ("三维", "3d", "曲面", "切平面", "threedscene"))
    explicit_moving_camera = _contains_positive_camera_api_reference(text)
    generic_camera_motion = bool(
        re.search(r"镜头(?:推近|拉远|平移|缩放)|camera\s+(?:zoom|pan)", text)
    )
    # 三维镜头运动属于 ThreeDScene 的专用相机能力，不能因为计划写了
    # “旋转/平移/推近”就推断出 MovingCameraScene。只有明确要求
    # self.camera.frame/MovingCameraScene 时，才将其标记为不兼容的能力。
    # 非三维 Cairo 场景仍可从泛化的镜头运动描述推断 MovingCameraScene。
    requires_moving_camera = explicit_moving_camera or (generic_camera_motion and not requires_3d)
    requires_tex = bool(constructors & {"Tex", "MathTex", "MarkupText"}) or bool(
        re.search(r"mathtex|textemplate|\\(?:frac|sum|int|alpha|beta)", text)
    )
    requires_numpy = "numpy" in text or "np." in text or "Surface" in constructors
    required_api_set = {"Surface", "ThreeDAxes"} & constructors
    if requires_moving_camera:
        required_api_set.update({"self.camera.frame", "MovingCameraScene"})
    if requires_tex:
        required_api_set.update({"TexTemplate", "MathTex"})
    required_apis = tuple(sorted(required_api_set))
    parent: SceneParent = (
        "MovingCameraScene" if requires_moving_camera else "ThreeDScene" if requires_3d else "Scene"
    )
    return CapabilityContract(
        scene_id=scene_plan.scene_id,
        renderer=effective_renderer,
        scene_parent=parent,
        requires_3d=requires_3d,
        requires_moving_camera=requires_moving_camera,
        requires_tex=requires_tex,
        requires_numpy=requires_numpy,
        requires_gpu=effective_renderer == "opengl",
        required_gpu_type=gpu_type,
        required_apis=required_apis,
        evidence=tuple(
            item
            for item in (
                "TechnicalSpec 对象/动画",
                "ScenePlan 视觉与计算描述",
                f"RenderProfile.renderer={render_profile.renderer}"
                if render_profile is not None
                else "",
            )
            if item
        ),
    )


def validate_capability_contract(
    contract: CapabilityContract,
    *,
    render_profile: RenderProfile | None = None,
    gpu_type: str = "",
    code: str = "",
) -> CapabilityValidationResult:
    """校验合同和已生成代码的确定性兼容性。"""

    errors: list[str] = []
    warnings: list[str] = []
    effective_renderer = (
        render_profile.renderer if render_profile is not None else contract.renderer
    )
    if effective_renderer != contract.renderer:
        errors.append(
            f"Scene {contract.scene_id} renderer 不一致：合同为 {contract.renderer}，"
            f"当前为 {effective_renderer}"
        )
    if contract.requires_moving_camera and effective_renderer == "opengl":
        errors.append(
            "OpenGL 不支持 self.camera.frame/MovingCameraScene；应改用 ThreeDScene 相机 API 或 Cairo"
        )
    if contract.requires_3d and contract.scene_parent == "Scene":
        errors.append("三维对象必须使用 ThreeDScene，而不能使用普通 Scene")
    if contract.requires_gpu and not (gpu_type or contract.required_gpu_type):
        errors.append("OpenGL 场景缺少 SLURM_GPU_TYPE")
    if contract.requires_tex and render_profile is not None and not render_profile.xelatex_version:
        warnings.append("无法从当前环境指纹确认 XeLaTeX；正式 Smoke 将继续验证")

    if code:
        class_match = re.search(r"class\s+[A-Za-z_]\w*\s*\(([^)]*)\)", code)
        parent = class_match.group(1).strip() if class_match else ""
        if contract.requires_3d and "ThreeDScene" not in parent:
            errors.append("代码未继承 ThreeDScene，但能力合同要求三维场景")
        if contract.requires_moving_camera and "MovingCameraScene" not in parent:
            errors.append("代码未继承 MovingCameraScene，但能力合同要求 camera.frame")
        if effective_renderer == "opengl" and re.search(r"self\.camera\.frame", code):
            errors.append("OpenGL 代码包含不兼容的 self.camera.frame")
        if (
            contract.requires_3d
            and re.search(r"\bSurface\s*\(", code)
            and "ThreeDScene" not in parent
        ):
            errors.append("Surface 必须位于 ThreeDScene 中")
    return CapabilityValidationResult(
        is_valid=not errors,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def recommended_renderer(contract: CapabilityContract) -> Renderer:
    """返回当前能力合同的首选 renderer。"""

    return "cairo" if contract.requires_moving_camera else contract.renderer


__all__ = [
    "CapabilityContract",
    "CapabilityValidationResult",
    "build_capability_contract",
    "recommended_renderer",
    "validate_capability_contract",
]
