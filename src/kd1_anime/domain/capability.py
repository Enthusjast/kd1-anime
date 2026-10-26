"""Scene execution capabilities, independent of model agents."""

from typing import Literal

from pydantic import ConfigDict, Field

from kd1_anime.models import ValidatedModel as BaseModel

SceneParent = Literal["Scene", "ThreeDScene", "MovingCameraScene"]
Renderer = Literal["cairo", "opengl"]


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
