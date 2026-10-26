"""One environment and container policy for every generated-code execution path."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from kd1_anime.config import settings

ENVIRONMENT_KEYS = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "LD_LIBRARY_PATH",
    "FONTCONFIG_FILE",
    "FONTCONFIG_PATH",
    "TEXMFHOME",
    "TEXMFLOCAL",
    "TEXMFVAR",
    "TEXMFCONFIG",
    "TEXINPUTS",
    "XDG_DATA_DIRS",
    "XDG_CACHE_HOME",
    "CONDA_PREFIX",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "SLURM_JOB_ID",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_CPUS_PER_TASK",
    "MANIM_RENDERER",
    "PYOPENGL_PLATFORM",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONNOUSERSITE",
)


@dataclass(frozen=True)
class ExecutionPolicy:
    image: str
    required: bool
    disable_network: bool
    image_sha256: str = ""

    @classmethod
    def current(cls) -> ExecutionPolicy:
        return cls(
            settings.SLURM_CONTAINER_IMAGE or "",
            settings.SLURM_REQUIRE_CONTAINER,
            settings.SLURM_CONTAINER_DISABLE_NETWORK,
        )

    @classmethod
    def from_spec(cls, spec):
        if spec is None:
            return cls.current()
        return cls(spec.image, spec.require_container, spec.disable_network, spec.image_sha256)

    def validate(self) -> None:
        if self.required and not self.image:
            raise RuntimeError("SLURM_REQUIRE_CONTAINER=true，但未配置 SLURM_CONTAINER_IMAGE")
        if self.image and not Path(self.image).expanduser().is_file():
            raise RuntimeError(f"Apptainer 镜像不存在: {self.image}")

        if self.image_sha256:
            from kd1_anime.domain.runtime import file_digest

            if file_digest(Path(self.image).expanduser()) != self.image_sha256:
                raise RuntimeError("Container image differs from the execution contract")

    def startup_check(self) -> str:
        if not self.image_sha256:
            return ""
        import shlex

        return (
            "printf '%s\\n' "
            + shlex.quote(self.image_sha256 + "  " + self.image)
            + " | sha256sum --check --status || exit 73"
        )

    @staticmethod
    def environment(source=None, *, renderer="cairo", platform="egl") -> dict[str, str]:
        source = os.environ if source is None else source
        result = {key: source[key] for key in ENVIRONMENT_KEYS if key in source}
        result["MANIM_RENDERER"] = renderer
        result["PYTHONNOUSERSITE"] = "1"
        for key in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            result[key] = "1"
        if renderer == "opengl":
            result["PYOPENGL_PLATFORM"] = platform
        else:
            result.pop("PYOPENGL_PLATFORM", None)
        return result

    def wrap(self, command: list[str], root: Path, *, renderer: str, platform: str) -> list[str]:
        self.validate()
        if not self.image:
            return list(command)
        args = ["apptainer", "exec", "--containall", "--cleanenv", "--no-home"]
        if renderer == "opengl":
            args.append("--nv")
        if self.disable_network:
            args.extend(["--net", "--network", "none"])
        environment = self.environment({}, renderer=renderer, platform=platform)
        for key, value in environment.items():
            args.extend(["--env", f"{key}={value}"])
        root = root.resolve()
        inner = list(command)
        if inner and inner[0] == sys.executable:
            inner[0] = "python"
        return [
            *args,
            "--bind",
            f"{root}:{root}",
            str(Path(self.image).expanduser().resolve()),
            *inner,
        ]

    @staticmethod
    def shell_environment_prefix() -> str:
        # Fixed names only; shell expansions remain quoted and are never evaluated as code.
        return "env -i " + " ".join(f'{key}="${{{key}-}}"' for key in ENVIRONMENT_KEYS) + " "
