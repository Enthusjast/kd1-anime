"""Bounded import/frame/video smoke using the formal execution isolation policy."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

from kd1_anime.agents.risk import assess_scene_risk
from kd1_anime.cluster.execution import ExecutionPolicy
from kd1_anime.config import settings
from kd1_anime.pipeline.scene import ScenePipeline
from kd1_anime.rendering import sha256_file
from kd1_anime.run_store import restore_run_path


def _run_limited_process(
    command: list[str],
    *,
    timeout: float,
    memory_mb: int,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """运行本地预检并在超时后清理整个进程组。

    生成代码可能启动额外子进程；``subprocess.run(timeout=...)`` 只保证
    主进程被终止，容易留下 Manim/FFmpeg 子进程。Linux 的 ``prlimit`` 可
    进一步限制地址空间和 CPU 时间，缺少时仍保留进程组清理兜底。
    """

    wrapped = list(command)
    prlimit = shutil.which("prlimit")
    if prlimit:
        wrapped = [
            prlimit,
            f"--as={int(memory_mb) * 1024 * 1024}",
            f"--cpu={max(1, int(timeout) + 5)}",
            "--",
            *wrapped,
        ]
    try:
        process = subprocess.Popen(
            wrapped,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
    except OSError:
        raise
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        with suppress(OSError, ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            wrapped,
            timeout,
            output=stdout,
            stderr=stderr,
        ) from exc
    return subprocess.CompletedProcess(wrapped, process.returncode, stdout, stderr)


class SmokePipeline:
    def run(self, inputs, context) -> None:
        """可选地在本地低质量渲染当前场景，尽早发现运行时错误。"""
        state = inputs.scene
        context.checkpoint()
        source = restore_run_path(inputs.root, state.code_file)
        if sha256_file(source) != state.code_sha256:
            raise ValueError("Smoke source identity mismatch")
        if not source.is_file() or not state.class_name:
            raise RuntimeError("本地 Smoke Render 缺少场景代码或类名")
        context.checkpoint()
        with tempfile.TemporaryDirectory(
            prefix=f"kd1-smoke-{state.plan.scene_id}-", dir=inputs.root
        ) as directory:
            media_dir = Path(directory) / "media"
            media_dir.mkdir(parents=True, exist_ok=True)
            smoke_width = max(16, inputs.render_profile.pixel_width // 8 // 2 * 2)
            smoke_height = max(16, inputs.render_profile.pixel_height // 8 // 2 * 2)
            smoke_fps = min(inputs.render_profile.frame_rate, 15)
            local_smoke_mode = settings.LOCAL_SMOKE_RENDER_MODE
            if inputs.dry_run and state.unknown_animation_detected:
                local_smoke_mode = "both"
            elif settings.ADAPTIVE_SMOKE_RENDER:
                local_smoke_mode = (
                    "both"
                    if assess_scene_risk(state.plan, state.technical_spec).level == "high"
                    else "frame"
                )
            policy = ExecutionPolicy.from_spec(inputs.execution_spec)
            policy.validate()
            env = policy.environment(
                renderer=inputs.render_profile.renderer,
                platform=inputs.render_profile.opengl_platform,
            )

            def run_smoke(
                command_args: list[str], *, write_video: bool = False, module: str = "manim"
            ) -> None:
                command_args = list(command_args)
                if write_video and inputs.render_profile.renderer == "opengl":
                    command_args.insert(-2, "--write_to_movie")
                command = (
                    [sys.executable, "-m", module, *command_args]
                    if module
                    else [sys.executable, *command_args]
                )
                command = policy.wrap(
                    command,
                    inputs.root,
                    renderer=inputs.render_profile.renderer,
                    platform=inputs.render_profile.opengl_platform,
                )
                try:
                    result = _run_limited_process(
                        command,
                        timeout=settings.LOCAL_SMOKE_RENDER_TIMEOUT,
                        memory_mb=settings.LOCAL_SMOKE_RENDER_MEMORY_MB,
                        env=env,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError("本地 Smoke Render 超时") from exc
                except OSError as exc:
                    raise RuntimeError(f"本地 Smoke Render 无法启动: {exc}") from exc
                if result.returncode != 0:
                    output = (result.stderr or result.stdout or "").strip()
                    detail = output[-10000:] if output else f"退出码 {result.returncode}"
                    raise RuntimeError("本地 Smoke Render 失败:\n" + detail)

            common_args = [
                "render",
                f"--renderer={inputs.render_profile.renderer}",
                f"-q{settings.LOCAL_SMOKE_RENDER_QUALITY}",
                "--resolution",
                f"{smoke_width},{smoke_height}",
                "--fps",
                str(smoke_fps),
                "--disable_caching",
            ]
            import_check = "import importlib.util, pathlib, sys\npath=pathlib.Path(sys.argv[1])\nname=sys.argv[2]\nspec=importlib.util.spec_from_file_location('kd1_smoke_scene', path)\nmodule=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\ncandidate=getattr(module, name, None)\nif not isinstance(candidate, type):\n    raise SystemExit(1)"
            run_smoke(["-c", import_check, str(source), state.class_name], module="")
            context.checkpoint()
            if local_smoke_mode in {"frame", "both"}:
                frame_dir = media_dir / "__frame_smoke__"
                frame_dir.mkdir(parents=True, exist_ok=True)
                run_smoke(
                    [
                        *common_args,
                        "--format",
                        "png",
                        "--save_last_frame",
                        "--media_dir",
                        str(frame_dir),
                        str(source),
                        state.class_name,
                    ]
                )
                frames = [
                    path
                    for path in frame_dir.rglob("*.png")
                    if path.is_file()
                    and path.stat().st_size > 0
                    and (
                        path.stem == state.class_name
                        or path.stem.startswith(f"{state.class_name}_")
                    )
                ]
                if not frames:
                    raise RuntimeError("本地 Frame Canary 完成但没有生成最后一帧 PNG")
            if local_smoke_mode in {"video", "both"}:
                video_dir = media_dir / "__smoke__"
                video_dir.mkdir(parents=True, exist_ok=True)
                run_smoke(
                    [
                        *common_args,
                        "--from_animation_number",
                        f"0,{settings.LOCAL_SMOKE_RENDER_SHORT_ANIMATIONS}",
                        "--media_dir",
                        str(video_dir),
                        str(source),
                        state.class_name,
                    ],
                    write_video=True,
                )
                videos = [
                    path
                    for path in video_dir.rglob(f"{state.class_name}.mp4")
                    if path.is_file() and path.stat().st_size > 0
                ]
                if not videos:
                    raise RuntimeError("本地 Video Canary 完成但没有生成最终 MP4")
        return self._report(
            inputs,
            {
                "schema_version": 1,
                "scene_id": state.plan.scene_id,
                "status": "passed",
                "renderer": inputs.render_profile.renderer,
                "quality": settings.LOCAL_SMOKE_RENDER_QUALITY,
                "mode": local_smoke_mode,
                "stages": ["import", "frame", "short_video"],
                "short_animations": settings.LOCAL_SMOKE_RENDER_SHORT_ANIMATIONS,
                "resolution": [smoke_width, smoke_height],
                "frame_rate": smoke_fps,
                "container": bool(settings.SLURM_CONTAINER_IMAGE),
            },
        )

    @staticmethod
    def _report(inputs, payload):
        return ScenePipeline._report(
            inputs,
            {
                "kind": "smoke",
                "code_sha256": inputs.scene.code_sha256,
                "execution_sha256": inputs.execution_spec.digest(),
                **payload,
            },
        )
