"""
视频拼接模块
使用 FFmpeg 将多个 Manim 渲染的 mp4 片段按序拼接为最终视频
"""

import subprocess
from pathlib import Path

from rich.console import Console

from config import settings

console = Console()


class VideoMerger:
    """FFmpeg 视频拼接器"""

    def find_scene_videos(self, scene_id: int) -> list[Path]:
        """
        查找指定场景渲染出的最终 mp4 文件

        Manim 输出结构 (slurm 脚本 --media_dir 指向 workspace/videos/scene_{id}):
          workspace/videos/scene_{id}/videos/<ClassName>/1080p60/<ClassName>.mp4   <- 最终
          workspace/videos/scene_{id}/videos/<ClassName>/1080p60/partial_movie_files/.../*.mp4  <- 中间段

        过滤掉 partial_movie_files, 避免误取中间片段.

        Args:
            scene_id: 场景 ID

        Returns:
            找到的最终 mp4 文件列表 (按名称排序)
        """
        scene_dir = settings.VIDEOS_DIR / f"scene_{scene_id}"

        if not scene_dir.exists():
            return []

        videos = [
            v for v in scene_dir.rglob("*.mp4")
            if not v.name.startswith(".")
            and "partial_movie_files" not in v.parts
        ]

        return sorted(videos, key=lambda p: p.name)

    def collect_all_videos(self, scene_ids: list[int]) -> list[Path]:
        """
        收集指定场景的视频文件,按 scene_id 排序

        Args:
            scene_ids: 实际成功渲染的场景 ID 列表

        Returns:
            排序后的 mp4 文件路径列表

        Raises:
            RuntimeError: 如果找不到任何视频文件
        """
        result: list[tuple[int, Path]] = []

        for scene_id in sorted(scene_ids):
            videos = self.find_scene_videos(scene_id)
            if videos:
                # 优先取与目录同名的最终文件; 否则取第一个
                chosen = videos[-1]
                result.append((scene_id, chosen))
                console.print(f"[dim][Merger][/] Scene {scene_id}: {chosen}")
            else:
                console.print(
                    f"[bold yellow][Merger][/] 警告: Scene {scene_id} 没有找到视频文件"
                )

        if not result:
            raise RuntimeError("没有找到任何场景的视频文件")

        return [v for _, v in result]

    def merge(
        self,
        video_paths: list[Path],
        output_path: Path | None = None,
    ) -> Path:
        """
        使用 FFmpeg 拼接视频

        优先使用 stream copy (-c copy, 无损快速); 若失败则回退到重编码,
        以应对片段编码/分辨率/fps 不一致的情况.

        Args:
            video_paths: 待拼接的 mp4 文件列表
            output_path: 输出文件路径,默认使用配置的 OUTPUT_FILE

        Returns:
            输出文件路径

        Raises:
            RuntimeError: FFmpeg 执行失败时抛出
        """
        if not video_paths:
            raise RuntimeError("没有视频文件可供拼接")

        output = Path(output_path or settings.OUTPUT_FILE)
        # 确保输出父目录存在
        output.parent.mkdir(parents=True, exist_ok=True)

        console.print(f"[bold blue][Merger][/] 开始拼接 {len(video_paths)} 个视频片段...")

        # 生成 filelist.txt (写入临时目录, 完成后清理)
        filelist_path = output.parent / "filelist.txt"
        try:
            with open(filelist_path, "w", encoding="utf-8") as f:
                for video in video_paths:
                    abs_path = video.resolve()
                    # 转义单引号: ffmpeg concat demuxer 要求路径用单引号包裹
                    safe = str(abs_path).replace("'", "'\\''")
                    f.write(f"file '{safe}'\n")
        except OSError as e:
            raise RuntimeError(f"写入 filelist.txt 失败: {e}") from e

        console.print(f"[dim][Merger][/] 文件列表: {filelist_path}")

        # 优先 stream copy
        copy_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(filelist_path),
            "-c", "copy",
            str(output),
        ]

        if self._run_ffmpeg(copy_cmd, output, "stream copy"):
            return output

        console.print("[yellow][Merger][/] stream copy 失败, 回退到重编码...[/]")

        # 回退: 重编码统一为 1080p60 yuv420p
        reencode_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(filelist_path),
            "-vf", "fps=60,scale=1920:1080,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            str(output),
        ]

        if self._run_ffmpeg(reencode_cmd, output, "re-encode"):
            return output

        raise RuntimeError("FFmpeg 拼接失败 (stream copy 与重编码均失败)")

    @staticmethod
    def _run_ffmpeg(cmd: list[str], output: Path, label: str) -> bool:
        """执行 ffmpeg 命令, 成功且产出了文件返回 True"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            console.print(f"[red][Merger][/] ffmpeg {label} 超时[/]")
            return False
        except FileNotFoundError:
            raise RuntimeError("未找到 ffmpeg 命令,请确认 FFmpeg 已安装")

        if result.returncode != 0 or not output.exists():
            console.print(f"[red][Merger][/] ffmpeg {label} 失败: {result.stderr[-500:]}", markup=False)
            return False

        size_mb = output.stat().st_size / (1024 * 1024)
        console.print(f"[bold green][Merger][/] 拼接完成 ({label}): {size_mb:.1f} MB")
        return True

    def merge_scenes(self, scene_ids: list[int]) -> Path:
        """
        一站式: 收集指定场景视频 + 拼接

        Args:
            scene_ids: 实际成功渲染的场景 ID 列表

        Returns:
            最终输出视频路径
        """
        videos = self.collect_all_videos(scene_ids)
        try:
            return self.merge(videos)
        finally:
            # 清理 filelist.txt
            filelist = Path(settings.OUTPUT_FILE).parent / "filelist.txt"
            if filelist.exists():
                try:
                    filelist.unlink()
                except OSError:
                    pass
