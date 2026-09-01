import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kd1_anime.agents.planner import ScenePlan
from kd1_anime.config import settings
from kd1_anime.eval.evaluator import Evaluator
from kd1_anime.eval.metrics import EvalMetric
from kd1_anime.eval.visual_eval import (
    FrameSample,
    VisualAnalysisResult,
    VisualEvaluator,
)
from kd1_anime.rendering import RenderProfile, SceneArtifact, VideoMetadata, sha256_file
from kd1_anime.run_store import RunManifest, StoredSceneState, sha256_text, write_manifest


def visual_payload() -> str:
    dimension = {"score": 4, "comprehensive_evaluation": "清晰"}
    return json.dumps(
        {
            "overall_analysis": "整体良好",
            "evaluation": {
                "mathematical_accuracy": dimension,
                "visual_relevance": dimension,
                "visual_quality": dimension,
                "visual_consistency": dimension,
                "element_layout": dimension,
            },
        }
    )


def test_visual_evaluator_sends_all_frames_with_correct_mime(tmp_path):
    png = tmp_path / "frame.png"
    jpg = tmp_path / "frame.jpg"
    png.write_bytes(b"png")
    jpg.write_bytes(b"jpeg")
    calls = []

    class FakeAgent:
        def call_llm(self, **kwargs):
            calls.append(kwargs)
            return visual_payload()

    evaluator = VisualEvaluator("vision-model")
    evaluator._agent = FakeAgent()

    result = evaluator.evaluate_video_frames([png, jpg], "勾股定理")

    assert result.overall_analysis == "整体良好"
    content = calls[0]["messages"][1]["content"]
    assert len(content) == 5
    assert content[2]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[4]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert calls[0]["json_mode"] is True
    assert calls[0]["stream"] is False


def test_visual_response_rejects_unknown_fields():
    payload = json.loads(visual_payload())
    payload["unexpected"] = "not allowed"

    with pytest.raises(ValidationError):
        VisualEvaluator._parse_response(json.dumps(payload))


@pytest.mark.parametrize("invalid_score", [True, "4"])
def test_visual_response_rejects_non_integer_scores(invalid_score):
    payload = json.loads(visual_payload())
    payload["evaluation"]["visual_quality"]["score"] = invalid_score

    with pytest.raises(ValidationError):
        VisualEvaluator._parse_response(json.dumps(payload))


def test_visual_evaluator_rejects_excessive_frame_count(tmp_path):
    frames = []
    for index in range(9):
        path = tmp_path / f"frame_{index}.jpg"
        path.write_bytes(b"jpeg")
        frames.append(path)

    with pytest.raises(ValueError, match="关键帧数量"):
        VisualEvaluator("vision-model").evaluate_video_frames(frames)


def test_visual_evaluator_rejects_oversized_image(tmp_path):
    image = tmp_path / "large.jpg"
    image.write_bytes(b"x" * (2 * 1024 * 1024 + 1))

    with pytest.raises(ValueError, match="too large"):
        VisualEvaluator.encode_image(image)


def test_visual_evaluator_rejects_issue_referencing_missing_frame(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg")
    payload = json.loads(visual_payload())
    payload["issues"] = [
        {
            "category": "layout",
            "severity": "major",
            "frame_ids": ["F02"],
            "evidence": "元素重叠",
            "recommendation": "调整布局",
        }
    ]

    class FakeAgent:
        def call_llm(self, **kwargs):
            return json.dumps(payload)

    evaluator = VisualEvaluator("vision-model")
    evaluator._agent = FakeAgent()

    with pytest.raises(ValueError, match="不存在的关键帧"):
        evaluator.evaluate_video_frames([image])


def test_visual_evaluator_rejects_issue_referencing_missing_boundary(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    payload = json.loads(visual_payload())
    payload["issues"] = [
        {
            "category": "consistency",
            "severity": "major",
            "repair_target": "continuity",
            "frame_ids": ["F01"],
            "boundary_ids": ["B02"],
            "evidence": "边界状态突变",
            "recommendation": "保持边界对象",
        }
    ]

    class FakeAgent:
        def call_llm(self, **kwargs):
            return json.dumps(payload)

    evaluator = VisualEvaluator("vision-model")
    evaluator._agent = FakeAgent()
    samples = [
        FrameSample(
            frame_id="F01",
            path=first,
            boundary_id="B01",
            role="boundary_end",
        ),
        FrameSample(
            frame_id="F02",
            path=second,
            boundary_id="B01",
            role="boundary_start",
        ),
    ]

    with pytest.raises(ValueError, match="不存在的场景边界"):
        evaluator.evaluate_video_frames(samples)


def test_visual_evaluator_accepts_boundary_issue_with_valid_references(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    payload = json.loads(visual_payload())
    payload["issues"] = [
        {
            "category": "consistency",
            "severity": "minor",
            "repair_target": "continuity",
            "confidence": 0.8,
            "frame_ids": ["F01", "F02"],
            "boundary_ids": ["B01"],
            "evidence": "边界略有变化",
            "recommendation": "保持对象锚点",
        }
    ]

    class FakeAgent:
        def call_llm(self, **kwargs):
            return json.dumps(payload)

    evaluator = VisualEvaluator("vision-model")
    evaluator._agent = FakeAgent()
    samples = [
        FrameSample(frame_id="F01", path=first, boundary_id="B01", role="boundary_end"),
        FrameSample(frame_id="F02", path=second, boundary_id="B01", role="boundary_start"),
    ]

    result = evaluator.evaluate_video_frames(samples)

    assert result.issues[0].repair_target == "continuity"
    assert result.issues[0].boundary_ids == ["B01"]


def test_extract_boundary_samples_records_real_adjacent_scene_ids(monkeypatch, tmp_path):
    first = tmp_path / "scene_1.mp4"
    second = tmp_path / "scene_2.mp4"
    third = tmp_path / "scene_3.mp4"
    for path in (first, second, third):
        path.write_bytes(b"video")

    def fake_run(command, **kwargs):
        output = Path(command[-1])
        output.write_bytes(b"frame")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("kd1_anime.eval.evaluator.shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("kd1_anime.eval.evaluator.subprocess.run", fake_run)

    samples = Evaluator.extract_boundary_samples(
        [(1, first), (2, second), (3, third)],
        tmp_path / "frames",
        max_boundaries=2,
    )

    assert [(sample.scene_id, sample.boundary_id, sample.role) for sample in samples] == [
        (1, "B01", "boundary_end"),
        (2, "B01", "boundary_start"),
        (2, "B02", "boundary_end"),
        (3, "B02", "boundary_start"),
    ]


def test_extract_boundary_samples_skips_missing_scene_boundaries(monkeypatch, tmp_path):
    first = tmp_path / "scene_1.mp4"
    third = tmp_path / "scene_3.mp4"
    first.write_bytes(b"video")
    third.write_bytes(b"video")

    def fake_run(command, **kwargs):
        output = Path(command[-1])
        output.write_bytes(b"frame")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("kd1_anime.eval.evaluator.shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("kd1_anime.eval.evaluator.subprocess.run", fake_run)

    samples = Evaluator.extract_boundary_samples(
        [(3, third), (1, first)],
        tmp_path / "frames",
        max_boundaries=1,
    )

    assert samples == []


def test_visual_failure_is_recorded_as_unknown_not_fake_score(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "output_final.mp4").write_bytes(b"video")
    evaluator = Evaluator(enable_visual_eval=False, output_dir=tmp_path / "reports")

    class BrokenVisualEvaluator:
        def evaluate_frames(self, frame_paths, description):
            raise RuntimeError("vision endpoint unavailable")

    evaluator.visual_evaluator = BrokenVisualEvaluator()
    monkeypatch.setattr(
        evaluator,
        "extract_video_frames",
        lambda video, output: [tmp_path / "frame.png"],
    )

    result = evaluator.evaluate_run(
        "external-run",
        run_dir,
        enable_visual=True,
    )

    assert result.overall_score is None
    assert result.scores == []
    assert "vision endpoint unavailable" in result.errors["visual"]


def test_evaluate_run_rejects_untrusted_manifest_video(monkeypatch, tmp_path):
    run_id = "20260728-120000-1234abcd"
    workspace = tmp_path / "workspace"
    run_dir = workspace / "runs" / run_id
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)
    manifest = RunManifest(
        run_id=run_id,
        user_prompt="demo",
        output_path=str(tmp_path / "configured.mp4"),
        final_video=str(tmp_path / "untrusted.mp4"),
    )
    write_manifest(run_dir / "manifest.json", manifest)

    result = Evaluator(enable_visual_eval=True, output_dir=tmp_path / "reports").evaluate_run(
        run_id,
        run_dir,
        enable_visual=True,
    )

    assert "路径不在允许范围内" in result.errors["visual"]


def test_evaluate_run_rejects_scene_code_changed_after_checkpoint(monkeypatch, tmp_path):
    run_id = "20260728-120000-1234abcd"
    workspace = tmp_path / "workspace"
    run_dir = workspace / "runs" / run_id
    code_file = run_dir / "scenes" / "scene_1.py"
    code_file.parent.mkdir(parents=True)
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    code_file.write_text(code, encoding="utf-8")
    plan = ScenePlan(
        scene_id=1,
        title="demo",
        duration_seconds=1,
        purpose="test",
        math_concept="circle",
        visual_design="fixed",
        camera_movement="fixed",
        visual_flow=["show"],
        key_moments=["pause"],
        computation="radius=1",
    )
    manifest = RunManifest(
        run_id=run_id,
        user_prompt="demo",
        output_path=str((run_dir / "output.mp4").resolve()),
        scenes={
            1: StoredSceneState(
                plan=plan,
                code_file="scenes/scene_1.py",
                code_sha256=sha256_text(code),
                class_name="Demo",
                reviewed=True,
            )
        },
    )
    write_manifest(run_dir / "manifest.json", manifest)
    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)
    code_file.write_text(code + "\n# changed\n", encoding="utf-8")

    result = Evaluator(enable_visual_eval=False, output_dir=tmp_path / "reports").evaluate_run(
        run_id,
        run_dir,
        enable_visual=False,
    )

    assert "code:scene_1" in result.errors
    assert not result.get_score(EvalMetric.CODE_SYNTAX)


def test_evaluate_run_scene_uses_exact_hashed_artifact(monkeypatch, tmp_path):
    run_id = "20260728-120000-1234abcd"
    workspace = tmp_path / "workspace"
    run_dir = workspace / "runs" / run_id
    video = run_dir / "videos" / "scene_1" / "Demo.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"scene-video")
    code = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"
    profile = RenderProfile.current()
    plan = ScenePlan(
        scene_id=1,
        title="demo",
        duration_seconds=10,
        purpose="test",
        math_concept="circle",
        visual_design="dark",
        camera_movement="fixed",
        visual_flow=["show"],
        key_moments=["pause"],
        computation="radius=1",
    )
    artifact = SceneArtifact(
        origin="rendered",
        source_run_id=run_id,
        job_id="123",
        scene_id=1,
        scene_class_name="Demo",
        code_sha256=sha256_text(code),
        render_profile_sha256=profile.digest(),
        video_path=video.relative_to(run_dir).as_posix(),
        video_sha256=sha256_file(video),
        metadata=VideoMetadata(
            size_bytes=video.stat().st_size,
            duration_seconds=1,
            width=profile.pixel_width,
            height=profile.pixel_height,
            frame_rate=profile.frame_rate,
        ),
    )
    write_manifest(
        run_dir / "manifest.json",
        RunManifest(
            run_id=run_id,
            user_prompt="demo",
            output_path=str((run_dir / "output.mp4").resolve()),
            render_profile=profile,
            scenes={
                1: StoredSceneState(
                    plan=plan,
                    code_sha256=sha256_text(code),
                    class_name="Demo",
                    artifact=artifact,
                    rendered=True,
                )
            },
        ),
    )
    frame = run_dir / "frame.jpg"
    frame.write_bytes(b"frame")
    sample = FrameSample(
        frame_id="F01",
        path=frame,
        timestamp_seconds=0.5,
        image_sha256=sha256_file(frame),
    )
    dimension = {"score": 4, "comprehensive_evaluation": "清晰"}
    analysis = VisualAnalysisResult(
        overall_analysis="良好",
        mathematical_accuracy=dimension,
        visual_relevance=dimension,
        visual_quality=dimension,
        visual_consistency=dimension,
        element_layout=dimension,
        issues=[],
    )

    class FakeVisualEvaluator:
        def evaluate_video_frames(self, samples, description, **kwargs):
            assert samples == [sample]
            return analysis

    monkeypatch.setattr(settings, "WORKSPACE_DIR", workspace)
    evaluator = Evaluator(enable_visual_eval=False, output_dir=tmp_path / "reports")
    evaluator.visual_evaluator = FakeVisualEvaluator()
    monkeypatch.setattr(evaluator, "extract_video_samples", lambda *args, **kwargs: [sample])

    result = evaluator.evaluate_run_scene(run_id, 1)

    assert result.overall_score == pytest.approx(4.0)
    assert result.metadata["video_sha256"] == artifact.video_sha256

    video.write_bytes(b"tampered")
    with pytest.raises(Exception, match="哈希不匹配"):
        evaluator.evaluate_run_scene(run_id, 1)
