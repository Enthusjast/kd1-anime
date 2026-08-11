import json

import pytest
from pydantic import ValidationError

from kd1_anime.eval.evaluator import Evaluator
from kd1_anime.eval.visual_eval import VisualEvaluator


def visual_payload() -> str:
    dimension = {"score": 4, "comprehensive_evaluation": "清晰"}
    return json.dumps(
        {
            "overall_analysis": "整体良好",
            "evaluation": {
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
    content = calls[0]["messages"][0]["content"]
    assert len(content) == 3
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
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
