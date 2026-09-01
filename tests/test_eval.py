"""
评估模块测试
"""

import json
from unittest.mock import patch

import pytest

from kd1_anime.eval.code_eval import CodeEvaluator
from kd1_anime.eval.evaluator import Evaluator
from kd1_anime.eval.metrics import (
    ComparisonResult,
    EvalMetric,
    EvalResult,
    QualityScore,
    ScoreLevel,
)


class TestQualityScore:
    """QualityScore 测试"""

    def test_valid_score(self):
        """测试有效分数"""
        score = QualityScore(metric=EvalMetric.CODE_SYNTAX, score=4, justification="Good syntax")
        assert score.score == 4
        assert score.level == ScoreLevel.GOOD

    def test_invalid_score(self):
        """测试无效分数"""
        with pytest.raises(ValueError):
            QualityScore(
                metric=EvalMetric.CODE_SYNTAX,
                score=6,  # 超出范围
                justification="Invalid",
            )

    @pytest.mark.parametrize("value", [True, 3.5, "4"])
    def test_score_requires_a_real_integer(self, value):
        with pytest.raises(TypeError):
            QualityScore(metric=EvalMetric.CODE_SYNTAX, score=value)

    def test_score_levels(self):
        """测试分数等级映射"""
        test_cases = [
            (1, ScoreLevel.VERY_POOR),
            (2, ScoreLevel.BELOW_AVG),
            (3, ScoreLevel.ACCEPTABLE),
            (4, ScoreLevel.GOOD),
            (5, ScoreLevel.EXCELLENT),
        ]

        for score_val, expected_level in test_cases:
            score = QualityScore(
                metric=EvalMetric.CODE_SYNTAX, score=score_val, justification="Test"
            )
            assert score.level == expected_level

    def test_to_dict(self):
        """测试转换为字典"""
        score = QualityScore(
            metric=EvalMetric.CODE_SYNTAX, score=4, justification="Good", details={"test": "value"}
        )

        d = score.to_dict()
        assert d["metric"] == "code_syntax"
        assert d["score"] == 4
        assert d["level"] == "good"
        assert d["details"]["test"] == "value"


class TestEvalResult:
    """EvalResult 测试"""

    def test_overall_score_calculation(self):
        """测试总分计算 (几何平均)"""
        result = EvalResult(run_id="test-run")

        # 添加评分
        result.add_score(QualityScore(metric=EvalMetric.CODE_SYNTAX, score=4, justification=""))
        result.add_score(QualityScore(metric=EvalMetric.CODE_SECURITY, score=5, justification=""))

        # 几何平均: sqrt(4 * 5) = sqrt(20) ≈ 4.47
        assert abs(result.overall_score - 4.47) < 0.01

    def test_get_score(self):
        """测试获取指定指标评分"""
        result = EvalResult(run_id="test-run")

        score = QualityScore(metric=EvalMetric.CODE_SYNTAX, score=4, justification="Test")
        result.add_score(score)

        retrieved = result.get_score(EvalMetric.CODE_SYNTAX)
        assert retrieved is not None
        assert retrieved.score == 4

        # 不存在的指标
        assert result.get_score(EvalMetric.VISUAL_QUALITY) is None

    def test_metric_average_includes_all_scene_scores(self):
        result = EvalResult(run_id="test-run")
        result.add_score(QualityScore(EvalMetric.CODE_STYLE, 2))
        result.add_score(QualityScore(EvalMetric.CODE_STYLE, 4))

        assert result.get_metric_average(EvalMetric.CODE_STYLE) == 3.0
        assert result.get_metric_average(EvalMetric.VISUAL_QUALITY) is None

    def test_get_scores_by_category(self):
        """测试按类别获取评分"""
        result = EvalResult(run_id="test-run")

        result.add_score(QualityScore(metric=EvalMetric.CODE_SYNTAX, score=4, justification=""))
        result.add_score(QualityScore(metric=EvalMetric.CODE_SECURITY, score=5, justification=""))
        result.add_score(QualityScore(metric=EvalMetric.VISUAL_QUALITY, score=3, justification=""))

        code_scores = result.get_scores_by_category("code")
        assert len(code_scores) == 2

        visual_scores = result.get_scores_by_category("visual")
        assert len(visual_scores) == 1

    def test_save_and_load(self, tmp_path):
        """测试保存和加载"""
        result = EvalResult(run_id="test-run", summary="Test summary")
        result.add_score(QualityScore(metric=EvalMetric.CODE_SYNTAX, score=4, justification="Test"))

        # 保存
        output_path = tmp_path / "eval_result.json"
        result.save(output_path)

        assert output_path.exists()
        assert output_path.stat().st_mode & 0o777 == 0o600

        # 加载
        loaded = EvalResult.load(output_path)
        assert loaded.run_id == "test-run"
        assert loaded.summary == "Test summary"
        assert len(loaded.scores) == 1
        assert loaded.scores[0].score == 4


class TestCodeEvaluator:
    """CodeEvaluator 测试"""

    def test_valid_code_analysis(self):
        """测试有效代码分析"""
        evaluator = CodeEvaluator()

        code = """
from manim import *

class TestScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
"""

        result = evaluator.analyze_code(code)

        assert result.syntax_valid is True
        assert result.syntax_errors == []
        assert result.class_count == 1
        assert result.function_count == 1
        assert result.import_count > 0

    def test_invalid_code_analysis(self):
        """测试无效代码分析"""
        evaluator = CodeEvaluator()

        code = """
def broken_function(
    print("This is broken")
"""

        result = evaluator.analyze_code(code)

        assert result.syntax_valid is False
        assert len(result.syntax_errors) > 0

    def test_security_check(self):
        """测试安全检查"""
        evaluator = CodeEvaluator()

        # 危险代码
        dangerous_code = """
import os
import subprocess

def hack():
    os.system("rm -rf /")
    subprocess.call(["ls", "-la"])
"""

        result = evaluator.analyze_code(dangerous_code)

        assert len(result.security_issues) > 0
        assert any("os" in issue for issue in result.security_issues)
        assert any("subprocess" in issue for issue in result.security_issues)

    def test_security_check_matches_generated_code_attribute_policy(self):
        evaluator = CodeEvaluator()
        code = """
from manim import *
import numpy as np
from functools import partial

class Demo(Scene):
    def construct(self):
        loader = partial(np.load, "secret.npy")
"""

        result = evaluator.analyze_code(code)

        assert any("load" in issue for issue in result.security_issues)

    @pytest.mark.parametrize("constructor", ["ImageMobject", "SVGMobject", "SceneFileWriter"])
    def test_security_check_rejects_qualified_manim_dangerous_objects(self, constructor):
        evaluator = CodeEvaluator()
        code = (
            "import manim\n"
            "class Demo(manim.Scene):\n"
            "    def construct(self):\n"
            f"        value = manim.{constructor}('/tmp/input')\n"
            "        self.add(value)\n"
        )

        result = evaluator.analyze_code(code)

        assert any(constructor in issue for issue in result.security_issues)

    def test_evaluate(self):
        """测试评估功能"""
        evaluator = CodeEvaluator()

        good_code = '''
from manim import *

class GoodScene(Scene):
    def construct(self):
        """Create a simple animation."""
        circle = Circle(color=BLUE)
        self.play(Create(circle))
        self.wait(1)
'''

        scores = evaluator.evaluate(good_code)

        assert len(scores) == 4  # 4个维度

        # 语法分数应该很高
        syntax_score = next(s for s in scores if s.metric == EvalMetric.CODE_SYNTAX)
        assert syntax_score.score == 5

        # 安全分数应该很高
        security_score = next(s for s in scores if s.metric == EvalMetric.CODE_SECURITY)
        assert security_score.score == 5

    def test_evaluate_uses_submission_validator_for_code_correctness(self):
        evaluator = CodeEvaluator()
        code = """
from manim import *

class InvalidScene(Scene):
    def construct(self):
        self.camera.frame.scale(1.1)
"""

        scores = evaluator.evaluate(code)

        syntax_score = next(s for s in scores if s.metric == EvalMetric.CODE_SYNTAX)
        assert syntax_score.score == 1
        assert evaluator.analyze_code(code).validation_errors

    def test_submission_validation_uses_explicit_renderer(self):
        evaluator = CodeEvaluator()
        code = """
from manim import *

class Shape(VMobject):
    pass

class Demo(Scene):
    def construct(self):
        self.add(Shape())
"""

        analysis = evaluator.analyze_code(code, renderer="opengl")

        assert any("自定义 mobject 子类" in error for error in analysis.validation_errors)

    def test_get_scene_complexity(self):
        """测试场景复杂度评估"""
        evaluator = CodeEvaluator()

        simple_code = """
from manim import *

class SimpleScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
"""

        result = evaluator.get_scene_complexity(simple_code)

        assert "complexity_level" in result
        assert "complexity_score" in result
        assert "factors" in result
        assert result["complexity_level"] in ["low", "medium", "high", "very_high"]


class TestEvaluator:
    """Evaluator 测试"""

    @patch("kd1_anime.eval.visual_eval.VisualEvaluator")
    def test_evaluate_code(self, mock_visual_eval):
        """测试代码评估"""
        evaluator = Evaluator(enable_visual_eval=False)

        code = """
from manim import *

class TestScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
"""

        result = evaluator.evaluate_code(code)

        assert result.run_id.startswith("code_")
        assert result.overall_score > 0
        assert len(result.scores) > 0

    @patch("kd1_anime.eval.visual_eval.VisualEvaluator")
    def test_evaluate_batch(self, mock_visual_eval, tmp_path):
        """测试批量评估"""
        evaluator = Evaluator(enable_visual_eval=False)

        # 创建模拟运行目录
        run_ids = []
        for i in range(3):
            run_id = f"test-run-{i}"
            run_dir = tmp_path / "runs" / run_id
            run_dir.mkdir(parents=True)

            # 创建模拟代码文件
            code_file = run_dir / "scene.py"
            code_file.write_text("""
from manim import *

class TestScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
""")

            # 创建模拟 manifest
            manifest = {
                "render_time_seconds": 30 + i * 10,
                "total_scenes": 3,
                "successful_scenes": 3 - i,
                "retry_count": i,
            }
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))

            run_ids.append(run_id)

        results = evaluator.evaluate_batch(run_ids, base_dir=tmp_path / "runs")

        assert len(results) == 3
        for result in results:
            assert result.overall_score > 0

    def test_evaluate_batch_preserves_failed_runs_as_unknown(self, monkeypatch, tmp_path):
        evaluator = Evaluator(enable_visual_eval=False)

        def fake_evaluate(run_id, *args, **kwargs):
            if run_id == "bad":
                raise RuntimeError("broken manifest")
            return EvalResult(run_id=run_id)

        monkeypatch.setattr(evaluator, "evaluate_run", fake_evaluate)

        results = evaluator.evaluate_batch(["first", "bad", "last"], base_dir=tmp_path)

        assert [result.run_id for result in results] == ["first", "bad", "last"]
        assert results[1].overall_score is None
        assert results[1].errors == {"evaluation": "broken manifest"}

    def test_evaluate_batch_rejects_invalid_worker_count(self, tmp_path):
        evaluator = Evaluator(enable_visual_eval=False)

        with pytest.raises(ValueError, match="max_workers"):
            evaluator.evaluate_batch([], base_dir=tmp_path, max_workers=0)

    def test_default_batch_rejects_path_traversal_run_id(self, monkeypatch, tmp_path):
        from kd1_anime.config import settings

        monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
        with pytest.raises(ValueError, match="run_id 格式无效"):
            Evaluator(enable_visual_eval=False).evaluate_batch(["../outside"])

    def test_explicit_batch_base_dir_rejects_path_traversal_run_id(self, tmp_path):
        with pytest.raises(ValueError, match="不安全路径"):
            Evaluator(enable_visual_eval=False).evaluate_batch(["../outside"], base_dir=tmp_path)

    def test_explicit_compare_base_dir_rejects_path_traversal_run_id(self, tmp_path):
        with pytest.raises(ValueError, match="不安全路径"):
            Evaluator(enable_visual_eval=False).compare_runs(
                "../outside", "current", base_dir=tmp_path
            )

    def test_default_compare_rejects_path_traversal_run_id(self, monkeypatch, tmp_path):
        from kd1_anime.config import settings

        monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
        with pytest.raises(ValueError, match="run_id 格式无效"):
            Evaluator(enable_visual_eval=False).compare_runs("../outside", "current")

    def test_compare_runs_aggregates_repeated_scene_metrics(self, monkeypatch, tmp_path):
        evaluator = Evaluator(enable_visual_eval=False)
        baseline = EvalResult(run_id="baseline")
        baseline.add_score(QualityScore(EvalMetric.CODE_STYLE, 2))
        baseline.add_score(QualityScore(EvalMetric.CODE_STYLE, 4))
        current = EvalResult(run_id="current")
        current.add_score(QualityScore(EvalMetric.CODE_STYLE, 4))
        current.add_score(QualityScore(EvalMetric.CODE_STYLE, 4))
        values = {"baseline": baseline, "current": current}
        monkeypatch.setattr(
            evaluator,
            "evaluate_run",
            lambda run_id, *args, **kwargs: values[run_id],
        )

        comparison = evaluator.compare_runs("baseline", "current", base_dir=tmp_path)

        assert comparison.improvements == ["code_style: 3.00 → 4.00"]

    def test_zero_retries_receives_best_efficiency_score(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "total_scenes": 1,
                    "successful_scenes": 1,
                    "retry_count": 0,
                }
            ),
            encoding="utf-8",
        )

        scores = Evaluator(enable_visual_eval=False)._evaluate_efficiency(run_dir)

        retry = next(score for score in scores if score.metric == EvalMetric.RETRY_COUNT)
        assert retry.score == 5

    @patch("kd1_anime.eval.visual_eval.VisualEvaluator")
    def test_generate_report(self, mock_visual_eval, tmp_path):
        """测试生成报告"""
        evaluator = Evaluator(enable_visual_eval=False, output_dir=tmp_path / "reports")

        # 创建模拟结果
        results = []
        for i in range(3):
            result = EvalResult(run_id=f"run-{i}")
            result.add_score(
                QualityScore(metric=EvalMetric.CODE_SYNTAX, score=4 + (i % 2), justification="")
            )
            results.append(result)

        report_path = evaluator.generate_report(results)

        assert report_path.exists()

        with open(report_path) as f:
            report = json.load(f)

        assert report["total_runs"] == 3
        assert "average_score" in report
        assert "summary" in report


class TestComparisonResult:
    """ComparisonResult 测试"""

    def test_score_diff(self):
        """测试分数差异"""
        baseline = EvalResult(run_id="baseline")
        baseline.add_score(QualityScore(metric=EvalMetric.CODE_SYNTAX, score=3, justification=""))

        current = EvalResult(run_id="current")
        current.add_score(QualityScore(metric=EvalMetric.CODE_SYNTAX, score=4, justification=""))

        comparison = ComparisonResult(
            baseline_run_id="baseline",
            current_run_id="current",
            baseline_result=baseline,
            current_result=current,
        )

        assert comparison.score_diff > 0  # 应该是正数

    def test_to_dict(self):
        """测试转换为字典"""
        baseline = EvalResult(run_id="baseline")
        current = EvalResult(run_id="current")

        comparison = ComparisonResult(
            baseline_run_id="baseline",
            current_run_id="current",
            baseline_result=baseline,
            current_result=current,
            improvements=["code_syntax: 3 → 4"],
            regressions=[],
        )

        d = comparison.to_dict()

        assert d["baseline_run_id"] == "baseline"
        assert d["current_run_id"] == "current"
        assert len(d["improvements"]) == 1
        assert len(d["regressions"]) == 0
        assert d["score_diff"] is None
