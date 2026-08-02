"""
主评估器 - 整合多维度评估功能

参照 TheoremExplainAgent 的评估系统设计，提供：
- 代码质量评估
- 视觉效果评估
- 生成效率评估
- 批量评估和对比功能
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .metrics import EvalMetric, EvalResult, QualityScore, ComparisonResult
from .code_eval import CodeEvaluator
from .visual_eval import VisualEvaluator
from ..config import settings
from ..exceptions import KD1Error


class EvaluationError(KD1Error):
    """评估相关错误"""
    pass


class Evaluator:
    """主评估器
    
    整合代码质量、视觉效果和生成效率的多维度评估。
    
    Example:
        >>> evaluator = Evaluator()
        >>> result = evaluator.evaluate_run("run-123")
        >>> print(f"Overall score: {result.overall_score}")
    """
    
    def __init__(
        self,
        enable_visual_eval: bool = True,
        visual_eval_model: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        """初始化评估器
        
        Args:
            enable_visual_eval: 是否启用视觉评估
            visual_eval_model: 视觉评估使用的模型
            output_dir: 评估结果输出目录
        """
        self.code_evaluator = CodeEvaluator()
        self.visual_evaluator = VisualEvaluator(visual_eval_model) if enable_visual_eval else None
        self.output_dir = output_dir or Path("eval_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_code(self, code: str) -> EvalResult:
        """评估代码质量
        
        Args:
            code: Python/Manim 代码
            
        Returns:
            EvalResult: 评估结果
        """
        result = EvalResult(run_id=f"code_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # 代码评估
        code_scores = self.code_evaluator.evaluate(code)
        for score in code_scores:
            result.add_score(score)
        
        result.summary = f"Code evaluation completed. Overall score: {result.overall_score:.2f}"
        return result
    
    def evaluate_visual(
        self,
        image_path: Union[str, Path],
        description: str = "",
    ) -> EvalResult:
        """评估视觉效果
        
        Args:
            image_path: 渲染截图路径
            description: 动画描述
            
        Returns:
            EvalResult: 评估结果
        """
        if not self.visual_evaluator:
            raise EvaluationError("Visual evaluation is disabled")
        
        result = EvalResult(run_id=f"visual_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # 视觉评估
        visual_scores = self.visual_evaluator.evaluate(image_path, description)
        for score in visual_scores:
            result.add_score(score)
        
        result.summary = f"Visual evaluation completed. Overall score: {result.overall_score:.2f}"
        return result
    
    def evaluate_run(
        self,
        run_id: str,
        run_dir: Optional[Path] = None,
        description: str = "",
        enable_visual: bool = True,
    ) -> EvalResult:
        """评估完整的运行结果
        
        Args:
            run_id: 运行 ID
            run_dir: 运行目录路径
            description: 动画描述
            enable_visual: 是否进行视觉评估
            
        Returns:
            EvalResult: 综合评估结果
        """
        if run_dir is None:
            run_dir = Path(f"workspace/runs/{run_id}")
        
        if not run_dir.exists():
            raise EvaluationError(f"Run directory not found: {run_dir}")
        
        result = EvalResult(
            run_id=run_id,
            metadata={
                "run_dir": str(run_dir),
                "description": description,
            }
        )
        
        # 1. 代码质量评估
        code_files = list(run_dir.glob("**/*.py"))
        if code_files:
            # 评估主场景文件
            for code_file in code_files[:3]:  # 最多评估3个文件
                try:
                    code = code_file.read_text(encoding='utf-8')
                    code_scores = self.code_evaluator.evaluate(code)
                    for score in code_scores:
                        # 添加文件信息到详情
                        score.details["file"] = str(code_file)
                        result.add_score(score)
                except Exception as e:
                    print(f"Warning: Failed to evaluate {code_file}: {e}")
        
        # 2. 视觉效果评估
        if enable_visual and self.visual_evaluator:
            # 查找渲染截图
            image_files = list(run_dir.glob("**/*.png")) + list(run_dir.glob("**/*.jpg"))
            if image_files:
                try:
                    visual_scores = self.visual_evaluator.evaluate(
                        image_files[0],
                        description,
                    )
                    for score in visual_scores:
                        result.add_score(score)
                except Exception as e:
                    print(f"Warning: Visual evaluation failed: {e}")
        
        # 3. 生成效率评估
        efficiency_scores = self._evaluate_efficiency(run_dir)
        for score in efficiency_scores:
            result.add_score(score)
        
        result.summary = self._generate_summary(result)
        return result
    
    def _evaluate_efficiency(self, run_dir: Path) -> List[QualityScore]:
        """评估生成效率"""
        scores = []
        
        # 检查 manifest 获取效率信息
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                
                # 渲染时间评估
                render_time = manifest.get("render_time_seconds", 0)
                if render_time > 0:
                    if render_time < 30:
                        time_score = 5
                    elif render_time < 60:
                        time_score = 4
                    elif render_time < 120:
                        time_score = 3
                    elif render_time < 300:
                        time_score = 2
                    else:
                        time_score = 1
                    
                    scores.append(QualityScore(
                        metric=EvalMetric.RENDER_TIME,
                        score=time_score,
                        justification=f"Render time: {render_time:.1f} seconds",
                        details={"render_time_seconds": render_time}
                    ))
                
                # 成功率评估
                total_scenes = manifest.get("total_scenes", 0)
                successful_scenes = manifest.get("successful_scenes", 0)
                
                if total_scenes > 0:
                    success_rate = successful_scenes / total_scenes
                    if success_rate >= 0.95:
                        rate_score = 5
                    elif success_rate >= 0.8:
                        rate_score = 4
                    elif success_rate >= 0.6:
                        rate_score = 3
                    elif success_rate >= 0.4:
                        rate_score = 2
                    else:
                        rate_score = 1
                    
                    scores.append(QualityScore(
                        metric=EvalMetric.SUCCESS_RATE,
                        score=rate_score,
                        justification=f"Success rate: {success_rate:.1%} ({successful_scenes}/{total_scenes})",
                        details={
                            "total_scenes": total_scenes,
                            "successful_scenes": successful_scenes,
                            "success_rate": success_rate,
                        }
                    ))
                
                # 重试次数评估
                retry_count = manifest.get("retry_count", 0)
                if retry_count > 0:
                    if retry_count <= 1:
                        retry_score = 5
                    elif retry_count <= 3:
                        retry_score = 4
                    elif retry_count <= 5:
                        retry_score = 3
                    elif retry_count <= 10:
                        retry_score = 2
                    else:
                        retry_score = 1
                    
                    scores.append(QualityScore(
                        metric=EvalMetric.RETRY_COUNT,
                        score=retry_score,
                        justification=f"Total retries: {retry_count}",
                        details={"retry_count": retry_count}
                    ))
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to parse manifest: {e}")
        
        return scores
    
    def _generate_summary(self, result: EvalResult) -> str:
        """生成评估摘要"""
        categories = {
            "code": result.get_scores_by_category("code"),
            "visual": result.get_scores_by_category("visual"),
            "render": result.get_scores_by_category("render"),
            "success": result.get_scores_by_category("success"),
            "retry": result.get_scores_by_category("retry"),
        }
        
        summary_parts = [f"Overall score: {result.overall_score:.2f}/5.00"]
        
        for category, scores in categories.items():
            if scores:
                avg = sum(s.score for s in scores) / len(scores)
                summary_parts.append(f"{category.capitalize()}: {avg:.1f}/5.0")
        
        return " | ".join(summary_parts)
    
    def evaluate_batch(
        self,
        run_ids: List[str],
        base_dir: Optional[Path] = None,
        description: str = "",
        max_workers: int = 4,
    ) -> List[EvalResult]:
        """批量评估多个运行
        
        Args:
            run_ids: 运行 ID 列表
            base_dir: 运行目录基础路径
            description: 动画描述
            max_workers: 最大并行数
            
        Returns:
            List[EvalResult]: 评估结果列表
        """
        if base_dir is None:
            base_dir = Path("workspace/runs")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for run_id in run_ids:
                run_dir = base_dir / run_id
                future = executor.submit(
                    self.evaluate_run,
                    run_id,
                    run_dir,
                    description,
                    False,  # 批量评估禁用视觉评估以提高速度
                )
                futures[future] = run_id
            
            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✓ Evaluated {run_id}: {result.overall_score:.2f}")
                except Exception as e:
                    print(f"✗ Failed to evaluate {run_id}: {e}")
        
        return results
    
    def compare_runs(
        self,
        baseline_run_id: str,
        current_run_id: str,
        base_dir: Optional[Path] = None,
    ) -> ComparisonResult:
        """对比两个运行的结果
        
        Args:
            baseline_run_id: 基准运行 ID
            current_run_id: 当前运行 ID
            base_dir: 运行目录基础路径
            
        Returns:
            ComparisonResult: 对比结果
        """
        if base_dir is None:
            base_dir = Path("workspace/runs")
        
        baseline_result = self.evaluate_run(baseline_run_id, base_dir / baseline_run_id)
        current_result = self.evaluate_run(current_run_id, base_dir / current_run_id)
        
        # 分析改进和退化
        improvements = []
        regressions = []
        
        for metric in EvalMetric:
            baseline_score = baseline_result.get_score(metric)
            current_score = current_result.get_score(metric)
            
            if baseline_score and current_score:
                diff = current_score.score - baseline_score.score
                if diff > 0:
                    improvements.append(f"{metric.value}: {baseline_score.score} → {current_score.score}")
                elif diff < 0:
                    regressions.append(f"{metric.value}: {baseline_score.score} → {current_score.score}")
        
        return ComparisonResult(
            baseline_run_id=baseline_run_id,
            current_run_id=current_run_id,
            baseline_result=baseline_result,
            current_result=current_result,
            improvements=improvements,
            regressions=regressions,
        )
    
    def generate_report(
        self,
        results: List[EvalResult],
        output_path: Optional[Path] = None,
    ) -> Path:
        """生成评估报告
        
        Args:
            results: 评估结果列表
            output_path: 输出路径
            
        Returns:
            Path: 报告文件路径
        """
        if output_path is None:
            output_path = self.output_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_runs": len(results),
            "average_score": sum(r.overall_score for r in results) / len(results) if results else 0,
            "results": [r.to_dict() for r in results],
            "summary": self._generate_batch_summary(results),
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _generate_batch_summary(self, results: List[EvalResult]) -> Dict[str, Any]:
        """生成批量评估摘要"""
        if not results:
            return {}
        
        scores = [r.overall_score for r in results]
        
        # 找出最佳和最差
        best_result = max(results, key=lambda r: r.overall_score)
        worst_result = min(results, key=lambda r: r.overall_score)
        
        return {
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "best_run": best_result.run_id,
            "worst_run": worst_result.run_id,
            "score_distribution": {
                "excellent (4-5)": sum(1 for s in scores if s >= 4),
                "good (3-4)": sum(1 for s in scores if 3 <= s < 4),
                "acceptable (2-3)": sum(1 for s in scores if 2 <= s < 3),
                "poor (1-2)": sum(1 for s in scores if s < 2),
            }
        }
