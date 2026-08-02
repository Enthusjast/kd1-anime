"""
评估指标和数据结构定义
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json


class EvalMetric(str, Enum):
    """评估指标枚举"""
    # 代码质量指标
    CODE_SYNTAX = "code_syntax"           # 语法正确性
    CODE_SECURITY = "code_security"       # 安全性
    CODE_COMPLEXITY = "code_complexity"   # 复杂度
    CODE_STYLE = "code_style"             # 代码风格
    
    # 视觉效果指标
    VISUAL_RELEVANCE = "visual_relevance"   # 视觉相关性
    VISUAL_QUALITY = "visual_quality"       # 视觉质量
    VISUAL_CONSISTENCY = "visual_consistency" # 视觉一致性
    ELEMENT_LAYOUT = "element_layout"       # 元素布局
    
    # 生成效率指标
    RENDER_TIME = "render_time"           # 渲染时间
    SUCCESS_RATE = "success_rate"         # 成功率
    RETRY_COUNT = "retry_count"           # 重试次数


class ScoreLevel(str, Enum):
    """评分等级"""
    VERY_POOR = "very_poor"      # 1分
    BELOW_AVG = "below_average"  # 2分
    ACCEPTABLE = "acceptable"    # 3分
    GOOD = "good"                # 4分
    EXCELLENT = "excellent"      # 5分


@dataclass
class QualityScore:
    """质量评分"""
    metric: EvalMetric
    score: int  # 1-5
    justification: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 1 <= self.score <= 5:
            raise ValueError(f"Score must be between 1 and 5, got {self.score}")
    
    @property
    def level(self) -> ScoreLevel:
        """获取评分等级"""
        levels = {
            1: ScoreLevel.VERY_POOR,
            2: ScoreLevel.BELOW_AVG,
            3: ScoreLevel.ACCEPTABLE,
            4: ScoreLevel.GOOD,
            5: ScoreLevel.EXCELLENT,
        }
        return levels[self.score]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "score": self.score,
            "level": self.level.value,
            "justification": self.justification,
            "details": self.details,
        }


@dataclass
class EvalResult:
    """评估结果"""
    run_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    scores: List[QualityScore] = field(default_factory=list)
    overall_score: float = 0.0
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_score(self, score: QualityScore):
        """添加评分"""
        self.scores.append(score)
        self._recalculate_overall()
    
    def _recalculate_overall(self):
        """重新计算总分 (几何平均)"""
        if not self.scores:
            self.overall_score = 0.0
            return
        
        # 使用几何平均，对低分更敏感
        product = 1.0
        for s in self.scores:
            product *= s.score
        
        self.overall_score = product ** (1.0 / len(self.scores))
    
    def get_score(self, metric: EvalMetric) -> Optional[QualityScore]:
        """获取指定指标的评分"""
        for score in self.scores:
            if score.metric == metric:
                return score
        return None
    
    def get_scores_by_category(self, category: str) -> List[QualityScore]:
        """按类别获取评分"""
        return [s for s in self.scores if s.metric.value.startswith(category)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": round(self.overall_score, 2),
            "summary": self.summary,
            "scores": [s.to_dict() for s in self.scores],
            "metadata": self.metadata,
        }
    
    def save(self, output_path: Path):
        """保存评估结果到 JSON 文件"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, input_path: Path) -> 'EvalResult':
        """从 JSON 文件加载评估结果"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = cls(
            run_id=data["run_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            overall_score=data["overall_score"],
            summary=data.get("summary", ""),
            metadata=data.get("metadata", {}),
        )
        
        for score_data in data.get("scores", []):
            score = QualityScore(
                metric=EvalMetric(score_data["metric"]),
                score=score_data["score"],
                justification=score_data.get("justification", ""),
                details=score_data.get("details", {}),
            )
            result.scores.append(score)
        
        return result


@dataclass
class ComparisonResult:
    """对比评估结果"""
    baseline_run_id: str
    current_run_id: str
    baseline_result: EvalResult
    current_result: EvalResult
    improvements: List[str] = field(default_factory=list)
    regressions: List[str] = field(default_factory=list)
    
    @property
    def score_diff(self) -> float:
        """分数差异"""
        return self.current_result.overall_score - self.baseline_result.overall_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "current_run_id": self.current_run_id,
            "baseline_score": round(self.baseline_result.overall_score, 2),
            "current_score": round(self.current_result.overall_score, 2),
            "score_diff": round(self.score_diff, 2),
            "improvements": self.improvements,
            "regressions": self.regressions,
        }
