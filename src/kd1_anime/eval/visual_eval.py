"""
视觉效果评估器

使用 LLM 分析渲染截图，评估动画视觉质量。
参照 TheoremExplainAgent 的视觉评估设计。
"""

import os
import json
import base64
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass

from .metrics import EvalMetric, QualityScore
from .prompts import VISUAL_EVAL_PROMPT
from ..config import settings
from ..agents.base import BaseAgent


@dataclass
class VisualAnalysisResult:
    """视觉分析结果"""
    overall_analysis: str
    visual_relevance: Dict[str, Any]
    visual_quality: Dict[str, Any]
    visual_consistency: Dict[str, Any]
    element_layout: Dict[str, Any]
    raw_response: str = ""


class VisualEvaluator:
    """视觉效果评估器
    
    使用 LLM 分析渲染截图，评估动画视觉质量。
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """初始化视觉评估器
        
        Args:
            model_name: 使用的模型名称，默认使用配置中的模型
        """
        self.model_name = model_name or settings.LLM_MODEL
        self._agent = None
    
    @property
    def agent(self) -> BaseAgent:
        """懒加载 Agent"""
        if self._agent is None:
            self._agent = BaseAgent(
                model=self.model_name,
                temperature=0.0,
                max_tokens=2000,
            )
        return self._agent
    
    def encode_image(self, image_path: Union[str, Path]) -> str:
        """将图片编码为 base64
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            base64 编码的图片数据
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def evaluate_image(
        self,
        image_path: Union[str, Path],
        description: str = "Mathematical animation",
    ) -> VisualAnalysisResult:
        """评估单张图片的视觉质量
        
        Args:
            image_path: 图片文件路径
            description: 动画描述
            
        Returns:
            VisualAnalysisResult: 评估结果
        """
        image_path = Path(image_path)
        
        # 编码图片
        image_base64 = self.encode_image(image_path)
        
        # 构造提示词
        prompt = VISUAL_EVAL_PROMPT.format(description=description)
        
        # 调用 LLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        response = self.agent.chat(messages)
        
        # 解析响应
        return self._parse_response(response)
    
    def evaluate_video_frames(
        self,
        frame_paths: List[Path],
        description: str = "Mathematical animation",
    ) -> VisualAnalysisResult:
        """评估多帧图片的视觉质量
        
        Args:
            frame_paths: 帧图片路径列表
            description: 动画描述
            
        Returns:
            VisualAnalysisResult: 综合评估结果
        """
        if not frame_paths:
            raise ValueError("No frame paths provided")
        
        # 评估每帧
        all_analyses = []
        for frame_path in frame_paths:
            try:
                result = self.evaluate_image(frame_path, description)
                all_analyses.append(result)
            except Exception as e:
                print(f"Warning: Failed to evaluate {frame_path}: {e}")
                continue
        
        if not all_analyses:
            raise RuntimeError("Failed to evaluate any frames")
        
        # 合并结果
        return self._merge_analyses(all_analyses)
    
    def _parse_response(self, response: str) -> VisualAnalysisResult:
        """解析 LLM 响应"""
        try:
            # 尝试提取 JSON
            json_match = response.find('```json')
            if json_match != -1:
                json_str = response[json_match + 7:]
                json_end = json_str.find('```')
                json_str = json_str[:json_end].strip()
            else:
                # 尝试直接解析
                json_str = response.strip()
            
            data = json.loads(json_str)
            
            evaluation = data.get('evaluation', {})
            
            return VisualAnalysisResult(
                overall_analysis=data.get('overall_analysis', ''),
                visual_relevance=evaluation.get('visual_relevance', {}),
                visual_quality=evaluation.get('visual_quality', {}),
                visual_consistency=evaluation.get('visual_consistency', {}),
                element_layout=evaluation.get('element_layout', {}),
                raw_response=response,
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            # 解析失败，返回默认结果
            return VisualAnalysisResult(
                overall_analysis="Failed to parse response",
                visual_relevance={"comprehensive_evaluation": "Parse error", "score": 3},
                visual_quality={"comprehensive_evaluation": "Parse error", "score": 3},
                visual_consistency={"comprehensive_evaluation": "Parse error", "score": 3},
                element_layout={"comprehensive_evaluation": "Parse error", "score": 3},
                raw_response=response,
            )
    
    def _merge_analyses(self, analyses: List[VisualAnalysisResult]) -> VisualAnalysisResult:
        """合并多个分析结果"""
        if len(analyses) == 1:
            return analyses[0]
        
        # 计算平均分数
        def avg_score(attr: str) -> float:
            scores = [getattr(a, attr).get('score', 3) for a in analyses]
            return sum(scores) / len(scores)
        
        def merge_evaluations(attr: str) -> Dict[str, Any]:
            scores = [getattr(a, attr).get('score', 3) for a in analyses]
            evals = [getattr(a, attr).get('comprehensive_evaluation', '') for a in analyses]
            
            return {
                "score": round(sum(scores) / len(scores)),
                "comprehensive_evaluation": f"Average of {len(analyses)} frames. " + 
                                          " | ".join(filter(None, evals[:3]))  # 取前3个
            }
        
        return VisualAnalysisResult(
            overall_analysis=f"Combined analysis of {len(analyses)} frames",
            visual_relevance=merge_evaluations('visual_relevance'),
            visual_quality=merge_evaluations('visual_quality'),
            visual_consistency=merge_evaluations('visual_consistency'),
            element_layout=merge_evaluations('element_layout'),
        )
    
    def evaluate(self, image_path: Union[str, Path], description: str = "") -> List[QualityScore]:
        """评估视觉质量并返回评分
        
        Args:
            image_path: 图片路径
            description: 动画描述
            
        Returns:
            List[QualityScore]: 各维度评分列表
        """
        result = self.evaluate_image(image_path, description)
        
        scores = []
        
        # 视觉相关性
        relevance_data = result.visual_relevance
        scores.append(QualityScore(
            metric=EvalMetric.VISUAL_RELEVANCE,
            score=relevance_data.get('score', 3),
            justification=relevance_data.get('comprehensive_evaluation', ''),
            details=relevance_data,
        ))
        
        # 视觉质量
        quality_data = result.visual_quality
        scores.append(QualityScore(
            metric=EvalMetric.VISUAL_QUALITY,
            score=quality_data.get('score', 3),
            justification=quality_data.get('comprehensive_evaluation', ''),
            details=quality_data,
        ))
        
        # 视觉一致性
        consistency_data = result.visual_consistency
        scores.append(QualityScore(
            metric=EvalMetric.VISUAL_CONSISTENCY,
            score=consistency_data.get('score', 3),
            justification=consistency_data.get('comprehensive_evaluation', ''),
            details=consistency_data,
        ))
        
        # 元素布局
        layout_data = result.element_layout
        scores.append(QualityScore(
            metric=EvalMetric.ELEMENT_LAYOUT,
            score=layout_data.get('score', 3),
            justification=layout_data.get('comprehensive_evaluation', ''),
            details=layout_data,
        ))
        
        return scores
