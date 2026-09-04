"""PlannerAgent 测试。

测试场景规划、outline生成和detail生成。
"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from kd1_anime.agents.planner import (
    CONTINUITY_BIBLE_PROMPT,
    DETAIL_PROMPT,
    OUTLINE_PROMPT,
    ContinuityBible,
    PlannerAgent,
    SceneDetail,
    SceneOutline,
    ScenePlan,
)


@pytest.fixture
def planner():
    """创建 PlannerAgent 实例。"""
    return PlannerAgent()


@pytest.fixture
def sample_outlines():
    """创建示例场景概要列表。"""
    return [
        SceneOutline(
            scene_id=1,
            title="引言",
            duration_seconds=30,
            purpose="介绍主题",
            math_concept="圆的定义",
        ),
        SceneOutline(
            scene_id=2,
            title="推导",
            duration_seconds=60,
            purpose="推导公式",
            math_concept="圆的面积公式",
        ),
    ]


class TestSceneOutline:
    """SceneOutline 模型测试。"""

    def test_valid_outline(self):
        """测试有效的场景概要。"""
        outline = SceneOutline(
            scene_id=1,
            title="Test",
            duration_seconds=30,
            purpose="Test purpose",
            math_concept="Test concept",
        )
        assert outline.scene_id == 1
        assert outline.title == "Test"
        assert outline.duration_seconds == 30

    def test_outline_requires_title(self):
        """测试场景概要需要标题。"""
        with pytest.raises(ValidationError):
            SceneOutline(
                scene_id=1,
                title="",
                duration_seconds=30,
                purpose="Test",
                math_concept="Test",
            )

    def test_outline_requires_positive_duration(self):
        """测试场景概要需要正时长。"""
        with pytest.raises(ValidationError):
            SceneOutline(
                scene_id=1,
                title="Test",
                duration_seconds=0,
                purpose="Test",
                math_concept="Test",
            )

    def test_outline_max_duration(self):
        """测试场景概要最大时长。"""
        with pytest.raises(ValidationError):
            SceneOutline(
                scene_id=1,
                title="Test",
                duration_seconds=601,
                purpose="Test",
                math_concept="Test",
            )


class TestScenePlan:
    """ScenePlan 模型测试。"""

    def test_valid_plan(self):
        """测试有效的场景规划。"""
        plan = ScenePlan(
            scene_id=1,
            title="Test",
            duration_seconds=30,
            purpose="Test purpose",
            math_concept="Test concept",
            visual_design="Test design",
            camera_movement="Fixed",
            visual_flow=["Step 1", "Step 2"],
            key_moments=["Moment 1"],
            computation="r=2, A=4π",
        )
        assert plan.scene_id == 1
        assert len(plan.visual_flow) == 2
        assert len(plan.key_moments) == 1

    def test_plan_requires_visual_flow(self):
        """测试场景规划需要视觉流程。"""
        with pytest.raises(ValidationError):
            ScenePlan(
                scene_id=1,
                title="Test",
                duration_seconds=30,
                purpose="Test",
                math_concept="Test",
                visual_design="Test",
                camera_movement="Test",
                visual_flow=[],
                key_moments=["Test"],
                computation="Test",
            )

    def test_plan_requires_key_moments(self):
        """测试场景规划需要关键时刻。"""
        with pytest.raises(ValidationError):
            ScenePlan(
                scene_id=1,
                title="Test",
                duration_seconds=30,
                purpose="Test",
                math_concept="Test",
                visual_design="Test",
                camera_movement="Test",
                visual_flow=["Test"],
                key_moments=[],
                computation="Test",
            )


class TestSceneDetail:
    """SceneDetail 模型测试。"""

    def test_valid_detail(self):
        """测试有效的场景细节。"""
        detail = SceneDetail(
            visual_design="Test design",
            camera_movement="Fixed",
            visual_flow=["Step 1"],
            key_moments=["Moment 1"],
            computation="r=2",
        )
        assert detail.visual_design == "Test design"

    def test_detail_accepts_removal_reason(self):
        detail = SceneDetail(
            visual_design="Test design",
            camera_movement="Fixed",
            visual_flow=["Step 1"],
            key_moments=["Moment 1"],
            computation="r=2",
            elements_to_remove=[{"element_id": "old_label", "reason": "本场景结束后不再保留"}],
        )

        assert detail.elements_to_remove[0].reason == "本场景结束后不再保留"

    def test_detail_requires_visual_design(self):
        """测试场景细节需要视觉设计。"""
        with pytest.raises(ValidationError):
            SceneDetail(
                visual_design="",
                camera_movement="Fixed",
                visual_flow=["Step 1"],
                key_moments=["Moment 1"],
                computation="Test",
            )


class TestPlannerAgent:
    """PlannerAgent 测试类。"""

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_plan_outline_basic(self, mock_call_llm, planner):
        """测试基本的场景概要生成。"""
        mock_call_llm.return_value = """{"items": [
            {"scene_id": 1, "title": "Introduction", "duration_seconds": 30, "purpose": "Introduce topic", "math_concept": "Circle"}
        ]}"""

        outlines = planner.plan_outline("Explain circle area formula")

        assert len(outlines) == 1
        assert outlines[0].scene_id == 1
        assert outlines[0].title == "Introduction"

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_plan_outline_normalizes_ids(self, mock_call_llm, planner):
        """测试场景 ID 规范化。"""
        mock_call_llm.return_value = """{"items": [
            {"scene_id": 5, "title": "First", "duration_seconds": 30, "purpose": "Test", "math_concept": "Test"},
            {"scene_id": 10, "title": "Second", "duration_seconds": 30, "purpose": "Test", "math_concept": "Test"}
        ]}"""

        outlines = planner.plan_outline("Test prompt")

        # ID 应该被规范化为 1, 2
        assert outlines[0].scene_id == 1
        assert outlines[1].scene_id == 2

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_plan_outline_prompt_respects_small_max_scenes(
        self, mock_call_llm, planner, monkeypatch
    ):
        from kd1_anime.config import settings

        monkeypatch.setattr(settings, "MAX_SCENES", 2)
        mock_call_llm.return_value = """{"items": [
            {"scene_id": 1, "title": "Only", "duration_seconds": 30, "purpose": "Test", "math_concept": "Test"}
        ]}"""

        planner.plan_outline("Test prompt")

        system_prompt = mock_call_llm.call_args.kwargs["system_prompt"]
        assert "本次场景数量应取最小必要数量" in system_prompt
        assert "通常不超过 2 个" in system_prompt
        assert "绝对不超过 2 个" in system_prompt

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_plan_outline_coalesces_same_canvas_sequence(self, mock_call_llm, planner):
        """同一画面中逐个出现并保持的对象应只生成一个场景。"""

        mock_call_llm.return_value = """{"items": [
            {"scene_id": 1, "title": "直线", "duration_seconds": 15, "purpose": "显示直线", "math_concept": "y=x"},
            {"scene_id": 2, "title": "抛物线", "duration_seconds": 15, "purpose": "显示抛物线", "math_concept": "y=x^2"},
            {"scene_id": 3, "title": "立方曲线", "duration_seconds": 15, "purpose": "显示立方曲线", "math_concept": "y=x^3"},
            {"scene_id": 4, "title": "平方根", "duration_seconds": 15, "purpose": "显示平方根曲线", "math_concept": "y=x^{1/2}"},
            {"scene_id": 5, "title": "反比例", "duration_seconds": 15, "purpose": "显示反比例曲线", "math_concept": "y=x^{-1}"}
        ]}"""

        prompt = """在同一坐标系中同时展示五个幂函数图像，函数逐个出现并保持显示直到视频结束。
视频总时长控制在 1 分钟以内。"""
        outlines = planner.plan_outline(prompt)

        assert len(outlines) == 1
        assert outlines[0].title == "幂函数图像整体展示"
        assert outlines[0].duration_seconds == 60
        assert "y=x" in outlines[0].math_concept
        assert "必须只输出 1 个场景概要" in mock_call_llm.call_args.kwargs["system_prompt"]

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_plan_outline_respects_explicit_scene_split(self, mock_call_llm, planner):
        """用户明确指定多场景时不能被单场景兜底规则吞并。"""

        mock_call_llm.return_value = """{"items": [
            {"scene_id": 1, "title": "第一章", "duration_seconds": 15, "purpose": "建立问题", "math_concept": "概念"},
            {"scene_id": 2, "title": "第二章", "duration_seconds": 15, "purpose": "展开推导", "math_concept": "推导"}
        ]}"""

        outlines = planner.plan_outline("请分成 2 个场景，分别展示问题和推导；两个场景使用同一坐标系。")

        assert len(outlines) == 2

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_plan_outline_rejects_too_many_scenes(self, mock_call_llm, planner):
        """测试拒绝过多场景。"""
        from kd1_anime.config import settings

        # 生成超过 MAX_SCENES 的场景
        items = [
            f'{{"scene_id": {i}, "title": "Scene {i}", "duration_seconds": 30, "purpose": "Test", "math_concept": "Test"}}'
            for i in range(1, settings.MAX_SCENES + 2)
        ]
        mock_call_llm.return_value = f'{{"items": [{", ".join(items)}]}}'

        with pytest.raises(RuntimeError, match="超过 MAX_SCENES"):
            planner.plan_outline("Test prompt")

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_plan_outline_rejects_long_prompt(self, mock_call_llm, planner):
        """测试拒绝过长的 prompt。"""
        from kd1_anime.config import settings

        long_prompt = "x" * (settings.MAX_PROMPT_CHARS + 1)

        with pytest.raises(ValueError, match="过长"):
            planner.plan_outline(long_prompt)

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_plan_detail_basic(self, mock_call_llm, planner, sample_outlines):
        """测试基本的场景细节生成。"""
        mock_call_llm.return_value = """{
            "visual_design": "深灰背景，蓝色圆形",
            "camera_movement": "固定机位",
            "visual_flow": ["显示圆形"],
            "key_moments": ["圆形出现"],
            "computation": "r=2, A=4π"
        }"""

        plan = planner.plan_detail(
            sample_outlines[0],
            sample_outlines,
            "Test prompt",
            stream=False,
        )

        assert isinstance(plan, ScenePlan)
        assert plan.scene_id == 1
        assert plan.visual_design == "深灰背景，蓝色圆形"

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_plan_continuity_bible_basic(self, mock_call_llm, planner, sample_outlines):
        mock_call_llm.return_value = """{
            "background": "#111111",
            "palette": ["蓝色表示输入"],
            "typography": "统一字体",
            "layout": "标题在顶部",
            "math_notation": "x 全片不改名",
            "persistent_elements": ["核心公式"],
            "camera_language": "固定中景",
            "narrative_arc": "问题到结论",
            "transition_rules": ["保留公式"]
        }"""

        bible = planner.plan_continuity_bible("Test prompt", sample_outlines)

        assert isinstance(bible, ContinuityBible)
        assert bible.palette == ["蓝色表示输入"]
        user_message = mock_call_llm.call_args.kwargs["user_message"]
        assert "引言" in user_message
        assert "scene_outlines" in user_message

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_plan_detail_includes_outline_info(self, mock_call_llm, planner, sample_outlines):
        """测试场景细节包含概要信息。"""
        mock_call_llm.return_value = """{
            "visual_design": "Test design",
            "camera_movement": "Fixed",
            "visual_flow": ["Step 1"],
            "key_moments": ["Moment 1"],
            "computation": "Test"
        }"""

        planner.plan_detail(
            sample_outlines[0],
            sample_outlines,
            "Test prompt",
            stream=False,
        )

        # 验证 outline 信息被传递到 prompt
        call_args = mock_call_llm.call_args
        user_message = (
            call_args[1]["user_message"]
            if "user_message" in call_args[1]
            else str(call_args[1].get("messages", []))
        )

        assert "引言" in user_message or "Introduction" in user_message
        assert "圆的定义" in user_message

    def test_outline_prompt_structure(self):
        """测试概要提示结构。"""
        assert "JSON" in OUTLINE_PROMPT
        assert "scene_id" in OUTLINE_PROMPT
        assert "title" in OUTLINE_PROMPT
        assert "duration_seconds" in OUTLINE_PROMPT
        assert "purpose" in OUTLINE_PROMPT
        assert "math_concept" in OUTLINE_PROMPT
        assert "连续编号" in OUTLINE_PROMPT
        assert "不可信数据" in OUTLINE_PROMPT

    def test_detail_prompt_structure(self):
        """测试细节提示结构。"""
        assert "visual_design" in DETAIL_PROMPT
        assert "camera_movement" in DETAIL_PROMPT
        assert "visual_flow" in DETAIL_PROMPT
        assert "key_moments" in DETAIL_PROMPT
        assert "computation" in DETAIL_PROMPT
        assert "输出字段契约" in DETAIL_PROMPT
        assert "绝不能是 {time, event, pause}" in DETAIL_PROMPT
        assert "不要使用 Markdown 代码块" in DETAIL_PROMPT
        assert "opening_state" in DETAIL_PROMPT
        assert "closing_state" in DETAIL_PROMPT
        assert "persistent_elements" in DETAIL_PROMPT

    def test_continuity_bible_prompt_structure(self):
        assert "palette" in CONTINUITY_BIBLE_PROMPT
        assert "math_notation" in CONTINUITY_BIBLE_PROMPT
        assert "transition_rules" in CONTINUITY_BIBLE_PROMPT


class TestPlannerAgentErrorHandling:
    """PlannerAgent 错误处理测试。"""

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_plan_outline_handles_llm_error(self, mock_call_llm, planner):
        """测试处理 LLM 调用错误。"""
        from kd1_anime.exceptions import LLMError

        mock_call_llm.side_effect = LLMError("API 调用失败")

        with pytest.raises(LLMError):
            planner.plan_outline("Test prompt")

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_plan_outline_handles_invalid_json(self, mock_call_llm, planner):
        """测试处理无效 JSON。"""
        mock_call_llm.return_value = "This is not JSON"

        with pytest.raises(RuntimeError):
            planner.plan_outline("Test prompt")

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_plan_detail_handles_llm_error(self, mock_call_llm, planner, sample_outlines):
        """测试处理 LLM 调用错误。"""
        from kd1_anime.exceptions import LLMError

        mock_call_llm.side_effect = LLMError("API 调用失败")

        with pytest.raises(LLMError):
            planner.plan_detail(
                sample_outlines[0],
                sample_outlines,
                "Test prompt",
                stream=False,
            )


class TestPlannerAgentIntegration:
    """PlannerAgent 集成测试。"""

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_full_planning_flow(self, mock_call_llm, planner):
        """测试完整的规划流程。"""
        # 第一次调用：生成概要
        # 第二次调用：生成细节
        mock_call_llm.side_effect = [
            """{"items": [
                {"scene_id": 1, "title": "Test", "duration_seconds": 30, "purpose": "Test", "math_concept": "Test"}
            ]}""",
            """{
                "visual_design": "Test design",
                "camera_movement": "Fixed",
                "visual_flow": ["Step 1"],
                "key_moments": ["Moment 1"],
                "computation": "Test"
            }""",
        ]

        # 生成概要
        outlines = planner.plan_outline("Test prompt")
        assert len(outlines) == 1

        # 生成细节
        plan = planner.plan_detail(
            outlines[0],
            outlines,
            "Test prompt",
            stream=False,
        )

        assert isinstance(plan, ScenePlan)
        assert plan.scene_id == 1
        assert plan.visual_design == "Test design"
