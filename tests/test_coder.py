"""CoderAgent 测试。

测试代码生成、代码块提取和错误处理。
"""

from unittest.mock import patch

import pytest

from kd1_anime.agents.coder import CODER_SYSTEM_PROMPT, CoderAgent
from kd1_anime.agents.planner import ContinuityBible, ScenePlan


@pytest.fixture
def sample_plan():
    """创建示例场景规划。"""
    return ScenePlan(
        scene_id=1,
        title="Test Scene",
        duration_seconds=30,
        purpose="测试场景",
        math_concept="圆形面积",
        visual_design="深灰背景，蓝色圆形",
        camera_movement="固定机位",
        visual_flow=["显示圆形", "计算面积"],
        key_moments=["面积公式出现"],
        computation="半径 r=2，面积 A=πr²=4π",
    )


@pytest.fixture
def coder_agent():
    """创建 CoderAgent 实例。"""
    return CoderAgent()


class TestCoderAgent:
    """CoderAgent 测试类。"""

    def test_extract_code_block_with_fences(self, coder_agent):
        """测试提取带围栏的代码块。"""
        response = """这是代码：

```python
from manim import *

class TestScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
```

希望对你有帮助！"""
        code = coder_agent._extract_code_block(response)
        assert "from manim import *" in code
        assert "class TestScene(Scene):" in code
        assert "希望对你有帮助" not in code

    def test_extract_code_block_without_fences(self, coder_agent):
        """测试提取不带围栏的代码块。"""
        response = """from manim import *

class TestScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))"""
        code = coder_agent._extract_code_block(response)
        assert "from manim import *" in code
        assert "class TestScene(Scene):" in code

    def test_extract_code_block_multiple_fences(self, coder_agent):
        """测试提取多个围栏中的第一个代码块。"""
        response = """第一个代码块：

```python
from manim import *
class First(Scene):
    def construct(self): pass
```

第二个代码块：

```python
class Second(Scene):
    def construct(self): pass
```"""
        code = coder_agent._extract_code_block(response)
        assert "class First(Scene):" in code
        assert "class Second(Scene):" not in code

    def test_extract_code_block_with_language_tag(self, coder_agent):
        """测试提取带语言标签的代码块。"""
        response = """```python
from manim import *
class Test(Scene):
    def construct(self): pass
```"""
        code = coder_agent._extract_code_block(response)
        assert "from manim import *" in code

    def test_extract_code_block_empty(self, coder_agent):
        """测试提取空响应。"""
        response = ""
        code = coder_agent._extract_code_block(response)
        assert code == ""

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_generate_code_basic(self, mock_call_llm, coder_agent, sample_plan):
        """测试基本的代码生成。"""
        mock_call_llm.return_value = """```python
from manim import *

class TestScene(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        tex_template.add_to_preamble(r"\\usepackage{ctex}")
        config.tex_template = tex_template
        
        circle = Circle()
        self.play(Create(circle))
        self.wait(1)
```"""

        code = coder_agent.generate_code(sample_plan, stream=False)

        assert "from manim import *" in code
        assert "class TestScene(Scene):" in code
        assert "TexTemplate" in code
        mock_call_llm.assert_called_once()

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_generate_code_with_feedback(self, mock_call_llm, coder_agent, sample_plan):
        """测试带反馈的代码生成。"""
        mock_call_llm.return_value = """```python
from manim import *

class TestScene(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        tex_template.add_to_preamble(r"\\usepackage{ctex}")
        config.tex_template = tex_template
        
        circle = Circle(color=BLUE)
        self.play(Create(circle))
        self.wait(2)
```"""

        feedback = "请使用蓝色圆形，并增加等待时间"
        code = coder_agent.generate_code(sample_plan, feedback=feedback, stream=False)

        assert "color=BLUE" in code
        assert "self.wait(2)" in code
        # 验证 feedback 被传递到 prompt
        call_args = mock_call_llm.call_args
        assert feedback in call_args[1]["user_message"] or feedback in str(
            call_args[1].get("messages", [])
        )

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_generate_code_with_previous_code(self, mock_call_llm, coder_agent, sample_plan):
        """测试带之前代码的代码生成。"""
        mock_call_llm.return_value = """```python
from manim import *

class TestScene(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        tex_template.add_to_preamble(r"\\usepackage{ctex}")
        config.tex_template = tex_template
        
        square = Square()
        self.play(Create(square))
```"""

        previous_code = """from manim import *

class TestScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))"""

        code = coder_agent.generate_code(sample_plan, previous_code=previous_code, stream=False)

        assert "class TestScene(Scene):" in code
        # 验证 previous_code 被传递到 prompt
        call_args = mock_call_llm.call_args
        assert previous_code in call_args[1]["user_message"] or previous_code in str(
            call_args[1].get("messages", [])
        )

    def test_system_prompt_contains_requirements(self):
        """测试系统提示包含必要要求。"""
        assert "TexTemplate" in CODER_SYSTEM_PROMPT
        assert "xelatex" in CODER_SYSTEM_PROMPT
        assert "ctex" in CODER_SYSTEM_PROMPT
        assert "from manim import *" in CODER_SYSTEM_PROMPT
        assert "construct" in CODER_SYSTEM_PROMPT

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_generate_code_uses_scene_plan_info(self, mock_call_llm, coder_agent, sample_plan):
        """测试代码生成使用场景规划信息。"""
        mock_call_llm.return_value = """```python
from manim import *
class TestScene(Scene):
    def construct(self): pass
```"""

        coder_agent.generate_code(sample_plan, stream=False)

        # 验证场景规划信息被传递到 prompt
        call_args = mock_call_llm.call_args
        user_message = (
            call_args[1]["user_message"]
            if "user_message" in call_args[1]
            else str(call_args[1].get("messages", []))
        )

        assert "Test Scene" in user_message
        assert "圆形面积" in user_message
        assert "30" in user_message or "30 秒" in user_message

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_generate_code_receives_continuity_bible(self, mock_call_llm, coder_agent, sample_plan):
        mock_call_llm.return_value = """```python
from manim import *
class TestScene(Scene):
    def construct(self): pass
```"""

        coder_agent.generate_code(
            sample_plan,
            continuity_bible=ContinuityBible(background="#101010"),
            stream=False,
        )

        assert "全片连续性圣经" in mock_call_llm.call_args.kwargs["user_message"]


class TestCodeExtraction:
    """代码提取边界情况测试。"""

    def test_extract_with_markdown_before(self, coder_agent):
        """测试提取前面有 Markdown 的代码块。"""
        response = """# 代码说明

这是一个示例：

```python
from manim import *
class Test(Scene):
    def construct(self): pass
```

## 注意事项"""
        code = coder_agent._extract_code_block(response)
        assert "from manim import *" in code

    def test_extract_with_markdown_after(self, coder_agent):
        """测试提取后面有 Markdown 的代码块。"""
        response = """```python
from manim import *
class Test(Scene):
    def construct(self): pass
```

## 解释"""
        code = coder_agent._extract_code_block(response)
        assert "from manim import *" in code
        assert "## 解释" not in code

    def test_extract_with_inline_code(self, coder_agent):
        """测试提取包含内联代码的响应。"""
        response = """使用 `Circle()` 创建圆形：

```python
from manim import *
class Test(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
```"""
        code = coder_agent._extract_code_block(response)
        assert "from manim import *" in code
        assert "`Circle()`" not in code

    def test_extract_preserves_indentation(self, coder_agent):
        """测试提取保持缩进。"""
        response = """```python
from manim import *

class Test(Scene):
    def construct(self):
        if True:
            circle = Circle()
            self.play(Create(circle))
```"""
        code = coder_agent._extract_code_block(response)
        assert "        if True:" in code
        assert "            circle = Circle()" in code


class TestCoderAgentErrorHandling:
    """CoderAgent 错误处理测试。"""

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_generate_code_handles_llm_error(self, mock_call_llm, coder_agent, sample_plan):
        """测试处理 LLM 调用错误。"""
        from kd1_anime.exceptions import LLMError

        mock_call_llm.side_effect = LLMError("API 调用失败")

        with pytest.raises(LLMError):
            coder_agent.generate_code(sample_plan, stream=False)

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_generate_code_handles_empty_response(self, mock_call_llm, coder_agent, sample_plan):
        """测试处理空响应。"""
        mock_call_llm.return_value = ""

        code = coder_agent.generate_code(sample_plan, stream=False)
        # 应该返回空字符串或抛出错误
        assert isinstance(code, str)
