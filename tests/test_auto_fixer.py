"""AutoFixerAgent 测试。

测试错误分类、修复逻辑和边界情况。
"""

from unittest.mock import patch

import pytest

from kd1_anime.agents.auto_fixer import AUTO_FIXER_SYSTEM_PROMPT, AutoFixerAgent


@pytest.fixture
def fixer():
    """创建 AutoFixerAgent 实例。"""
    return AutoFixerAgent()


class TestErrorClassification:
    """错误分类测试。"""

    def test_classify_latex_error(self, fixer):
        """测试 LaTeX 错误分类。"""
        error_log = """
LaTeX Error: Missing $ inserted.

See the LaTeX manual or LaTeX Companion for explanation.
Type  H <return>  for immediate emergency stop.
        """
        error_type = fixer._classify_error(error_log)
        assert "LaTeX" in error_type

    def test_classify_import_error(self, fixer):
        """测试导入错误分类。"""
        error_log = """
Traceback (most recent call last):
  File "scene_1.py", line 1, in <module>
    from manim import *
ImportError: cannot import name 'ShowCreation' from 'manim'
        """
        error_type = fixer._classify_error(error_log)
        assert "导入" in error_type or "Import" in error_type

    def test_classify_name_error(self, fixer):
        """测试命名错误分类。"""
        error_log = """
Traceback (most recent call last):
  File "scene_1.py", line 5, in construct
    self.play(ShowCreation(circle))
NameError: name 'ShowCreation' is not defined
        """
        error_type = fixer._classify_error(error_log)
        assert "命名" in error_type or "Name" in error_type

    def test_classify_attribute_error(self, fixer):
        """测试属性错误分类。"""
        error_log = """
Traceback (most recent call last):
  File "scene_1.py", line 5, in construct
    circle.setColor(BLUE)
AttributeError: 'Circle' object has no attribute 'setColor'
        """
        error_type = fixer._classify_error(error_log)
        assert "属性" in error_type or "Attribute" in error_type

    def test_classify_type_error(self, fixer):
        """测试类型错误分类。"""
        error_log = """
Traceback (most recent call last):
  File "scene_1.py", line 5, in construct
    axes = Axes(-3, 3, -2, 2)
TypeError: __init__() takes 1 positional argument but 5 were given
        """
        error_type = fixer._classify_error(error_log)
        assert "参数" in error_type or "TypeError" in error_type

    def test_classify_timeout_error(self, fixer):
        """测试超时错误分类。"""
        error_log = """
#SBATCH -t 00:05:00
CANCELLED (TIMEOUT)
        """
        error_type = fixer._classify_error(error_log)
        assert "超时" in error_type or "timeout" in error_type.lower()

    def test_classify_oom_error(self, fixer):
        """测试内存不足错误分类。"""
        error_log = """
slurmstepd: error: Detected 1 oom-kill event(s)
        """
        error_type = fixer._classify_error(error_log)
        assert "内存" in error_type or "OOM" in error_type

    def test_classify_recursion_error(self, fixer):
        """测试递归错误分类。"""
        error_log = """
Traceback (most recent call last):
  File "scene_1.py", line 10, in construct
    ...
RecursionError: maximum recursion depth exceeded
        """
        error_type = fixer._classify_error(error_log)
        assert "递归" in error_type or "Recursion" in error_type

    def test_classify_font_error(self, fixer):
        """测试字体错误分类。"""
        error_log = """
PangoError: couldn't load font "Noto Sans CJK SC"
        """
        error_type = fixer._classify_error(error_log)
        assert "字体" in error_type or "font" in error_type.lower()

    def test_classify_should_render_error(self, fixer):
        """OpenGL mobject 缺少 should_render 的分类。"""
        error_log = """
Traceback (most recent call last):
  File "scene_4.py", line 63, in construct
    self.play(Create(outer_square))
  File ".../opengl_renderer.py", line 941, in update_frame
    if not mobject.should_render:
AttributeError: Polygon object has no attribute 'should_render'
        """
        error_type = fixer._classify_error(error_log)
        assert "should_render" in error_type or "OpenGL" in error_type

    def test_classify_index_error(self, fixer):
        """测试下标越界分类 (split/分组结果直接取下标)。"""
        error_log = """
Traceback (most recent call last):
  File "scene_2.py", line 92, in construct
    a2 = a_parts[1].get_center()
IndexError: list index out of range
        """
        error_type = fixer._classify_error(error_log)
        assert "IndexError" in error_type or "下标越界" in error_type

    def test_classify_unknown_error(self, fixer):
        """测试未知错误分类。"""
        error_log = """
Some random error occurred
        """
        error_type = fixer._classify_error(error_log)
        assert "未知" in error_type or "unknown" in error_type.lower()


class TestInfrastructureErrorDetection:
    """基础设施错误检测测试。"""

    def test_detect_conda_not_found(self, fixer):
        """测试检测 conda 未找到。"""
        error_log = "conda: command not found"
        assert fixer.is_infrastructure_error(error_log) is True

    def test_detect_conda_environment_not_found(self, fixer):
        """测试检测 conda 环境未找到。"""
        error_log = "could not find conda environment: manim_env"
        assert fixer.is_infrastructure_error(error_log) is True

    def test_detect_module_not_found(self, fixer):
        """测试检测 module 命令未找到。"""
        error_log = "module: command not found"
        assert fixer.is_infrastructure_error(error_log) is True

    def test_detect_apptainer_not_found(self, fixer):
        """测试检测 apptainer 未找到。"""
        error_log = "apptainer: command not found"
        assert fixer.is_infrastructure_error(error_log) is True

    def test_detect_xelatex_not_found(self, fixer):
        """测试检测 xelatex 未找到。"""
        error_log = "no such file or directory: 'xelatex'"
        assert fixer.is_infrastructure_error(error_log) is True

    def test_detect_dvisvgm_not_found(self, fixer):
        """测试检测 dvisvgm 未找到。"""
        error_log = "dvisvgm: command not found"
        assert fixer.is_infrastructure_error(error_log) is True

    def test_detect_egl_not_initialized(self, fixer):
        """测试检测 EGL 初始化失败。"""
        error_log = "egl_not_initialized"
        assert fixer.is_infrastructure_error(error_log) is True

    def test_detect_display_error(self, fixer):
        """测试检测显示连接错误。"""
        error_log = "cannot connect to display"
        assert fixer.is_infrastructure_error(error_log) is True

    def test_detect_slurm_account_error(self, fixer):
        """测试检测 Slurm 账户错误。"""
        error_log = "invalid account"
        assert fixer.is_infrastructure_error(error_log) is True

    def test_detect_slurm_partition_error(self, fixer):
        """测试检测 Slurm 分区错误。"""
        error_log = "invalid partition"
        assert fixer.is_infrastructure_error(error_log) is True

    def test_detect_slurm_qos_error(self, fixer):
        """测试检测 Slurm QOS 错误。"""
        error_log = "invalid qos"
        assert fixer.is_infrastructure_error(error_log) is True

    def test_not_infrastructure_error(self, fixer):
        """测试非基础设施错误。"""
        error_log = """
Traceback (most recent call last):
  File "scene_1.py", line 5, in construct
    circle = Circle()
AttributeError: 'NoneType' object has no attribute 'play'
        """
        assert fixer.is_infrastructure_error(error_log) is False


class TestAutoFixerFix:
    """修复逻辑测试。"""

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_fix_basic(self, mock_call_llm, fixer):
        """测试基本修复。"""
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

        original_code = """from manim import *

class TestScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(ShowCreation(circle))"""

        error_log = "NameError: name 'ShowCreation' is not defined"

        fixed_code = fixer.fix(original_code, error_log)

        assert "from manim import *" in fixed_code
        assert "class TestScene(Scene):" in fixed_code
        assert "ShowCreation" not in fixed_code

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_fix_includes_rag_context(self, mock_call_llm, fixer):
        mock_call_llm.return_value = """```python
from manim import *
class TestScene(Scene):
    def construct(self): pass
```"""

        fixer.fix(
            "from manim import *",
            "NameError: Create",
            rag_context="<reference>Use Create for standard mobjects.</reference>",
        )

        message = mock_call_llm.call_args.kwargs["user_message"]
        assert "RAG Reference Context" in message
        assert "Use Create" in message

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_fix_preserves_structure(self, mock_call_llm, fixer):
        """测试修复保持代码结构。"""
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
        
        square = Square(color=RED)
        self.play(Create(square))
        self.wait(2)
```"""

        original_code = """from manim import *

class TestScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
        self.wait(1)
        
        square = Square()
        self.play(Create(square))
        self.wait(1)"""

        error_log = "Some error"

        fixed_code = fixer.fix(original_code, error_log)

        # 验证结构保持
        assert "circle = Circle" in fixed_code
        assert "square = Square" in fixed_code
        assert "self.play(Create(circle))" in fixed_code
        assert "self.play(Create(square))" in fixed_code

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_fix_adds_tex_template(self, mock_call_llm, fixer):
        """测试修复添加 TexTemplate。"""
        mock_call_llm.return_value = """```python
from manim import *

class TestScene(Scene):
    def construct(self):
        tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")
        tex_template.add_to_preamble(r"\\usepackage{ctex}")
        config.tex_template = tex_template
        
        eq = MathTex(r"\\frac{a}{b}", tex_template=tex_template)
        self.play(Write(eq))
```"""

        original_code = """from manim import *

class TestScene(Scene):
    def construct(self):
        eq = MathTex(r"\\frac{a}{b}")
        self.play(Write(eq))"""

        error_log = "Tex/MathTex 必须使用 TexTemplate"

        fixed_code = fixer.fix(original_code, error_log)

        assert "TexTemplate" in fixed_code
        assert "tex_template" in fixed_code


class TestAutoFixerSystemPrompt:
    """系统提示测试。"""

    def test_prompt_contains_latex_patterns(self):
        """测试提示包含 LaTeX 错误模式。"""
        assert "LaTeX" in AUTO_FIXER_SYSTEM_PROMPT
        assert "\\frac" in AUTO_FIXER_SYSTEM_PROMPT
        assert "\\alpha" in AUTO_FIXER_SYSTEM_PROMPT

    def test_prompt_contains_import_patterns(self):
        """测试提示包含导入错误模式。"""
        assert "ImportError" in AUTO_FIXER_SYSTEM_PROMPT
        assert "NameError" in AUTO_FIXER_SYSTEM_PROMPT
        assert "from manim import *" in AUTO_FIXER_SYSTEM_PROMPT

    def test_prompt_contains_attribute_patterns(self):
        """测试提示包含属性错误模式。"""
        assert "AttributeError" in AUTO_FIXER_SYSTEM_PROMPT
        assert ".set_color()" in AUTO_FIXER_SYSTEM_PROMPT

    def test_prompt_contains_type_patterns(self):
        """测试提示包含类型错误模式。"""
        assert "TypeError" in AUTO_FIXER_SYSTEM_PROMPT
        assert "Axes" in AUTO_FIXER_SYSTEM_PROMPT

    def test_prompt_contains_timeout_patterns(self):
        """测试提示包含超时错误模式。"""
        assert "TIMEOUT" in AUTO_FIXER_SYSTEM_PROMPT
        assert "OOM" in AUTO_FIXER_SYSTEM_PROMPT

    def test_prompt_contains_recursion_patterns(self):
        """测试提示包含递归错误模式。"""
        assert "RecursionError" in AUTO_FIXER_SYSTEM_PROMPT
        assert "updater" in AUTO_FIXER_SYSTEM_PROMPT

    def test_prompt_contains_font_patterns(self):
        """测试提示包含字体错误模式。"""
        assert "PangoError" in AUTO_FIXER_SYSTEM_PROMPT
        assert "Noto Sans CJK SC" in AUTO_FIXER_SYSTEM_PROMPT

    def test_prompt_contains_repair_locator_principles(self):
        """测试提示包含修复定位原则。"""
        assert "修复定位原则" in AUTO_FIXER_SYSTEM_PROMPT
        assert "多段" in AUTO_FIXER_SYSTEM_PROMPT
        assert "保留 Scene 类名" in AUTO_FIXER_SYSTEM_PROMPT

    def test_prompt_contains_fix_principles(self):
        """测试提示包含修复原则。"""
        assert "最小改动" in AUTO_FIXER_SYSTEM_PROMPT
        assert "保持风格" in AUTO_FIXER_SYSTEM_PROMPT
        assert "完整输出" in AUTO_FIXER_SYSTEM_PROMPT
        assert "安全限制" in AUTO_FIXER_SYSTEM_PROMPT


class TestAutoFixerErrorHandling:
    """AutoFixer 错误处理测试。"""

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_fix_handles_llm_error(self, mock_call_llm, fixer):
        """测试处理 LLM 调用错误。"""
        from kd1_anime.exceptions import LLMError

        mock_call_llm.side_effect = LLMError("API 调用失败")

        with pytest.raises(LLMError):
            fixer.fix("code", "error")

    @patch("kd1_anime.agents.base.BaseAgent.call_llm")
    def test_fix_handles_empty_response(self, mock_call_llm, fixer):
        """测试处理空响应。"""
        mock_call_llm.return_value = ""

        code = fixer.fix("original code", "error log")
        # 应该返回空字符串或原始代码
        assert isinstance(code, str)
