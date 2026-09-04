import pytest

from kd1_anime.agents.validator import validate_manim_code
from kd1_anime.config import settings


def test_valid_manim_scene():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        self.add(Circle())\n"
    )
    assert result.is_valid
    assert result.scene_classes == ["Demo"]


def test_allows_scene_remove_for_mobject_lifecycle():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        circle = Circle()\n"
        "        self.add(circle)\n"
        "        self.remove(circle)\n"
    )

    assert result.is_valid, result.feedback


def test_still_rejects_remove_on_arbitrary_object():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        value = VGroup()\n"
        "        value.remove(Circle())\n"
    )

    assert not result.is_valid
    assert "remove" in result.feedback


def test_allows_common_local_names_that_shadow_manim_internals():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        scene = VGroup(Circle())\n"
        "        animation = Create(scene)\n"
        "        self.play(animation)\n"
    )
    assert result.is_valid, result.feedback


def test_rejects_dangerous_import_and_call():
    result = validate_manim_code(
        "from manim import *\nimport os\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        os.system('id')\n"
    )
    assert not result.is_valid
    assert "禁止导入模块" in result.feedback
    assert "禁止调用属性方法" in result.feedback


@pytest.mark.parametrize("constructor", ["ImageMobject", "SVGMobject", "SceneFileWriter"])
def test_rejects_module_qualified_dangerous_manim_objects(constructor):
    result = validate_manim_code(
        "import manim\n"
        "class Demo(manim.Scene):\n"
        "    def construct(self):\n"
        f"        value = manim.{constructor}('/tmp/input')\n"
        "        self.add(value)\n"
    )

    assert not result.is_valid
    assert constructor in result.feedback


def test_rejects_obvious_unbounded_loop():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        while True:\n"
        "            self.wait(0.1)\n"
    )

    assert not result.is_valid
    assert "无界循环" in result.feedback


def test_nested_loop_break_does_not_whitelist_outer_unbounded_loop():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        while True:\n"
        "            for _ in range(1):\n"
        "                break\n"
        "            self.wait(0.1)\n"
    )

    assert not result.is_valid
    assert "无界循环" in result.feedback


def test_rejects_missing_or_multiple_scene_classes():
    missing = validate_manim_code("from manim import *\nx = 1\n")
    assert not missing.is_valid
    multiple = validate_manim_code(
        "from manim import *\n"
        "class A(Scene):\n    def construct(self): pass\n"
        "class B(Scene):\n    def construct(self): pass\n"
    )
    assert not multiple.is_valid
    assert "只能定义一个" in multiple.feedback


def test_rejects_numpy_file_io():
    result = validate_manim_code(
        "from manim import *\n"
        "import numpy as np\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        np.save('/tmp/output.npy', np.array([1]))\n"
    )
    assert not result.is_valid
    assert "save" in result.feedback


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            "from manim import *\n"
            "from numpy import load\n"
            "class Demo(Scene):\n"
            "    def construct(self):\n"
            "        load('secret.npy')\n",
            "导入符号 'load'",
        ),
        (
            "from manim import *\n"
            "from numpy import *\n"
            "class Demo(Scene):\n"
            "    def construct(self): pass\n",
            "通配符导入",
        ),
        (
            "from manim import *\n"
            "from manim import config as cfg\n"
            "class Demo(Scene):\n"
            "    def construct(self):\n"
            "        cfg.media_dir = '/tmp/escape'\n",
            "config 创建别名",
        ),
        (
            "from manim import *\n"
            "import manim.utils.file_ops\n"
            "class Demo(Scene):\n"
            "    def construct(self): pass\n",
            "模块路径",
        ),
    ],
)
def test_rejects_import_allowlist_bypasses(source, expected):
    result = validate_manim_code(source)
    assert not result.is_valid
    assert expected in result.feedback


def test_rejects_manim_module_alias_config_access():
    result = validate_manim_code(
        "from manim import *\n"
        "import manim as m\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        m.config.media_dir = '/tmp/escape'\n"
    )
    assert not result.is_valid
    assert "模块别名" in result.feedback


def test_rejects_config_alias_created_by_assignment():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        cfg = config\n"
        "        cfg.media_dir = '/tmp/escape'\n"
    )
    assert not result.is_valid
    assert "config 别名" in result.feedback


def test_rejects_manim_internal_module_exposed_by_wildcard_import():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        utils.file_ops.write_file('/tmp/escape', 'x')\n"
    )
    assert not result.is_valid
    assert "内部模块" in result.feedback or "file_ops" in result.feedback


def test_rejects_dynamic_module_assignment():
    result = validate_manim_code(
        "from manim import *\n"
        "value = len([1, 2, 3])\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        self.wait()\n"
    )
    assert not result.is_valid
    assert "模块顶层" in result.feedback


def test_rejects_class_decorators_and_dynamic_class_body():
    result = validate_manim_code(
        "from manim import *\n"
        "def deco(cls):\n"
        "    return cls\n"
        "@deco\n"
        "class Demo(Scene):\n"
        "    value = len([1])\n"
        "    def construct(self):\n"
        "        self.wait()\n"
    )
    assert not result.is_valid
    assert "装饰器" in result.feedback
    assert "类体中禁止执行动态语句" in result.feedback


def test_accepts_xelatex_ctex_template_for_mathtex():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        tex_template = TexTemplate(tex_compiler='xelatex', output_format='.xdv')\n"
        "        tex_template.add_to_preamble(r'\\usepackage{ctex}')\n"
        "        config.tex_template = tex_template\n"
        "        equation = MathTex(r'x^2', tex_template=tex_template)\n"
        "        self.add(equation)\n"
    )

    assert result.is_valid, result.feedback


def test_rejects_xelatex_template_without_xdv_output():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        template = TexTemplate(tex_compiler='xelatex', output_format='.pdf')\n"
        "        template.add_to_preamble(r'\\usepackage{ctex}')\n"
        "        config.tex_template = template\n"
        "        self.add(Tex('x', tex_template=template))\n"
    )

    assert result.is_valid is False
    assert "output_format='.xdv'" in result.feedback


def test_rejects_dangerous_attribute_reference_without_direct_call():
    result = validate_manim_code(
        "from manim import *\n"
        "import numpy as np\n"
        "from functools import partial\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        loader = partial(np.load, 'secret.npy')\n"
        "        self.wait()\n"
    )

    assert result.is_valid is False
    assert "禁止引用危险属性 'load'" in result.feedback


def test_rejects_mathtex_using_default_pdflatex_template():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        self.add(MathTex(r'x^2'))\n"
    )

    assert not result.is_valid
    assert "xelatex" in result.feedback
    assert "tex_template=tex_template" in result.feedback


def test_rejects_xelatex_template_without_ctex_or_global_registration():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        template = TexTemplate(tex_compiler='xelatex', output_format='.xdv')\n"
        "        equation = Tex('中文', tex_template=template)\n"
        "        self.add(equation)\n"
    )

    assert not result.is_valid
    assert "ctex" in result.feedback
    assert "config.tex_template" in result.feedback


def test_rejects_custom_mobject_subclass(monkeypatch):
    """自定义 mobject 子类 (class X(VMobject)) 在 OpenGL 下缺 should_render, 必须拦截。"""
    monkeypatch.setattr(settings, "MANIM_RENDERER", "opengl")
    result = validate_manim_code(
        "from manim import *\n"
        "class Polygon(VMobject):\n"
        "    def __init__(self, **kwargs):\n"
        "        super().__init__(**kwargs)\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        self.add(Polygon())\n"
    )
    assert not result.is_valid
    assert "自定义 mobject 子类" in result.feedback


def test_allows_custom_mobject_subclass_for_cairo(monkeypatch):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "cairo")
    result = validate_manim_code(
        "from manim import *\n"
        "class Shape(VMobject):\n"
        "    def __init__(self, **kwargs):\n"
        "        super().__init__(**kwargs)\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        self.add(Shape())\n"
    )
    assert result.is_valid


def test_rejects_qualified_custom_mobject_subclass_for_opengl(monkeypatch):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "opengl")
    result = validate_manim_code(
        "from manim import *\n"
        "import manim\n"
        "class Shape(manim.VMobject):\n"
        "    pass\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        self.add(Shape())\n"
    )
    assert not result.is_valid
    assert "自定义 mobject 子类" in result.feedback


def test_rejects_dangerous_callable_hidden_in_container():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        funcs = [open]\n"
        "        funcs[0]('/tmp/out', 'w')\n"
    )
    assert not result.is_valid
    assert "危险能力 'open'" in result.feedback


def test_qualified_tex_still_requires_template():
    result = validate_manim_code(
        "from manim import *\n"
        "import manim\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        self.add(manim.Tex('中文'))\n"
    )
    assert not result.is_valid
    assert "xelatex" in result.feedback


def test_allows_helper_class():
    """非 mobject 的普通辅助类不应被误拦截。"""
    result = validate_manim_code(
        "from manim import *\n"
        "class Helper:\n"
        '    LABEL = "demo"\n'
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        self.add(Circle())\n"
    )
    assert result.is_valid


def test_rejects_camera_frame_in_plain_scene():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        self.camera.frame.set(width=14, height=6.5)\n"
    )
    assert not result.is_valid
    assert "camera.frame" in result.feedback
    assert "MovingCameraScene" in result.feedback


def test_rejects_3d_objects_in_plain_opengl_scene():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        axes = ThreeDAxes()\n"
        "        surface = Surface(lambda u, v: np.array([u, v, u + v]))\n"
        "        self.add(axes, surface)\n",
        renderer="opengl",
    )

    assert not result.is_valid
    assert "必须继承 ThreeDScene" in result.feedback


def test_rejects_three_d_camera_setup_in_plain_scene():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        self.set_camera_orientation(phi=75 * DEGREES)\n"
    )

    assert not result.is_valid
    assert "set_camera_orientation" in result.feedback
    assert "ThreeDScene" in result.feedback


def test_accepts_camera_frame_in_moving_camera_scene():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(MovingCameraScene):\n"
        "    def construct(self):\n"
        "        self.camera.frame.set(width=14, height=6.5)\n"
    )
    assert result.is_valid


def test_explicit_renderer_context_overrides_process_setting(monkeypatch):
    monkeypatch.setattr(settings, "MANIM_RENDERER", "opengl")
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(MovingCameraScene):\n"
        "    def construct(self):\n"
        "        self.camera.frame.set(width=10)\n",
        renderer="cairo",
    )

    assert result.is_valid


def test_rejects_camera_frame_in_helper_method():
    """camera.frame 出现在辅助方法 (非 construct) 时也必须被拦截。"""
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def _set_camera_width(self, width):\n"
        "        frame = self.camera.frame\n"
        "        frame.set(width=width)\n"
        "    def construct(self):\n"
        "        self._set_camera_width(14)\n"
    )
    assert not result.is_valid
    assert "camera.frame" in result.feedback


def test_rejects_camera_frame_under_opengl_even_in_moving_camera_scene(monkeypatch):
    """OpenGL 渲染器 (OpenGLCamera 无 frame) 下, MovingCameraScene 也无效。"""
    monkeypatch.setattr(settings, "MANIM_RENDERER", "opengl")
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(MovingCameraScene):\n"
        "    def construct(self):\n"
        "        self.camera.frame.set(width=14, height=6.5)\n"
    )
    assert not result.is_valid
    assert "OpenGL" in result.feedback


def test_accepts_moving_camera_scene_without_frame_under_opengl(monkeypatch):
    """OpenGL 下不使用 camera.frame 的普通 Scene 仍然合法。"""
    monkeypatch.setattr(settings, "MANIM_RENDERER", "opengl")
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        self.add(Circle())\n"
    )
    assert result.is_valid


def test_rejects_generated_code_overriding_global_output_configuration():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        config.media_dir = '/tmp/escaped'\n"
        "        self.add(Circle())\n"
    )

    assert not result.is_valid
    assert "config.media_dir" in result.feedback


def test_rejects_module_level_attribute_assignment():
    result = validate_manim_code(
        "from manim import *\n"
        "config.output_file = '/tmp/escaped.mp4'\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        self.add(Circle())\n"
    )

    assert not result.is_valid
    assert "顶层" in result.feedback
