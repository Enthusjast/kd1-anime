from kd1_anime.agents.validator import validate_manim_code


def test_valid_manim_scene():
    result = validate_manim_code(
        "from manim import *\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        self.add(Circle())\n"
    )
    assert result.is_valid
    assert result.scene_classes == ["Demo"]


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
