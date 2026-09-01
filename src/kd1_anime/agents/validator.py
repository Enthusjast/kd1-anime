"""生成代码的确定性校验与基础安全策略。

LLM Reviewer 负责视觉和 Manim 逻辑质量；本模块负责不能交给 LLM
作为安全边界的确定性检查：Python 语法、Scene 类结构和危险能力。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Literal

from kd1_anime.config import settings

ALLOWED_IMPORT_ROOTS = {
    "manim",
    "math",
    "numpy",
    "random",
    "itertools",
    "functools",
    "collections",
}

# 生成代码只需要这些顶层模块。模块路径必须精确匹配；例如
# ``import manim.utils.file_ops`` 和 ``import numpy.random`` 都不能因为根模块
# 在白名单中而被放行。
ALLOWED_IMPORT_MODULES = ALLOWED_IMPORT_ROOTS
ALLOWED_MODULE_ALIASES = {
    "manim": {"manim"},
    "math": {"math"},
    "numpy": {"numpy", "np"},
    "random": {"random"},
    "itertools": {"itertools"},
    "functools": {"functools"},
    "collections": {"collections"},
}

# 只允许从纯计算模块导入明确的无文件/网络能力。Manim 的公开类很多，
# 因此保留项目提示词要求的 ``from manim import *``，但禁止它的别名形式
# 以及危险对象；其它模块不允许通配符导入。
SAFE_FROM_IMPORTS = {
    "math": {
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "ceil",
        "copysign",
        "cos",
        "cosh",
        "degrees",
        "e",
        "erf",
        "exp",
        "expm1",
        "fabs",
        "factorial",
        "floor",
        "fmod",
        "frexp",
        "gamma",
        "gcd",
        "hypot",
        "inf",
        "isclose",
        "isfinite",
        "isinf",
        "isnan",
        "lcm",
        "ldexp",
        "lgamma",
        "log",
        "log10",
        "log1p",
        "log2",
        "modf",
        "nan",
        "perm",
        "pi",
        "pow",
        "radians",
        "remainder",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "tau",
        "trunc",
    },
    "numpy": {
        "abs",
        "absolute",
        "arange",
        "array",
        "asarray",
        "bool_",
        "ceil",
        "clip",
        "cos",
        "dot",
        "e",
        "exp",
        "eye",
        "float32",
        "float64",
        "floor",
        "full",
        "int32",
        "int64",
        "linspace",
        "matmul",
        "maximum",
        "minimum",
        "nan",
        "newaxis",
        "ndarray",
        "ones",
        "pi",
        "sign",
        "sin",
        "sqrt",
        "tan",
        "uint8",
        "where",
        "zeros",
    },
    "random": {
        "choice",
        "randint",
        "random",
        "randrange",
        "sample",
        "uniform",
    },
    "itertools": {
        "chain",
        "combinations",
        "cycle",
        "islice",
        "permutations",
        "product",
        "repeat",
        "starmap",
        "zip_longest",
    },
    "functools": {"partial", "reduce"},
    "collections": {"Counter", "OrderedDict", "defaultdict", "deque", "namedtuple"},
}

# 明确禁止的模块 - 这些模块可能提供危险能力
BANNED_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "breakpoint",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "print",
    # 项目没有资产上传/白名单机制，禁止通过任意路径加载本地图片或 SVG。
    "ImageMobject",
    "OpenGLImageMobject",
    "SVGMobject",
    "SceneFileWriter",
}

# ``from manim import *`` 也会把若干内部模块带入局部命名空间；它们不是
# 场景 API，且可能间接提供文件/渲染器访问。显式导入这些名称同样拒绝。
BANNED_MANIM_INTERNALS = {
    "animation",
    "camera",
    "core",
    "data_structures",
    "mobject",
    "opengl",
    "plugins",
    "renderer",
    "scene",
    "typing",
    "utils",
}

BANNED_ATTRIBUTE_NAMES = {
    "system",
    "popen",
    "spawn",
    "fork",
    "execv",
    "execve",
    "remove",
    "unlink",
    "rmdir",
    "rmtree",
    "chmod",
    "chown",
    "connect",
    "request",
    "urlopen",
    # NumPy/类似对象暴露的文件和动态库能力。
    "save",
    "savez",
    "savez_compressed",
    "load",
    "loadtxt",
    "genfromtxt",
    "savetxt",
    "memmap",
    "fromfile",
    "tofile",
    "dump",
    "dumps",
    "load_library",
    "open_memmap",
    "file_ops",
    "write_file",
    "read_file",
    "guarantee_existence",
    "modify_path",
    "__subclasses__",
    "__globals__",
    "__code__",
    "__builtins__",
}

SCENE_BASES = {"Scene", "ThreeDScene", "MovingCameraScene"}
# 这些类需要三维场景的相机/渲染上下文。OpenGL 下把它们放进普通
# ``Scene`` 是一个确定的类型错误；不要等到远端渲染才由 Reviewer 猜测。
THREE_D_CONSTRUCTORS = {
    "Arrow3D",
    "Cone",
    "Cube",
    "Cylinder",
    "DashedLine3D",
    "Dot3D",
    "Line3D",
    "ParametricSurface",
    "Polyhedron",
    "Prism",
    "Sphere",
    "Surface",
    "Tetrahedron",
    "ThreeDAxes",
    "Torus",
}
# mobject 继承树的根类: manim 只对"子类的基类"做 OpenGL 转换, Mobject/VMobject
# 这两个根类本身始终是 Cairo 版。场景文件里自定义这些根类的子类, 在 OpenGL
# 渲染下会得到缺少 should_render 的 Cairo 对象, 渲染时直接 AttributeError。
MOBJECT_BASES = {
    "Mobject",
    "VMobject",
    "PMobject",
    "Mobject1D",
    "Mobject2D",
    "OpenGLMobject",
    "OpenGLVMobject",
    "OpenGLPMobject",
}
TEX_MOBJECTS = {"Tex", "MathTex"}


def _find_camera_frame_access(node: ast.AST) -> list[ast.Attribute]:
    """查找 self.camera.frame 属性访问节点 (仅在 MovingCameraScene 中合法)。"""
    found: list[ast.Attribute] = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Attribute)
            and child.attr == "frame"
            and isinstance(child.value, ast.Attribute)
            and child.value.attr == "camera"
            and isinstance(child.value.value, ast.Name)
            and child.value.value.id == "self"
        ):
            found.append(child)
    return found


def _is_static_expression(node: ast.AST | None) -> bool:
    """模块/类定义阶段允许求值的无副作用表达式。"""

    if node is None:
        return True
    if isinstance(node, ast.Constant | ast.Name):
        return True
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)) and _is_static_expression(
            node.operand
        )
    if isinstance(node, ast.BinOp):
        return _is_static_expression(node.left) and _is_static_expression(node.right)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_static_expression(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(_is_static_expression(item) for item in node.keys) and all(
            _is_static_expression(item) for item in node.values
        )
    return False


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _collect_bound_names(tree: ast.AST) -> set[str]:
    """收集源码中明确绑定的局部名称。

    ``from manim import *`` 会把若干内部模块名放入命名空间；但 ``scene``
    或 ``animation`` 也是生成代码中很常见的普通变量名。只对未被源码
    绑定、确实可能来自 wildcard import 的名称做内部模块拦截，避免安全
    检查误伤合法的 VGroup/Animation 变量。
    """

    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    bound.add(alias.asname or alias.name)
    return bound


@dataclass(slots=True)
class CodeValidationResult:
    """确定性代码校验结果。"""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    scene_classes: list[str] = field(default_factory=list)

    @property
    def feedback(self) -> str:
        return "\n".join(f"- {error}" for error in self.errors)


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self, renderer: Literal["cairo", "opengl"] | None = None) -> None:
        self.errors: list[str] = []
        self.scene_classes: list[str] = []
        self.has_manim_import = False
        self.xelatex_templates: set[str] = set()
        self.ctex_templates: set[str] = set()
        self.configured_templates: set[str] = set()
        self.tex_calls: list[tuple[ast.Call, str | None]] = []
        self.imported_modules: set[str] = set()
        self.dangerous_aliases: set[str] = set()
        self.module_aliases: dict[str, str] = {}
        self.imported_names: dict[str, tuple[str, str]] = {}
        self.bound_names: set[str] = set()
        self.config_aliases: set[str] = {"config"}
        self.renderer = renderer or settings.MANIM_RENDERER

    def error(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", "?")
        self.errors.append(f"第 {line} 行: {message}")

    def _check_config_assignment(self, target: ast.expr, node: ast.AST) -> None:
        path = self._attribute_path(target)
        if not path:
            return
        if path[0] == "config":
            if len(path) != 2 or path[1] != "tex_template":
                self.error(node, f"禁止修改 Manim 全局配置 {'.'.join(path)}")
        elif path[0] in self.config_aliases:
            self.error(node, f"禁止通过 config 别名修改 Manim 全局配置 {'.'.join(path)}")
        elif len(path) >= 2 and self.module_aliases.get(path[0]) == "manim" and path[1] == "config":
            self.error(node, f"禁止通过模块别名修改 Manim 全局配置 {'.'.join(path)}")

    def _attribute_path(self, node: ast.AST) -> list[str] | None:
        """返回简单的 ``a.b.c`` 路径，复杂表达式不视为可安全追踪路径。"""

        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, ast.Attribute):
            parent = self._attribute_path(node.value)
            return [*parent, node.attr] if parent else None
        return None

    def _check_import_module(self, node: ast.AST, module: str, *, from_import: bool) -> str:
        root = module.split(".", 1)[0]
        self.imported_modules.add(root)
        if root not in ALLOWED_IMPORT_MODULES:
            self.error(
                node,
                f"禁止{'从模块' if from_import else ''}导入模块 {module!r}",
            )
        elif module != root:
            self.error(node, f"只允许导入白名单顶层模块，不允许模块路径 {module!r}")
        return root

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = self._check_import_module(node, alias.name, from_import=False)
            local_name = alias.asname or alias.name.split(".", 1)[0]
            allowed_aliases = ALLOWED_MODULE_ALIASES.get(root, set())
            if alias.asname and alias.asname not in allowed_aliases:
                self.error(node, f"模块 {alias.name!r} 不允许使用别名 {alias.asname!r}")
            if alias.name == root:
                self.module_aliases[local_name] = root
            if root == "manim":
                self.has_manim_import = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level:
            self.error(node, "禁止相对导入")
        root = self._check_import_module(node, module, from_import=True) if not node.level else ""
        if not node.level and root in ALLOWED_IMPORT_MODULES:
            if any(alias.name == "*" for alias in node.names) and root != "manim":
                self.error(node, f"禁止从模块 {module!r} 使用通配符导入")
            allowed_names = SAFE_FROM_IMPORTS.get(root, set())
            for alias in node.names:
                local_name = alias.asname or alias.name
                if alias.name == "*":
                    continue
                if root != "manim" and alias.name not in allowed_names:
                    self.error(node, f"禁止从模块 {module!r} 导入符号 {alias.name!r}")
                if root == "manim" and alias.name in BANNED_MANIM_INTERNALS:
                    self.error(node, f"禁止导入 Manim 内部模块 {alias.name!r}")
                if alias.asname and alias.asname != alias.name:
                    self.error(node, f"禁止为导入符号 {alias.name!r} 创建别名 {alias.asname!r}")
                if root == "manim" and alias.name == "config" and alias.asname:
                    self.error(
                        node, "禁止为 Manim config 创建别名；只能直接使用 config.tex_template"
                    )
                self.imported_names[local_name] = (root, alias.name)
        if root == "manim":
            self.has_manim_import = True
            for alias in node.names:
                if alias.name in BANNED_CALLS:
                    local_name = alias.asname or alias.name
                    self.dangerous_aliases.add(local_name)
                    self.error(node, f"禁止导入危险 Manim 对象 {alias.name!r}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # 检查直接函数调用
        if isinstance(node.func, ast.Name) and (
            node.func.id in BANNED_CALLS or node.func.id in self.dangerous_aliases
        ):
            self.error(node, f"禁止调用 {node.func.id}()")
        # 检查属性方法调用
        elif (
            isinstance(node.func, ast.Attribute)
            and (
                node.func.attr in BANNED_ATTRIBUTE_NAMES
                or (node.func.attr.startswith("__") and not self._is_super_init(node.func))
            )
            and not self._is_scene_remove(node.func)
        ):
            self.error(node, f"禁止调用属性方法 {node.func.attr}()")

        # 检查 add_to_preamble 调用并提取模板变量名
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_to_preamble":
            # 检查是否加载了 ctex
            for arg in node.args:
                if isinstance(arg, ast.Constant) and "ctex" in str(arg.value):
                    # 提取调用对象的变量名
                    if isinstance(node.func.value, ast.Name):
                        self.ctex_templates.add(node.func.value.id)
                    break

        # 检查 Tex/MathTex 调用
        tex_constructor = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        if tex_constructor in TEX_MOBJECTS:
            template_keyword = next(
                (keyword for keyword in node.keywords if keyword.arg == "tex_template"),
                None,
            )
            template_name = (
                template_keyword.value.id
                if template_keyword and isinstance(template_keyword.value, ast.Name)
                else None
            )
            self.tex_calls.append((node, template_name))

        # 检查动态构造危险调用（如 getattr(os, "system")）
        if isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2:
            attr_arg = node.args[1]
            if (
                isinstance(attr_arg, ast.Constant)
                and isinstance(attr_arg.value, str)
                and attr_arg.value in BANNED_ATTRIBUTE_NAMES
            ):
                self.error(node, f"禁止通过 getattr 访问危险属性 {attr_arg.value!r}")

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            self.error(node, f"禁止访问双下划线名称 {node.id!r}")
        if (
            isinstance(node.ctx, ast.Load)
            and node.id in BANNED_MANIM_INTERNALS
            and node.id not in self.bound_names
        ):
            self.error(node, f"禁止访问 Manim 内部模块 {node.id!r}")
        if isinstance(node.ctx, ast.Load) and (
            node.id in BANNED_CALLS or node.id in self.dangerous_aliases
        ):
            self.error(node, f"禁止引用危险能力 {node.id!r}")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in {"__builtins__", "builtins"}:
            self.error(node, "禁止通过 builtins 下标访问运行时能力")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BANNED_ATTRIBUTE_NAMES and not self._is_scene_remove(node):
            self.error(node, f"禁止引用危险属性 {node.attr!r}")
        if node.attr.startswith("__") and not self._is_super_init(node):
            self.error(node, f"禁止访问双下划线属性 {node.attr}")
        path = self._attribute_path(node)
        if path and path[0] == "config" and (len(path) != 2 or path[1] != "tex_template"):
            self.error(node, f"禁止访问 Manim 全局配置 {'.'.join(path)}")
        if path and path[0] in self.config_aliases and path[0] != "config":
            self.error(node, "禁止通过 config 别名访问 Manim 全局配置")
        if (
            path
            and len(path) >= 2
            and self.module_aliases.get(path[0]) == "manim"
            and (path[1] == "config" or path[1] in BANNED_MANIM_INTERNALS)
        ):
            if path[1] == "config":
                self.error(
                    node, "禁止通过模块别名访问 Manim 全局配置；只能直接使用 config.tex_template"
                )
            else:
                self.error(node, f"禁止通过模块别名访问 Manim 内部模块 {path[1]!r}")
        self.generic_visit(node)

    @staticmethod
    def _is_super_init(node: ast.Attribute) -> bool:
        """只放行类构造器中常见且无额外能力的 ``super().__init__``。"""

        return (
            node.attr == "__init__"
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "super"
            and not node.value.args
            and not node.value.keywords
        )

    @staticmethod
    def _is_scene_remove(node: ast.Attribute) -> bool:
        """只放行 Manim 的 ``self.remove``，不放行任意对象的 remove。

        ``remove`` 同时是容器和文件 API 的常见危险属性，因此仍保留在
        全局黑名单中。Scene 的 remove 只会从当前 Manim 场景移除 Mobject，
        且生命周期检查器需要理解这个操作；仅对明确的 ``self.remove`` 做
        窄例外，避免把 ``some_path.remove`` 等能力放进生成代码。
        """

        return (
            node.attr == "remove" and isinstance(node.value, ast.Name) and node.value.id == "self"
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        value_path = self._attribute_path(node.value)
        value_is_config = bool(
            value_path
            and (
                value_path[0] in self.config_aliases
                or (
                    len(value_path) >= 2
                    and self.module_aliases.get(value_path[0]) == "manim"
                    and value_path[1] == "config"
                )
            )
        )
        value_is_module = isinstance(node.value, ast.Name) and node.value.id in self.module_aliases
        if value_is_config or value_is_module:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if value_is_config:
                        self.config_aliases.add(target.id)
                    if value_is_module:
                        self.module_aliases[target.id] = self.module_aliases[node.value.id]
        if isinstance(node.value, ast.Name) and (
            node.value.id in BANNED_CALLS or node.value.id in self.dangerous_aliases
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.dangerous_aliases.add(target.id)
                    self.error(node, f"禁止为危险 callable 创建别名 {target.id!r}")
        if isinstance(node.value, ast.Attribute) and (
            node.value.attr in BANNED_ATTRIBUTE_NAMES or node.value.attr.startswith("__")
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.dangerous_aliases.add(target.id)
                    self.error(node, f"禁止为危险属性创建别名 {target.id!r}")
        # 检查 TexTemplate 赋值: tex_template = TexTemplate(tex_compiler='xelatex', ...)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
        ):
            func_name = ""
            if isinstance(node.value.func, ast.Name):
                func_name = node.value.func.id
            elif isinstance(node.value.func, ast.Attribute):
                func_name = node.value.func.attr
            if func_name == "TexTemplate":
                keyword_values = {
                    keyword.arg: keyword.value
                    for keyword in node.value.keywords
                    if keyword.arg is not None
                }
                compiler = keyword_values.get("tex_compiler")
                output = keyword_values.get("output_format")
                if (
                    isinstance(compiler, ast.Constant)
                    and compiler.value == "xelatex"
                    and isinstance(output, ast.Constant)
                    and output.value == ".xdv"
                ):
                    self.xelatex_templates.add(node.targets[0].id)

        # 不允许生成代码改写 Manim 的输出目录/渲染器等全局配置；
        # 唯一放行的属性赋值是项目要求的 config.tex_template。
        for target in node.targets:
            self._check_config_assignment(target, node)

        # 检查 config.tex_template = ... 赋值
        if len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "tex_template"
                and isinstance(target.value, ast.Name)
                and target.value.id == "config"
                and isinstance(node.value, ast.Name)
            ):
                self.configured_templates.add(node.value.id)

        # 允许类属性赋值（已在 visit_ClassDef 中检查）
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_config_assignment(node.target, node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_config_assignment(node.target, node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.decorator_list:
            self.error(node, f"禁止在函数 {node.name!r} 上使用装饰器")
        defaults = [*node.args.defaults, *node.args.kw_defaults]
        if not all(_is_static_expression(default) for default in defaults):
            self.error(node, f"函数 {node.name!r} 的默认参数不能执行动态表达式")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.error(node, "Manim 场景不允许定义 async 函数")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.decorator_list:
            self.error(node, f"禁止在类 {node.name!r} 上使用装饰器")
        if node.keywords:
            self.error(node, f"类 {node.name!r} 不允许 metaclass 等动态关键字")
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.Pass)) or _is_docstring(statement):
                continue
            if isinstance(statement, ast.Assign) and _is_static_expression(statement.value):
                continue
            if isinstance(statement, ast.AnnAssign) and _is_static_expression(statement.value):
                continue
            self.error(statement, f"类 {node.name!r} 的类体中禁止执行动态语句")
        bases = {
            base.id if isinstance(base, ast.Name) else base.attr
            for base in node.bases
            if isinstance(base, (ast.Name, ast.Attribute))
        }
        # 禁止自定义 mobject 子类: OpenGL 渲染器只接受 OpenGLMobject 家族,
        # class X(Mobject)/class X(VMobject) 会产出缺少 should_render 的
        # Cairo 对象, 在 opengl_renderer.update_frame 渲染崩溃。
        if self.renderer == "opengl" and bases & MOBJECT_BASES and not (bases & SCENE_BASES):
            self.error(
                node,
                "禁止自定义 mobject 子类 (Mobject/VMobject/PMobject 及其 OpenGL 版)。"
                "OpenGL 渲染器下这类对象没有 should_render 属性, 渲染会崩溃; "
                "请直接在 construct() 内用 manim 标准类 (Polygon/VGroup/Line/"
                "Square/MathTex 等) 组合, 不要自定义继承",
            )
        if bases & SCENE_BASES:
            self.scene_classes.append(node.name)
            construct = next(
                (
                    item
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "construct"
                ),
                None,
            )
            if construct is None:
                self.error(node, f"Scene 类 {node.name!r} 缺少 construct() 方法")
            if "ThreeDScene" not in bases:
                if self.renderer == "opengl":
                    for call in ast.walk(node):
                        constructor = (
                            call.func.id
                            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                            else call.func.attr
                            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                            else ""
                        )
                        if constructor in THREE_D_CONSTRUCTORS:
                            self.error(
                                call,
                                f"OpenGL 场景使用 {constructor}() 等三维对象时必须继承 "
                                "ThreeDScene；请将当前场景基类改为 ThreeDScene，"
                                "不要在普通 Scene 中创建三维对象",
                            )
                for call in ast.walk(node):
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "self"
                        and call.func.attr == "set_camera_orientation"
                    ):
                        self.error(
                            call,
                            "self.set_camera_orientation() 只适用于 ThreeDScene；"
                            "请将场景继承改为 ThreeDScene 或删除三维相机设置",
                        )
            # self.camera.frame 检查: 扫描整个类体 (含辅助方法, 不只是 construct)。
            # - OpenGL 渲染器固定使用 OpenGLCamera (没有 frame 属性), 继承
            #   MovingCameraScene 也无效 → 一律禁止相机运镜, 必须删除。
            # - Cairo 渲染器下 frame 仅在 MovingCameraScene 中可用。
            frame_access = _find_camera_frame_access(node)
            renderer = self.renderer
            if renderer == "opengl":
                if "MovingCameraScene" in bases:
                    self.error(
                        node,
                        "本项目使用 OpenGL 渲染器 (OpenGLCamera 没有 frame 属性), "
                        "MovingCameraScene 的相机运镜不可用。请删除所有相机移动代码, "
                        "改用静态布局 (next_to/to_edge/arrange) 或 Transform 动画",
                    )
                for frame_node in frame_access:
                    self.error(
                        frame_node,
                        "OpenGL 渲染器 (OpenGLCamera) 没有 frame 属性, 禁止使用 "
                        "self.camera.frame / 相机运镜。请删除所有相关代码 (含辅助方法), "
                        "改用静态布局 (next_to/to_edge/arrange) 或 Transform 动画",
                    )
            elif "MovingCameraScene" not in bases:
                for frame_node in frame_access:
                    self.error(
                        frame_node,
                        "self.camera.frame 只在 MovingCameraScene 中可用；"
                        "普通 Scene 的相机(Camera/OpenGLCamera)没有 frame 属性。"
                        "需要推近/缩放镜头时请继承 MovingCameraScene",
                    )
        self.generic_visit(node)

    def validate_tex_configuration(self) -> None:
        """检查 TexTemplate 配置是否完整。"""
        if not self.tex_calls:
            return
        first_call = self.tex_calls[0][0]
        HINT = " (修复: 参考 coder.py 提示词中的 TexTemplate 模板)"
        if not self.xelatex_templates:
            self.error(
                first_call,
                "Tex/MathTex 必须使用 TexTemplate(tex_compiler='xelatex', output_format='.xdv') (修复: 参考 coder.py 提示词中的 TexTemplate 模板)",
            )
        if not (self.xelatex_templates & self.ctex_templates):
            self.error(first_call, r"XeLaTeX 模板必须加载 \usepackage{ctex}" + HINT)
        if not (self.xelatex_templates & self.configured_templates):
            self.error(first_call, "必须将 XeLaTeX 模板赋给 config.tex_template" + HINT)

        for call, template_name in self.tex_calls:
            if template_name is None:
                self.error(
                    call, "每个 Tex/MathTex 调用都必须显式传入 tex_template=tex_template" + HINT
                )
            elif template_name not in self.xelatex_templates:
                self.error(call, "Tex/MathTex 的 tex_template 必须引用 XeLaTeX .xdv 模板" + HINT)


def validate_manim_code(
    code: str,
    *,
    renderer: Literal["cairo", "opengl"] | None = None,
) -> CodeValidationResult:
    """校验生成的 Manim 代码，返回可展示给 Coder 的确定性反馈。"""

    if not code.strip():
        return CodeValidationResult(False, ["代码为空"])

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        location = f"第 {exc.lineno or '?'} 行"
        return CodeValidationResult(False, [f"Python 语法错误（{location}）: {exc.msg}"])

    visitor = _SafetyVisitor(renderer)
    visitor.bound_names = _collect_bound_names(tree)
    visitor.visit(tree)
    visitor.validate_tex_configuration()

    # 检查顶层语句
    for statement in tree.body:
        if isinstance(
            statement,
            (
                ast.Import,
                ast.ImportFrom,
                ast.ClassDef,
                ast.FunctionDef,
            ),
        ):
            continue
        if (
            isinstance(statement, ast.Assign)
            and all(isinstance(target, ast.Name) for target in statement.targets)
            and _is_static_expression(statement.value)
        ):
            continue
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and _is_static_expression(statement.value)
        ):
            continue
        if _is_docstring(statement):
            continue
        visitor.error(statement, "禁止在模块顶层执行语句")

    # 检查导入
    if not visitor.has_manim_import:
        visitor.errors.append("缺少 Manim 导入（应使用 from manim import *）")

    # 检查 Scene 类数量
    if not visitor.scene_classes:
        visitor.errors.append("未找到继承 Scene、ThreeDScene 或 MovingCameraScene 的场景类")
    elif len(visitor.scene_classes) != 1:
        visitor.errors.append(
            f"每个场景文件必须且只能定义一个可渲染 Scene 类，当前为 {len(visitor.scene_classes)} 个"
        )

    # 去重并保持报告顺序稳定。
    errors = list(dict.fromkeys(visitor.errors))
    return CodeValidationResult(
        is_valid=not errors,
        errors=errors,
        scene_classes=visitor.scene_classes,
    )
