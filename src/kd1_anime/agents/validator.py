"""生成代码的确定性校验与基础安全策略。

LLM Reviewer 负责视觉和 Manim 逻辑质量；本模块负责不能交给 LLM
作为安全边界的确定性检查：Python 语法、Scene 类结构和危险能力。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

ALLOWED_IMPORT_ROOTS = {
    "manim",
    "math",
    "numpy",
    "random",
    "itertools",
    "functools",
    "collections",
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
    "__subclasses__",
    "__globals__",
    "__code__",
    "__builtins__",
}

SCENE_BASES = {"Scene", "ThreeDScene", "MovingCameraScene"}
TEX_MOBJECTS = {"Tex", "MathTex"}


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
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.scene_classes: list[str] = []
        self.has_manim_import = False
        self.xelatex_templates: set[str] = set()
        self.ctex_templates: set[str] = set()
        self.configured_templates: set[str] = set()
        self.tex_calls: list[tuple[ast.Call, str | None]] = []
        self.imported_modules: set[str] = set()

    def error(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", "?")
        self.errors.append(f"第 {line} 行: {message}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            self.imported_modules.add(root)
            if root not in ALLOWED_IMPORT_ROOTS:
                self.error(node, f"禁止导入模块 {alias.name!r}")
            if root == "manim":
                self.has_manim_import = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".", 1)[0]
        self.imported_modules.add(root)
        if node.level:
            self.error(node, "禁止相对导入")
        elif root not in ALLOWED_IMPORT_ROOTS:
            self.error(node, f"禁止从模块 {module!r} 导入")
        if root == "manim":
            self.has_manim_import = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # 检查直接函数调用
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            self.error(node, f"禁止调用 {node.func.id}()")
        # 检查属性方法调用
        elif isinstance(node.func, ast.Attribute) and (
            node.func.attr in BANNED_ATTRIBUTE_NAMES or node.func.attr.startswith("__")
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
        if isinstance(node.func, ast.Name) and node.func.id in TEX_MOBJECTS:
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
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) >= 2:
                obj_arg = node.args[0]
                attr_arg = node.args[1]
                # 检查是否在访问危险属性
                if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
                    if attr_arg.value in BANNED_ATTRIBUTE_NAMES:
                        self.error(node, f"禁止通过 getattr 访问危险属性 {attr_arg.value!r}")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.error(node, f"禁止访问双下划线属性 {node.attr}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # 检查 TexTemplate 赋值: tex_template = TexTemplate(tex_compiler='xelatex', ...)
        if (len(node.targets) == 1 and 
            isinstance(node.targets[0], ast.Name) and 
            isinstance(node.value, ast.Call)):
            func_name = ""
            if isinstance(node.value.func, ast.Name):
                func_name = node.value.func.id
            elif isinstance(node.value.func, ast.Attribute):
                func_name = node.value.func.attr
            if func_name == "TexTemplate":
                # 检查是否使用 xelatex 编译器
                for kw in node.value.keywords:
                    if kw.arg == "tex_compiler" and isinstance(kw.value, ast.Constant):
                        if kw.value.value == "xelatex":
                            self.xelatex_templates.add(node.targets[0].id)
                        break
        
        # 检查 config.tex_template = ... 赋值
        if len(node.targets) == 1:
            target = node.targets[0]
            if (isinstance(target, ast.Attribute) and 
                target.attr == "tex_template" and
                isinstance(target.value, ast.Name) and 
                target.value.id == "config"):
                # 找到被赋值的模板变量名
                if isinstance(node.value, ast.Name):
                    self.configured_templates.add(node.value.id)
        
        # 允许类属性赋值（已在 visit_ClassDef 中检查）
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
        bases = {base.id for base in node.bases if isinstance(base, ast.Name)}
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
                self.error(call, "每个 Tex/MathTex 调用都必须显式传入 tex_template=tex_template" + HINT)
            elif template_name not in self.xelatex_templates:
                self.error(call, "Tex/MathTex 的 tex_template 必须引用 XeLaTeX .xdv 模板" + HINT)


def validate_manim_code(code: str) -> CodeValidationResult:
    """校验生成的 Manim 代码，返回可展示给 Coder 的确定性反馈。"""

    if not code.strip():
        return CodeValidationResult(False, ["代码为空"])

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        location = f"第 {exc.lineno or '?'} 行"
        return CodeValidationResult(False, [f"Python 语法错误（{location}）: {exc.msg}"])

    visitor = _SafetyVisitor()
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
        if isinstance(statement, ast.Assign) and _is_static_expression(statement.value):
            continue
        if isinstance(statement, ast.AnnAssign) and _is_static_expression(statement.value):
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
