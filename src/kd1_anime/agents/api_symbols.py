"""Lexical resolution of statically identifiable Manim calls, without execution."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class Scope:
    parent: Scope | None
    kind: str
    bindings: dict[str, str | None] = field(default_factory=dict)
    star: bool = False
    globals: set[str] = field(default_factory=set)
    nonlocals: set[str] = field(default_factory=set)

    def resolve(self, name: str) -> str | None:
        scope = self
        if name in scope.nonlocals:
            scope = scope.parent
            while scope and scope.kind != "function":
                scope = scope.parent
            return scope.resolve(name) if scope else None
        if name in scope.globals:
            while scope.parent:
                scope = scope.parent
        while scope:
            if name in scope.bindings:
                return scope.bindings[name]
            if scope.star:
                import builtins

                return None if hasattr(builtins, name) else "manim." + name
            scope = scope.parent
            while scope is not None and scope.kind == "class":
                scope = scope.parent
        return None


class Bindings(ast.NodeVisitor):
    def __init__(self, scope: Scope):
        self.scope = scope
        self.imported: set[str] = set()

    def bind(self, name, value):
        if name in self.scope.bindings and self.scope.bindings[name] != value:
            self.scope.bindings[name] = None
        else:
            self.scope.bindings[name] = value

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bind(node.id, None)

    def visit_Import(self, node):
        for alias in node.names:
            self.bind(
                alias.asname or alias.name.split(".")[0], "manim" if alias.name == "manim" else None
            )

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if node.module == "manim" and alias.name == "*":
                self.scope.star = True
            elif alias.name != "*":
                self.bind(
                    alias.asname or alias.name,
                    "manim." + alias.name if node.module == "manim" else None,
                )
                if node.module == "manim":
                    self.imported.add(alias.name)

    def visit_FunctionDef(self, node):
        self.bind(node.name, None)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Global(self, node):
        self.scope.globals.update(node.names)

    def visit_Nonlocal(self, node):
        self.scope.nonlocals.update(node.names)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bind(node.name, None)
        self.generic_visit(node)

    def visit_MatchAs(self, node):
        if node.name:
            self.bind(node.name, None)
        self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_Lambda(self, node):
        pass

    def visit_ListComp(self, node):
        pass

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


class ApiResolver(ast.NodeVisitor):
    def __init__(self, tree):
        self.scope = Scope(None, "module")
        self.calls: dict[ast.Call, str] = {}
        self.imported: set[str] = set()
        self._collect(tree.body)
        self.visit(tree)

    def _collect(self, body, arguments=None):
        collector = Bindings(self.scope)
        if arguments is not None:
            for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
                self.scope.bindings[argument.arg] = None
            for argument in (arguments.vararg, arguments.kwarg):
                if argument:
                    self.scope.bindings[argument.arg] = None
        for node in body:
            collector.visit(node)
        self.imported.update(collector.imported)

    def qualified(self, node):
        if isinstance(node, ast.Name):
            return self.scope.resolve(node.id)
        if isinstance(node, ast.Attribute):
            parent = self.qualified(node.value)
            return parent + "." + node.attr if parent == "manim" else None
        return None

    def visit_Call(self, node):
        name = self.qualified(node.func)
        if name and name.startswith("manim."):
            self.calls[node] = name.removeprefix("manim.")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        for expression in (*node.decorator_list, *node.args.defaults):
            self.visit(expression)
        for expression in node.args.kw_defaults:
            if expression:
                self.visit(expression)
        previous = self.scope
        self.scope = Scope(previous, "function")
        self._collect(node.body, node.args)
        for statement in node.body:
            self.visit(statement)
        self.scope = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        for expression in (*node.bases, *node.decorator_list):
            self.visit(expression)
        previous = self.scope
        self.scope = Scope(previous, "class")
        self._collect(node.body)
        for statement in node.body:
            self.visit(statement)
        self.scope = previous

    def visit_Lambda(self, node):
        previous = self.scope
        self.scope = Scope(previous, "function")
        self._collect([], node.args)
        self.visit(node.body)
        self.scope = previous

    def visit_ListComp(self, node):
        previous = self.scope
        self.visit(node.generators[0].iter)
        self.scope = Scope(previous, "function")
        for generator in node.generators:
            Bindings(self.scope).visit(generator.target)
        for generator in node.generators:
            if generator is not node.generators[0]:
                self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.scope = previous

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


def signature_errors(call: ast.Call, signature: str | None) -> list[str]:
    if not signature or any(isinstance(arg, ast.Starred) for arg in call.args):
        return []
    if any(item.arg is None for item in call.keywords):
        return []
    try:
        args = ast.parse("def _api" + signature + ":\n pass\n").body[0].args
    except (SyntaxError, ValueError):
        return []
    positional = [*args.posonlyargs, *args.args]
    allowed = {arg.arg for arg in (*args.args, *args.kwonlyargs)}
    errors = []
    if args.vararg is None and len(call.args) > len(positional):
        errors.append("位置参数超过公开签名")
    if args.kwarg is None:
        errors.extend(
            f"公开签名不接受关键字 {item.arg}" for item in call.keywords if item.arg not in allowed
        )
    bound = {arg.arg for arg in positional[: len(call.args)]}
    errors.extend(f"参数 {item.arg} 被重复传入" for item in call.keywords if item.arg in bound)
    supplied = bound | {item.arg for item in call.keywords if item.arg in allowed}
    required = positional[: len(positional) - len(args.defaults)]
    required += [
        item
        for item, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
        if default is None
    ]
    errors.extend(f"缺少必需参数 {item.arg}" for item in required if item.arg not in supplied)
    arguments = {item.arg: item.value for item in call.keywords if item.arg in allowed}
    arguments.update({item.arg: value for item, value in zip(positional, call.args, strict=False)})

    def literal_types(annotation, depth=0):
        if depth > 3:
            return None
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                parsed = ast.parse(annotation.value, mode="eval").body
            except (SyntaxError, ValueError):
                return None
            return literal_types(parsed, depth + 1)
        if isinstance(annotation, ast.Name) and annotation.id in {"float", "int", "str", "bool"}:
            return {annotation.id}
        if isinstance(annotation, ast.Constant) and annotation.value is None:
            return {"None"}
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            left = literal_types(annotation.left, depth + 1)
            right = literal_types(annotation.right, depth + 1)
            return left | right if left is not None and right is not None else None
        return None

    for parameter in (*positional, *args.kwonlyargs):
        if parameter.arg not in arguments:
            continue
        expected = literal_types(parameter.annotation)
        if expected is None:
            continue
        try:
            value = ast.literal_eval(arguments[parameter.arg])
        except (ValueError, TypeError, SyntaxError):
            continue
        actual = "None" if value is None else type(value).__name__
        if actual not in expected and not (actual == "int" and "float" in expected):
            errors.append(f"参数 {parameter.arg} 的常量类型不符合公开签名")
    return errors
