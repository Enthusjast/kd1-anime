"""Content fidelity and explicit fallback contracts for the optional compiler."""

import ast

import pytest
from pydantic import ValidationError

from kd1_anime.domain.program import SceneProgram
from kd1_anime.eval.benchmark import fixed_tasks, task_plan
from kd1_anime.services.program_compiler import (
    ProgramCompileError,
    compile_program,
    expression_source,
)


def task(identifier):
    return next(item for item in fixed_tasks() if item.task_id == identifier)


def compile_task(identifier, *, objects=None, events=None):
    selected = task(identifier)
    raw = dict(selected.program)
    if objects is not None:
        raw["objects"] = objects
    if events is not None:
        raw["events"] = events
    return compile_program(SceneProgram.model_validate(raw), task_plan(selected))


def calls(code, name):
    return [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


@pytest.mark.parametrize(
    "identifier",
    [
        "formula_quadratic",
        "formula_triangle",
        "formula_limit",
        "formula_series",
        "unicode_text",
        "unicode_tex",
    ],
)
def test_compiler_preserves_formula_and_cjk_text_exactly(identifier):
    selected = task(identifier)
    obj = selected.program["objects"][0]
    code = compile_task(identifier)
    assert ast.literal_eval(calls(code, obj["kind"])[0].args[0]) == obj["text"]
    if obj["kind"] in {"Tex", "MathTex"}:
        assert 'tex_compiler="xelatex"' in code
        assert 'output_format=".xdv"' in code
        assert r"\usepackage{ctex}" in code
        assert "config.tex_template = tex_template" in code


def test_polygon_vertices_and_translation_preserve_absolute_geometry():
    selected = task("geometry_triangle")
    obj = {**selected.program["objects"][0], "position": [1, -2, 0]}
    code = compile_task(selected.task_id, objects=[obj])
    polygon = calls(code, "Polygon")[0]
    assert [ast.literal_eval(arg) for arg in polygon.args] == obj["vertices"]
    assignment = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_kd1_object_0" for target in node.targets
        )
    )
    assert assignment.value.func.attr == "shift"
    assert ast.literal_eval(assignment.value.args[0]) == [1, -2, 0]


def test_segment_endpoints_are_not_trimmed_by_default_buff():
    code = compile_task("geometry_segment")
    line = calls(code, "Line")[0]
    assert [ast.literal_eval(arg) for arg in line.args] == [[-2, 1, 0], [2, -1, 0]]
    assert next(ast.literal_eval(arg.value) for arg in line.keywords if arg.arg == "buff") == 0


@pytest.mark.parametrize(
    "identifier, name, expected",
    [
        ("geometry_circle", "Circle", {"radius": 1.75}),
        ("geometry_rectangle", "Rectangle", {"width": 3, "height": 1.5}),
    ],
)
def test_geometry_dimensions_are_explicit_and_unchanged(identifier, name, expected):
    constructor = calls(compile_task(identifier), name)[0]
    values = {
        item.arg: ast.literal_eval(item.value)
        for item in constructor.keywords
        if item.arg in expected
    }
    assert values == expected


def test_function_domain_and_axis_ranges_are_preserved():
    selected = task("graph_reciprocal")
    code = compile_task(selected.task_id)
    axes = calls(code, "Axes")[0]
    assert next(
        ast.literal_eval(item.value) for item in axes.keywords if item.arg == "x_range"
    ) == [-4, 4, 1]
    plot = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "plot"
    )
    assert next(
        ast.literal_eval(item.value) for item in plot.keywords if item.arg == "x_range"
    ) == [1, 4]
    body = plot.args[0].body
    assert isinstance(body, ast.BinOp) and isinstance(body.op, ast.Div)
    assert ast.literal_eval(body.left) == 1
    assert body.right.id == "x"


def test_event_markers_gaps_and_total_duration_are_exact():
    code = compile_task(
        "geometry_circle",
        events=[
            {
                "event_id": "reveal_circle",
                "operation": "create",
                "start_seconds": 0.5,
                "end_seconds": 1.75,
                "source_ids": ["subject"],
            },
            {
                "event_id": "hold_circle",
                "operation": "wait",
                "start_seconds": 2,
                "end_seconds": 3.25,
            },
        ],
    )
    tree = ast.parse(code)
    durations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "wait":
            durations.append(ast.literal_eval(node.args[0]))
        elif node.func.attr == "play":
            durations.append(
                next(
                    ast.literal_eval(item.value) for item in node.keywords if item.arg == "run_time"
                )
            )
    assert durations == [0.5, 1.25, 0.25, 1.25, 0.75]
    assert sum(durations) == 4
    assert code.index("KD1_ANIMATION_EVENT: reveal_circle") < code.index(
        "KD1_ANIMATION_EVENT: hold_circle"
    )


def test_exported_replacement_matches_the_final_object_identity():
    code = compile_task("continuity_replace")
    replacement = calls(code, "ReplacementTransform")[0]
    assert [node.id for node in replacement.args] == ["_kd1_object_0", "subject"]
    export = code.split("# KD1_CONTINUITY_EXPORT_BEGIN", 1)[1].split(
        "# KD1_CONTINUITY_EXPORT_END", 1
    )[0]
    assert "element_id: subject" in export
    assert 'MathTex("2x"' in export
    assert 'MathTex("x+x"' not in export


@pytest.mark.parametrize(
    "updates", [{"target_ids": ["subject"]}, {"vector": [1, 2, 0]}, {"factor": 2}, {"angle": 1}]
)
def test_compiler_never_silently_drops_animation_parameters(updates):
    with pytest.raises(ProgramCompileError, match="Unexpected"):
        compile_task(
            "geometry_circle",
            events=[
                {
                    "event_id": "reveal",
                    "operation": "create",
                    "start_seconds": 0,
                    "end_seconds": 2,
                    "source_ids": ["subject"],
                    **updates,
                }
            ],
        )


def test_missing_geometry_parameter_falls_back_instead_of_inventing_radius():
    from copy import deepcopy

    program = deepcopy(task("geometry_circle").program)
    program["objects"][0].pop("radius")
    with pytest.raises(ValidationError):
        SceneProgram.model_validate(program)


@pytest.mark.parametrize(
    "expression",
    ["open('secret')", "np.sin(x)", "x[0]", "lambda: x", "x if x else 0", "sin(x, where=True)"],
)
def test_expression_language_does_not_accept_arbitrary_python(expression):
    with pytest.raises(ProgramCompileError):
        expression_source(expression)


def test_mutated_export_requires_explicit_final_boundary():
    with pytest.raises(ProgramCompileError, match="final boundary"):
        compile_task(
            "continuity_export",
            events=[
                {
                    "event_id": "in",
                    "operation": "create",
                    "start_seconds": 0,
                    "end_seconds": 1,
                    "source_ids": ["subject"],
                },
                {
                    "event_id": "move",
                    "operation": "shift",
                    "start_seconds": 1,
                    "end_seconds": 2,
                    "source_ids": ["subject"],
                    "vector": [1, 0, 0],
                },
            ],
        )
