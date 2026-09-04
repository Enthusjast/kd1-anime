from pathlib import Path

import pytest

from kd1_anime.agents.validator import validate_manim_code

CORPUS = Path(__file__).parent / "fixtures" / "recipe_corpus"
EXPECTED_CLASSES = {
    "formula.py": "FormulaRecipe",
    "graph.py": "GraphRecipe",
    "geometry.py": "GeometryRecipe",
    "three_d.py": "ThreeDRecipe",
    "updater.py": "UpdaterRecipe",
    "camera.py": "CameraRecipe",
}


@pytest.mark.parametrize("filename,class_name", EXPECTED_CLASSES.items())
def test_recipe_corpus_has_one_valid_scene(filename, class_name):
    code = (CORPUS / filename).read_text(encoding="utf-8")
    result = validate_manim_code(code, renderer="cairo")
    assert result.is_valid, result.feedback
    assert result.scene_classes == [class_name]
