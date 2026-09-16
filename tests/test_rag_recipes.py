"""本地匿名配方存储与增量索引测试。"""

from pathlib import Path

from kd1_anime.config import Settings
from kd1_anime.rag.recipes import RecipeStore, anonymize_code
from kd1_anime.rag.service import RagService


def test_anonymize_code_removes_run_identity_paths_and_secret_lines():
    code = """from manim import *
API_KEY = 'do-not-save'
# workspace/runs/20260829-210659-abcdef12
token = 'sk-1234567890abcdef'
scene_file = '/home/user/workspace/runs/20260829-210659-abcdef12/scene.py'
class Demo(Scene):
    def construct(self):
        self.wait()
"""

    safe = anonymize_code(code)

    assert "do-not-save" not in safe
    assert "sk-1234567890abcdef" not in safe
    assert "20260829-210659-abcdef12" not in safe
    assert "/home/user" not in safe
    assert "class Demo(Scene)" in safe


def test_recipe_store_writes_private_deduplicated_markdown(tmp_path: Path):
    store = RecipeStore(tmp_path / "recipes")
    code = (
        "from manim import *\nclass Demo(Scene):\n    def construct(self):\n        self.wait()\n"
    )

    first, path, created = store.save(
        code,
        renderer="cairo",
        semantic_intent="show a stable object",
        object_kinds=["text"],
        capabilities=["MathTex"],
        verification="visual_pass",
    )
    second, second_path, second_created = store.save(
        code,
        renderer="cairo",
        semantic_intent="same recipe",
    )

    assert created is True
    assert second_created is False
    assert path == second_path
    assert first.recipe_id == second.recipe_id
    assert path.suffix == ".md"
    assert path.stat().st_mode & 0o777 == 0o600
    assert "Anonymous animation recipe" in path.read_text(encoding="utf-8")
    assert "same recipe" not in path.read_text(encoding="utf-8")


def test_refresh_recipe_reembeds_only_new_chunks(tmp_path: Path, monkeypatch):
    recipes = tmp_path / "recipes"
    config = Settings(
        _env_file=None,
        RAG_ENABLED=True,
        RAG_INDEX_PATH=tmp_path / "index.sqlite3",
        RAG_DOCS_DIR=None,
        RAG_EXAMPLES_DIR=None,
        RAG_RECIPES_DIR=recipes,
        RAG_EMBEDDING_BASE_URL="https://embedding.invalid/v1",
        RAG_EMBEDDING_MODEL="embed-test",
        RAG_RERANK_BASE_URL="https://rerank.invalid/v1",
        RAG_RERANK_MODEL="rerank-test",
    )
    store = RecipeStore(recipes)
    _, first_path, _ = store.save(
        "from manim import *\nclass A(Scene):\n    def construct(self):\n        self.wait()\n",
        renderer="cairo",
    )
    service = RagService(config)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        service.embedding,
        "embed",
        lambda texts: calls.append(list(texts)) or [[1.0, 0.0] for _ in texts],
    )
    service.build_index()
    calls.clear()

    _, second_path, _ = store.save(
        "from manim import *\nclass B(Scene):\n    def construct(self):\n        self.wait(1)\n",
        renderer="cairo",
    )
    result = service.refresh_recipe(second_path)

    assert first_path != second_path
    assert result.chunk_count == 2
    assert len(calls) == 1
    assert len(calls[0]) == 1


def test_anonymized_scene_preserves_python_indentation():
    import ast

    code = (
        "from manim import *\nclass Demo(Scene):\n    def construct(self):\n        self.wait()\n"
    )
    ast.parse(anonymize_code(code))
    assert "        self.wait()" in anonymize_code(code)
