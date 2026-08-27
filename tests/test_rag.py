"""RAG 索引、服务客户端和流水线降级行为测试。"""

from pathlib import Path

import pytest

from kd1_anime.config import Settings
from kd1_anime.rag import RagService
from kd1_anime.rag.chunker import SourceChunk, chunk_file, iter_source_files
from kd1_anime.rag.clients import EmbeddingClient, RagClientError, RerankerClient
from kd1_anime.rag.store import RagIndex


def _config(tmp_path: Path, **overrides) -> Settings:
    values = {
        "RAG_ENABLED": True,
        "RAG_INDEX_PATH": tmp_path / "index.sqlite3",
        "RAG_DOCS_DIR": tmp_path / "docs",
        "RAG_EXAMPLES_DIR": tmp_path / "examples",
        "RAG_EMBEDDING_BASE_URL": "https://embedding.invalid/v1",
        "RAG_EMBEDDING_MODEL": "embed-test",
        "RAG_RERANK_BASE_URL": "https://rerank.invalid/v1",
        "RAG_RERANK_MODEL": "rerank-test",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _source_chunk(text: str, ordinal: int = 0) -> SourceChunk:
    return SourceChunk(
        path=Path("example.py"),
        source_kind="example",
        source_sha256="a" * 64,
        ordinal=ordinal,
        text=text,
        metadata={"suffix": ".py"},
    )


def test_chunker_splits_python_and_removes_sensitive_lines(tmp_path):
    source = tmp_path / "example.py"
    source.write_text(
        "API_KEY = 'do-not-index'\n\n"
        "def make_circle():\n    return Circle()\n\n"
        "class Demo:\n    pass\n",
        encoding="utf-8",
    )

    chunks = chunk_file(source, "example", chunk_size=100, overlap=10)

    assert chunks
    assert all("do-not-index" not in chunk.text for chunk in chunks)
    assert any("make_circle" in chunk.text for chunk in chunks)


def test_iter_source_files_excludes_runtime_and_unknown_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "api.md").write_text("# API", encoding="utf-8")
    (docs / "secrets.env").write_text("TOKEN=x", encoding="utf-8")
    runtime = docs / "workspace"
    runtime.mkdir()
    (runtime / "scene.py").write_text("from manim import *", encoding="utf-8")
    (docs / "image.png").write_bytes(b"not indexed")
    outside = tmp_path / "outside.py"
    outside.write_text("secret = True", encoding="utf-8")
    (docs / "outside-link.py").symlink_to(outside)

    files = iter_source_files(docs, None)

    assert [(path.name, kind) for path, kind in files] == [("api.md", "manim_doc")]


def test_embedding_client_reorders_and_validates_batch(monkeypatch):
    from kd1_anime.rag import clients

    monkeypatch.setattr(
        clients,
        "_post_json",
        lambda *args, **kwargs: {
            "data": [
                {"index": 1, "embedding": [0, 1]},
                {"index": 0, "embedding": [1, 0]},
            ]
        },
    )
    client = EmbeddingClient(
        api_key="key",
        base_url="https://embedding.invalid/v1/",
        model="embed",
    )

    assert client.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_embedding_client_rejects_wrong_count(monkeypatch):
    from kd1_anime.rag import clients

    monkeypatch.setattr(clients, "_post_json", lambda *args, **kwargs: {"data": []})
    client = EmbeddingClient(
        api_key="",
        base_url="https://embedding.invalid/v1",
        model="embed",
    )

    with pytest.raises(RagClientError, match="数量不一致"):
        client.embed(["a"])


def test_reranker_client_parses_cohere_response(monkeypatch):
    from kd1_anime.rag import clients

    monkeypatch.setattr(
        clients,
        "_post_json",
        lambda *args, **kwargs: {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ]
        },
    )
    client = RerankerClient(
        api_key="key",
        base_url="https://rerank.invalid/v1",
        model="rerank",
    )

    assert client.rerank("query", ["one", "two"], top_n=2) == [(1, 0.9), (0, 0.2)]


def test_rag_index_builds_atomic_sqlite_index_and_searches(tmp_path):
    chunks = [_source_chunk("circle api", 0), _source_chunk("square api", 1)]
    path = tmp_path / "nested" / "index.sqlite3"

    info = RagIndex.build(path, chunks, [[1, 0], [0, 1]], embedding_model="embed")
    results = RagIndex(path).search([1, 0], top_k=2)

    assert info.chunk_count == 2
    assert info.embedding_dimension == 2
    assert path.stat().st_mode & 0o777 == 0o600
    assert results[0].chunk.text == "circle api"
    assert results[0].score == pytest.approx(1.0)


def test_rag_index_rejects_tampered_chunk(tmp_path):
    import sqlite3

    path = tmp_path / "index.sqlite3"
    RagIndex.build(path, [_source_chunk("original")], [[1, 0]], embedding_model="embed")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE chunks SET text = 'tampered'")
        connection.commit()

    with pytest.raises(ValueError, match="哈希不一致"):
        RagIndex(path).verify_integrity()


def test_rag_service_returns_context_and_receipt(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.RAG_DOCS_DIR.mkdir()
    (config.RAG_DOCS_DIR / "api.md").write_text("# Circle API\nUse Circle().", encoding="utf-8")
    service = RagService(config)
    monkeypatch.setattr(service.embedding, "embed", lambda texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(service.reranker, "rerank", lambda *args, **kwargs: [(0, 0.95)])
    service.build_index()

    result = service.search("Circle API", stage="coder")

    assert result.receipt.status == "active"
    assert result.receipt.index_sha256
    assert result.receipt.query_sha256
    assert result.receipt.code_sha256 == ""
    assert "Circle API" in result.context
    assert str(config.RAG_DOCS_DIR) not in result.context
    assert result.receipt.chunks[0].content_sha256


def test_rag_service_accepts_relative_source_directory(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "api.md").write_text("# API", encoding="utf-8")
    config = _config(tmp_path, RAG_DOCS_DIR=Path("docs"), RAG_EXAMPLES_DIR=None)
    service = RagService(config)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(service.embedding, "embed", lambda texts: [[1.0, 0.0] for _ in texts])

    result = service.build_index()

    assert result.chunk_count == 1


def test_rag_service_falls_back_when_reranker_fails(tmp_path, monkeypatch):
    config = _config(tmp_path)
    service = RagService(config)
    path = tmp_path / "index.sqlite3"
    RagIndex.build(
        path, [_source_chunk("fallback reference")], [[1, 0]], embedding_model="embed-test"
    )
    monkeypatch.setattr(service.embedding, "embed", lambda texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(
        service.reranker,
        "rerank",
        lambda *args, **kwargs: (_ for _ in ()).throw(RagClientError("offline")),
    )

    result = service.search("reference", stage="fix")

    assert result.receipt.status == "degraded"
    assert result.chunks
    assert "初排" in result.receipt.warning


def test_rag_service_caches_active_query_result(tmp_path, monkeypatch):
    config = _config(tmp_path)
    service = RagService(config)
    RagIndex.build(
        config.RAG_INDEX_PATH,
        [_source_chunk("cached reference")],
        [[1, 0]],
        embedding_model="embed-test",
    )
    calls = []
    monkeypatch.setattr(
        service.embedding,
        "embed",
        lambda texts: calls.append(list(texts)) or [[1.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(service.reranker, "rerank", lambda *args, **kwargs: [(0, 0.9)])

    first = service.search("cached", stage="code")
    second = service.search("cached", stage="code")

    assert first == second
    assert len(calls) == 1


def test_rag_service_rejects_index_from_different_embedding_model(tmp_path, monkeypatch):
    config = _config(tmp_path)
    service = RagService(config)
    RagIndex.build(
        config.RAG_INDEX_PATH,
        [_source_chunk("old model reference")],
        [[1, 0]],
        embedding_model="old-model",
    )
    monkeypatch.setattr(service.embedding, "embed", lambda texts: [[1.0, 0.0] for _ in texts])

    result = service.search("reference", stage="code")

    assert result.receipt.status == "degraded"
    assert "不同的 Embedding 模型" in result.receipt.warning


def test_disabled_rag_does_not_need_index_or_network(tmp_path):
    config = Settings(_env_file=None, RAG_INDEX_PATH=tmp_path / "missing.sqlite3")
    service = RagService(config)

    result = service.search("anything", stage="outline")

    assert result.receipt.status == "disabled"
    assert result.context == ""


def test_rag_service_migrates_legacy_default_index(tmp_path, monkeypatch):
    import kd1_anime.rag.service as service_module

    legacy_index = tmp_path / "legacy" / "index.sqlite3"
    current_index = tmp_path / "current" / "index.sqlite3"
    RagIndex.build(
        legacy_index,
        [_source_chunk("legacy reference")],
        [[1, 0]],
        embedding_model="embed-test",
    )
    monkeypatch.setattr(service_module, "DEFAULT_RAG_INDEX_PATH", current_index)
    monkeypatch.setattr(service_module, "LEGACY_RAG_INDEX_PATH", legacy_index)

    service = RagService(Settings(_env_file=None, RAG_INDEX_PATH=current_index))

    assert current_index.is_file()
    assert current_index.stat().st_mode & 0o777 == 0o600
    assert service.runtime_status()["index"]["chunk_count"] == 1


def test_rag_configuration_is_separate_from_main_and_visual_profiles(tmp_path):
    config = Settings(
        _env_file=None,
        LLM_API_KEY="main-key",
        LLM_BASE_URL="https://main.invalid/v1",
        LLM_MODEL="main-model",
        VISUAL_LLM_API_KEY="visual-key",
        VISUAL_LLM_BASE_URL="https://visual.invalid/v1",
        VISUAL_LLM_MODEL="visual-model",
        RAG_EMBEDDING_API_KEY="embedding-key",
        RAG_EMBEDDING_BASE_URL="https://embedding.invalid/v1",
        RAG_EMBEDDING_MODEL="embedding-model",
        RAG_RERANK_API_KEY="rerank-key",
        RAG_RERANK_BASE_URL="https://rerank.invalid/v1",
        RAG_RERANK_MODEL="rerank-model",
        RAG_INDEX_PATH=tmp_path / "index.sqlite3",
    )
    service = RagService(config)

    assert service.embedding.api_key == "embedding-key"
    assert service.embedding.base_url != config.LLM_BASE_URL
    assert service.reranker.api_key == "rerank-key"


def test_rag_service_uses_process_shared_semaphore():
    from kd1_anime.resources import ResourceCoordinator

    resources = ResourceCoordinator(llm_limit=1, slurm_limit=0, rag_limit=3)
    service = RagService()
    service_with_shared_limit = RagService(rag_semaphore=resources.rag)

    assert service._semaphore is not resources.rag
    assert service_with_shared_limit._semaphore is resources.rag
