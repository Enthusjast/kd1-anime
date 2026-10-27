"""Public retrieval identity capture, with no network or SQLite provider boundary."""

from types import SimpleNamespace

import pytest

from kd1_anime.rag.clients import RagClientError
from kd1_anime.rag.models import RagRuntimeProfile
from kd1_anime.rag.service import RagService


class ProfileService(RagService):
    def __init__(self):
        self.config = SimpleNamespace(
            RAG_ENABLED=True, RAG_TOP_K=8, RAG_RERANK_TOP_N=4, RAG_MAX_CONTEXT_CHARS=12_000
        )
        self.status = {
            "enabled": True,
            "status": "active",
            "embedding_model": "fixed-embedding",
            "reranker_model": "fixed-reranker",
            "index": {
                "index_sha256": "a" * 64,
                "chunker_version": "1",
                "chunk_size": 1000,
                "chunk_overlap": 100,
            },
        }
        self.ready_calls = 0

    def runtime_status(self):
        return self.status

    def require_ready(self):
        self.ready_calls += 1


def test_runtime_profile_captures_existing_verified_index_and_all_semantic_settings():
    service = ProfileService()
    profile = service.runtime_profile()
    assert profile == RagRuntimeProfile(
        enabled=True,
        status="active",
        index_sha256="a" * 64,
        embedding_model="fixed-embedding",
        reranker_model="fixed-reranker",
        top_k=8,
        rerank_top_n=4,
        max_context_chars=12_000,
        chunker_version="1",
        chunk_size=1000,
        chunk_overlap=100,
    )
    service.status["index"]["index_sha256"] = "b" * 64
    assert profile.index_sha256 == "a" * 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("index_sha256", "b" * 64),
        ("chunker_version", "2"),
        ("chunk_size", 2000),
        ("chunk_overlap", 200),
    ],
)
def test_require_profile_rejects_changed_verified_index_identity(field, value):
    service = ProfileService()
    profile = service.runtime_profile()
    service.status["index"][field] = value
    with pytest.raises(RagClientError, match="fixed run profile"):
        service.require_profile(profile)
    assert service.ready_calls == 1


@pytest.mark.parametrize("field", ["embedding_model", "reranker_model"])
def test_require_profile_rejects_changed_model_identity(field):
    service = ProfileService()
    profile = service.runtime_profile()
    service.status[field] = "different-model"
    with pytest.raises(RagClientError, match="fixed run profile"):
        service.require_profile(profile)


def test_transient_status_change_does_not_rebind_semantic_profile():
    service = ProfileService()
    profile = service.runtime_profile()
    service.status["status"] = "degraded"
    current = service.require_profile(profile)
    assert current.status == "degraded"
    assert current.index_sha256 == profile.index_sha256


def test_disabled_profile_does_not_claim_an_active_index():
    service = ProfileService()
    service.config.RAG_ENABLED = False
    assert service.runtime_profile() == RagRuntimeProfile()
