"""基于 SQLite 的轻量向量索引。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import struct
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from kd1_anime.rag.chunker import SourceChunk, to_rag_chunk
from kd1_anime.rag.models import RagChunk, RagIndexInfo, RetrievedChunk

INDEX_SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE meta (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL
);
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY NOT NULL,
    source_path TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL
);
CREATE INDEX chunks_source_kind_idx ON chunks(source_kind);
"""


def _pack_vector(vector: Sequence[float]) -> bytes:
    if not vector:
        raise ValueError("Embedding 向量不能为空")
    values = [float(value) for value in vector]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Embedding 向量包含非有限数值")
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_vector(payload: bytes, dimension: int) -> tuple[float, ...]:
    expected_bytes = dimension * 4
    if dimension < 1 or len(payload) != expected_bytes:
        raise ValueError("索引中的 Embedding BLOB 长度与维度不一致")
    return struct.unpack(f"<{dimension}f", payload)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("查询向量与索引向量维度不一致")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _source_digest(chunks: Iterable[RagChunk]) -> str:
    values = sorted({(chunk.source_path, chunk.source_sha256) for chunk in chunks})
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _index_digest(
    chunks: Iterable[RagChunk],
    vectors: Iterable[bytes],
    *,
    embedding_model: str,
    dimension: int,
) -> str:
    pairs = sorted(zip(chunks, vectors, strict=True), key=lambda item: item[0].chunk_id)
    digest = hashlib.sha256()
    digest.update(embedding_model.encode())
    digest.update(str(dimension).encode("ascii"))
    for chunk, vector in pairs:
        digest.update(chunk.chunk_id.encode("ascii"))
        digest.update(chunk.content_sha256.encode("ascii"))
        digest.update(
            json.dumps(
                chunk.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(vector)
    return digest.hexdigest()


def _chunk_id(source_sha256: str, source_kind: str, ordinal: int, content_sha256: str) -> str:
    return hashlib.sha256(
        f"{source_sha256}:{source_kind}:{ordinal}:{content_sha256}".encode()
    ).hexdigest()


class RagIndex:
    """可原子替换的本地 Embedding 索引。"""

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    @classmethod
    def build(
        cls,
        path: Path,
        chunks: Sequence[SourceChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        embedding_model: str,
    ) -> RagIndexInfo:
        if not chunks:
            raise ValueError("没有可写入索引的知识库分块")
        if not embeddings:
            raise ValueError("没有可写入索引的 Embedding")
        if not embedding_model.strip():
            raise ValueError("Embedding 模型名不能为空")
        if len(chunks) != len(embeddings):
            raise ValueError("知识库分块数量与 Embedding 数量不一致")
        rag_chunks = [to_rag_chunk(chunk) for chunk in chunks]
        vectors = [_pack_vector(vector) for vector in embeddings]
        dimension = len(embeddings[0])
        if dimension < 1 or any(len(vector) != dimension for vector in embeddings):
            raise ValueError("知识库 Embedding 维度不一致")
        source_sha256 = _source_digest(rag_chunks)
        index_sha256 = _index_digest(
            rag_chunks,
            vectors,
            embedding_model=embedding_model,
            dimension=dimension,
        )

        destination = path.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.parent.chmod(0o700)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with sqlite3.connect(temporary) as connection:
                connection.executescript(_SCHEMA)
                metadata = {
                    "schema_version": str(INDEX_SCHEMA_VERSION),
                    "index_sha256": index_sha256,
                    "source_sha256": source_sha256,
                    "embedding_model": embedding_model,
                    "embedding_dimension": str(dimension),
                    "chunk_count": str(len(rag_chunks)),
                }
                connection.executemany(
                    "INSERT INTO meta(key, value) VALUES (?, ?)", metadata.items()
                )
                connection.executemany(
                    """
                    INSERT INTO chunks(
                        chunk_id, source_path, source_kind, source_sha256,
                        content_sha256, ordinal, text, metadata_json,
                        embedding, embedding_dimension
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.chunk_id,
                            chunk.source_path,
                            chunk.source_kind,
                            chunk.source_sha256,
                            chunk.content_sha256,
                            chunk.ordinal,
                            chunk.text,
                            json.dumps(chunk.metadata, ensure_ascii=False, separators=(",", ":")),
                            vector,
                            dimension,
                        )
                        for chunk, vector in zip(rag_chunks, vectors, strict=True)
                    ],
                )
                connection.commit()
            temporary.chmod(0o600)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return cls(destination).info()
        finally:
            if temporary.exists():
                temporary.unlink()

    def info(self) -> RagIndexInfo:
        if not self.exists:
            raise FileNotFoundError(f"RAG 索引不存在: {self.path}")
        try:
            with sqlite3.connect(self.path) as connection:
                rows = dict(connection.execute("SELECT key, value FROM meta"))
        except sqlite3.Error as exc:
            raise ValueError(f"RAG 索引无法读取: {self.path}") from exc
        try:
            return RagIndexInfo(
                schema_version=int(rows["schema_version"]),
                index_path=str(self.path),
                index_sha256=rows["index_sha256"],
                source_sha256=rows["source_sha256"],
                embedding_model=rows["embedding_model"],
                embedding_dimension=int(rows["embedding_dimension"]),
                chunk_count=int(rows["chunk_count"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"RAG 索引元数据不完整: {self.path}") from exc

    def verify_integrity(self) -> RagIndexInfo:
        """校验元数据和所有分块/向量内容，拒绝被修改的索引。"""

        info = self.info()
        chunks: list[RagChunk] = []
        vectors: list[bytes] = []
        try:
            with sqlite3.connect(self.path) as connection:
                rows = connection.execute(
                    """
                    SELECT chunk_id, source_path, source_kind, source_sha256,
                           content_sha256, ordinal, text, metadata_json,
                           embedding, embedding_dimension
                    FROM chunks ORDER BY chunk_id
                    """
                )
                for row in rows:
                    content_sha256 = hashlib.sha256(str(row[6]).encode("utf-8")).hexdigest()
                    if content_sha256 != row[4]:
                        raise ValueError("分块内容哈希不一致")
                    if _chunk_id(row[3], row[2], int(row[5]), content_sha256) != row[0]:
                        raise ValueError("分块 ID 与内容哈希不一致")
                    chunks.append(
                        RagChunk(
                            chunk_id=row[0],
                            source_path=row[1],
                            source_kind=row[2],
                            source_sha256=row[3],
                            content_sha256=row[4],
                            ordinal=int(row[5]),
                            text=row[6],
                            metadata=json.loads(row[7]),
                        )
                    )
                    vectors.append(_pack_vector(_unpack_vector(row[8], int(row[9]))))
        except (sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"RAG 索引完整性校验失败: {self.path}: {exc}") from exc
        if len(chunks) != info.chunk_count:
            raise ValueError(f"RAG 索引分块数量不一致: {len(chunks)} != {info.chunk_count}")
        if _source_digest(chunks) != info.source_sha256:
            raise ValueError("RAG 索引源文件哈希不一致")
        actual_digest = _index_digest(
            chunks,
            vectors,
            embedding_model=info.embedding_model,
            dimension=info.embedding_dimension,
        )
        if actual_digest != info.index_sha256:
            raise ValueError("RAG 索引内容哈希不一致")
        return info

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        source_kinds: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        info = self.verify_integrity()
        query = [float(value) for value in query_embedding]
        if len(query) != info.embedding_dimension:
            raise ValueError(f"查询向量维度不一致: {len(query)} != {info.embedding_dimension}")
        if any(not math.isfinite(value) for value in query):
            raise ValueError("查询向量包含非有限数值")
        candidates: list[RetrievedChunk] = []
        try:
            with sqlite3.connect(self.path) as connection:
                rows = connection.execute(
                    """
                    SELECT chunk_id, source_path, source_kind, source_sha256,
                           content_sha256, ordinal, text, metadata_json,
                           embedding, embedding_dimension
                    FROM chunks
                    """
                )
                for row in rows:
                    if source_kinds is not None and row[2] not in source_kinds:
                        continue
                    vector = _unpack_vector(row[8], int(row[9]))
                    chunk = RagChunk(
                        chunk_id=row[0],
                        source_path=row[1],
                        source_kind=row[2],
                        source_sha256=row[3],
                        content_sha256=row[4],
                        ordinal=int(row[5]),
                        text=row[6],
                        metadata=json.loads(row[7]),
                    )
                    candidates.append(
                        RetrievedChunk(chunk=chunk, score=_cosine_similarity(query, vector))
                    )
        except (sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and "维度" in str(exc):
                raise
            raise ValueError(f"RAG 索引检索失败: {self.path}") from exc
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]
