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
from dataclasses import dataclass
from pathlib import Path

from kd1_anime.rag.chunker import (
    CHUNKER_VERSION,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    SourceChunk,
    chunk_id_for,
    to_rag_chunk,
)
from kd1_anime.rag.models import RagChunk, RagIndexInfo, RetrievedChunk

# v4 的 chunk_id 包含 source_path；v3 及更早索引必须重建。
INDEX_SCHEMA_VERSION = 4
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
    chunker_version: str = CHUNKER_VERSION,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> str:
    pairs = sorted(zip(chunks, vectors, strict=True), key=lambda item: item[0].chunk_id)
    digest = hashlib.sha256()
    digest.update(embedding_model.encode())
    digest.update(str(dimension).encode("ascii"))
    digest.update(chunker_version.encode("utf-8"))
    digest.update(str(chunk_size).encode("ascii"))
    digest.update(str(chunk_overlap).encode("ascii"))
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


def _chunk_id_with_source_path(
    source_path: str,
    source_sha256: str,
    source_kind: str,
    ordinal: int,
    content_sha256: str,
) -> str:
    return chunk_id_for(source_path, source_sha256, source_kind, ordinal, content_sha256)


@dataclass(frozen=True, slots=True)
class VerifiedIndexSnapshot:
    """完成完整性校验后可复用的文本块和解包向量。"""

    info: RagIndexInfo
    chunks: tuple[RagChunk, ...]
    vectors: tuple[tuple[float, ...], ...]


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
        source_docs_dir: Path | None = None,
        source_examples_dir: Path | None = None,
        source_recipes_dir: Path | None = None,
        chunker_version: str = CHUNKER_VERSION,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> RagIndexInfo:
        if not chunks:
            raise ValueError("没有可写入索引的知识库分块")
        if not embeddings:
            raise ValueError("没有可写入索引的 Embedding")
        if not embedding_model.strip():
            raise ValueError("Embedding 模型名不能为空")
        if (
            not chunker_version.strip()
            or chunk_size < 100
            or chunk_overlap < 0
            or chunk_overlap >= chunk_size
        ):
            raise ValueError("RAG chunker 配置无效")
        if len(chunks) != len(embeddings):
            raise ValueError("知识库分块数量与 Embedding 数量不一致")
        rag_chunks = [to_rag_chunk(chunk) for chunk in chunks]
        seen_chunk_ids: dict[str, RagChunk] = {}
        for chunk in rag_chunks:
            previous = seen_chunk_ids.get(chunk.chunk_id)
            if previous is not None:
                raise ValueError(
                    "知识库输入包含重复 chunk_id："
                    f"{chunk.chunk_id}（来源 {previous.source_path} 与 {chunk.source_path}）"
                )
            seen_chunk_ids[chunk.chunk_id] = chunk
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
            chunker_version=chunker_version,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
                    "source_docs_dir": str(source_docs_dir or ""),
                    "source_examples_dir": str(source_examples_dir or ""),
                    "source_recipes_dir": str(source_recipes_dir or ""),
                    "chunker_version": chunker_version,
                    "chunk_size": str(chunk_size),
                    "chunk_overlap": str(chunk_overlap),
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
                source_docs_dir=rows.get("source_docs_dir", ""),
                source_examples_dir=rows.get("source_examples_dir", ""),
                source_recipes_dir=rows.get("source_recipes_dir", ""),
                chunker_version=rows.get("chunker_version", ""),
                chunk_size=int(rows.get("chunk_size", 0)),
                chunk_overlap=int(rows.get("chunk_overlap", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"RAG 索引元数据不完整: {self.path}") from exc

    def load_verified(self) -> VerifiedIndexSnapshot:
        """校验并加载索引，返回可复用的文本块和解包向量。"""

        info = self.info()
        if info.schema_version != INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"RAG 索引 schema 版本不兼容: {info.schema_version} != {INDEX_SCHEMA_VERSION}"
            )
        chunks: list[RagChunk] = []
        vectors: list[bytes] = []
        decoded_vectors: list[tuple[float, ...]] = []
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
                    row_dimension = int(row[9])
                    if row_dimension != info.embedding_dimension:
                        raise ValueError(
                            "分块 Embedding 维度与索引元数据不一致: "
                            f"{row_dimension} != {info.embedding_dimension}"
                        )
                    content_sha256 = hashlib.sha256(str(row[6]).encode("utf-8")).hexdigest()
                    if content_sha256 != row[4]:
                        raise ValueError("分块内容哈希不一致")
                    if (
                        _chunk_id_with_source_path(
                            row[1], row[3], row[2], int(row[5]), content_sha256
                        )
                        != row[0]
                    ):
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
                    vector = _unpack_vector(row[8], row_dimension)
                    vectors.append(_pack_vector(vector))
                    decoded_vectors.append(vector)
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
            chunker_version=info.chunker_version,
            chunk_size=info.chunk_size,
            chunk_overlap=info.chunk_overlap,
        )
        if actual_digest != info.index_sha256:
            raise ValueError("RAG 索引内容哈希不一致")
        return VerifiedIndexSnapshot(
            info=info,
            chunks=tuple(chunks),
            vectors=tuple(decoded_vectors),
        )

    def verify_integrity(self) -> RagIndexInfo:
        """校验元数据和所有分块/向量内容，拒绝被修改的索引。"""

        return self.load_verified().info

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        source_kinds: set[str] | None = None,
        exclude_frameworks: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        snapshot = self.load_verified()
        return self._search_verified(
            query_embedding,
            top_k=top_k,
            source_kinds=source_kinds,
            exclude_frameworks=exclude_frameworks,
            info=snapshot.info,
            snapshot=snapshot,
        )

    def search_verified(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        source_kinds: set[str] | None = None,
        exclude_frameworks: set[str] | None = None,
        info: RagIndexInfo,
        snapshot: VerifiedIndexSnapshot | None = None,
    ) -> list[RetrievedChunk]:
        """使用调用方刚完成的完整性校验结果检索。"""

        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        if Path(info.index_path).expanduser().resolve() != self.path:
            raise ValueError("复用的 RAG 索引元数据与当前索引路径不一致")
        # ``RagService`` 在进入这里前已经完成全量校验，但索引可以被另一个
        # 进程通过 os.replace 原子重建。再次读取轻量 meta，避免把旧收据绑定
        # 到已经替换的新文件；完整内容扫描只做一次。
        current_info = self.info()
        if (
            current_info.index_sha256 != info.index_sha256
            or current_info.source_sha256 != info.source_sha256
            or current_info.embedding_model != info.embedding_model
            or current_info.embedding_dimension != info.embedding_dimension
            or current_info.chunk_count != info.chunk_count
            or current_info.chunker_version != info.chunker_version
            or current_info.chunk_size != info.chunk_size
            or current_info.chunk_overlap != info.chunk_overlap
        ):
            raise ValueError("RAG 索引在校验后发生变化，请重试检索")
        return self._search_verified(
            query_embedding,
            top_k=top_k,
            source_kinds=source_kinds,
            exclude_frameworks=exclude_frameworks,
            info=info,
            snapshot=snapshot,
        )

    def _search_verified(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        source_kinds: set[str] | None,
        exclude_frameworks: set[str] | None,
        info: RagIndexInfo,
        snapshot: VerifiedIndexSnapshot | None = None,
    ) -> list[RetrievedChunk]:
        # RagService 在同一次查询前已经完成完整性校验；这里复用结果，避免
        # 对较大的 SQLite 知识库重复扫描全部分块和向量。
        query = [float(value) for value in query_embedding]
        if len(query) != info.embedding_dimension:
            raise ValueError(f"查询向量维度不一致: {len(query)} != {info.embedding_dimension}")
        if any(not math.isfinite(value) for value in query):
            raise ValueError("查询向量包含非有限数值")
        candidates: list[RetrievedChunk] = []

        def append_candidate(chunk: RagChunk, vector: Sequence[float]) -> None:
            if source_kinds is not None and chunk.source_kind not in source_kinds:
                return
            if (
                exclude_frameworks is not None
                and chunk.metadata.get("framework", "") in exclude_frameworks
            ):
                return
            candidates.append(RetrievedChunk(chunk=chunk, score=_cosine_similarity(query, vector)))

        try:
            if snapshot is not None:
                if snapshot.info.index_sha256 != info.index_sha256:
                    raise ValueError("RAG 索引快照与元数据不一致")
                for chunk, vector in zip(snapshot.chunks, snapshot.vectors, strict=True):
                    append_candidate(chunk, vector)
            else:
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
                        append_candidate(chunk, vector)
        except (sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and "维度" in str(exc):
                raise
            raise ValueError(f"RAG 索引检索失败: {self.path}") from exc
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]
