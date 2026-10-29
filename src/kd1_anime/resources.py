"""Process-wide model/RAG pools and live render-attempt capacity shared by runtimes."""

from __future__ import annotations

import threading


class ResourceCoordinator:
    def __init__(
        self,
        *,
        llm_limit: int,
        slurm_limit: int,
        local_limit: int | None = None,
        visual_llm_limit: int | None = None,
        rag_limit: int | None = None,
    ) -> None:
        self.llm = threading.Semaphore(max(1, llm_limit))
        # 视觉模型使用独立端点和并发池。批处理中的多个运行共享
        # 此信号量，避免每个任务各自达到上限后把视觉服务瞬间打满。
        if visual_llm_limit is None:
            visual_llm_limit = llm_limit
        self.visual_llm = threading.Semaphore(max(1, visual_llm_limit))
        if rag_limit is None:
            rag_limit = llm_limit
        self.rag = threading.Semaphore(max(1, rag_limit))
        self._slurm_limit = max(0, slurm_limit)
        self._slurm_active = 0
        self._slurm_lock = threading.Lock()
        self._local_limit = max(1, local_limit if local_limit is not None else 1)
        self._local_active = 0
        self._local_lock = threading.Lock()
        self._render_claims: dict[tuple[str, str], str] = {}
        self._render_claims_lock = threading.Lock()

    def claim_render(self, key: tuple[str, str], backend: str, *, existing=False) -> bool:
        """Reserve an attempt across runtimes; uncertain dispatches keep their claim.

        Restored identities are counted once, even when they exceed today's limit.
        This only accounts for capacity: it never adopts or operates a process.
        """
        if backend not in {"slurm", "local"}:
            raise ValueError("Unknown render capacity backend")
        with self._render_claims_lock:
            previous = self._render_claims.get(key)
            if previous is not None:
                if previous != backend:
                    raise ValueError("Render capacity identity changed backend")
                return True
            acquire = self.try_acquire_slurm if backend == "slurm" else self.try_acquire_local
            register = (
                self.register_existing_slurm if backend == "slurm" else self.register_existing_local
            )
            if existing:
                register()
            elif not acquire():
                return False
            self._render_claims[key] = backend
            return True

    def release_render(self, key: tuple[str, str]) -> None:
        """Release only after the owning runtime durably confirms terminal state."""
        with self._render_claims_lock:
            backend = self._render_claims.pop(key, None)
            if backend is not None:
                (self.release_slurm if backend == "slurm" else self.release_local)()

    def try_acquire_slurm(self) -> bool:
        with self._slurm_lock:
            if self._slurm_limit and self._slurm_active >= self._slurm_limit:
                return False
            self._slurm_active += 1
            return True

    def register_existing_slurm(self) -> None:
        """把恢复时已在远端存在的作业计入配额，即使它已超过当前限制。"""

        with self._slurm_lock:
            self._slurm_active += 1

    def release_slurm(self) -> None:
        with self._slurm_lock:
            if self._slurm_active <= 0:
                raise RuntimeError("Slurm 资源名额被重复释放")
            self._slurm_active -= 1

    def try_acquire_local(self) -> bool:
        """获取本地正式渲染名额；默认跨批次也保持串行。"""

        with self._local_lock:
            if self._local_active >= self._local_limit:
                return False
            self._local_active += 1
            return True

    def register_existing_local(self) -> None:
        """兼容接口：本地旧进程不会在恢复时被认领。"""

        with self._local_lock:
            self._local_active += 1

    def release_local(self) -> None:
        with self._local_lock:
            if self._local_active <= 0:
                raise RuntimeError("本地渲染资源名额被重复释放")
            self._local_active -= 1
