"""跨 Orchestrator 共享的进程级资源配额。"""

from __future__ import annotations

import threading


class ResourceCoordinator:
    def __init__(self, *, llm_limit: int, slurm_limit: int) -> None:
        self.llm = threading.Semaphore(max(1, llm_limit))
        self._slurm_limit = max(0, slurm_limit)
        self._slurm_active = 0
        self._slurm_lock = threading.Lock()

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
