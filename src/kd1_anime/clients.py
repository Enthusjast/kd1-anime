"""Run-owned connection pools; per-agent request state remains independent."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from contextvars import ContextVar
from threading import RLock


class ClientPool:
    def __init__(self):
        self._clients = {}
        self._lock = RLock()
        self.created = 0
        self.closed = False

    def get(self, key, factory):
        with self._lock:
            if self.closed:
                raise RuntimeError("Run client pool is closed")
            if key not in self._clients:
                self._clients[key] = factory()
                self.created += 1
            return self._clients[key]

    def close(self):
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
            self.closed = True
        for client in clients:
            with suppress(Exception):
                client.close()


CURRENT_CLIENTS: ContextVar[ClientPool | None] = ContextVar("kd1_clients", default=None)


@contextmanager
def client_scope():
    existing = CURRENT_CLIENTS.get()
    if existing is not None:
        yield existing
        return
    pool = ClientPool()
    token = CURRENT_CLIENTS.set(pool)
    try:
        yield pool
    finally:
        CURRENT_CLIENTS.reset(token)
        pool.close()
