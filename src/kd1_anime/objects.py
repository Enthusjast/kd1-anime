"""Private immutable objects referenced by an atomic run manifest."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


class ObjectStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def put(self, relative_dir: str, content: str, suffix: str) -> str:
        encoded = content.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        directory = self.root / relative_dir
        relative = Path(relative_dir)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe immutable object directory")
        current = self.root
        for component in relative.parts:
            current /= component
            if current.is_symlink():
                raise ValueError("Unsafe immutable object directory")
        if directory.is_symlink() or not directory.resolve().is_relative_to(self.root):
            raise ValueError("Unsafe immutable object directory")
        # Each new ancestor is private, including intermediate revision folders.
        missing = []
        current = directory
        while not current.exists():
            missing.append(current)
            current = current.parent
        for parent in reversed(missing):
            parent.mkdir(mode=0o700, exist_ok=True)
            parent.chmod(0o700)
        target = directory / (digest + suffix)
        if target.is_symlink():
            raise ValueError("Immutable object cannot be a symbolic link")
        if target.exists():
            if target.read_bytes() != encoded:
                raise ValueError("Immutable object identity mismatch")
            return target.relative_to(self.root).as_posix()
        descriptor, name = tempfile.mkstemp(prefix=".object-", dir=directory)
        temporary = Path(name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.is_symlink() or target.read_bytes() != encoded:
                    raise ValueError("Immutable object identity mismatch") from None
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        return target.relative_to(self.root).as_posix()

    def source(self, scene_id: int, code: str) -> str:
        if isinstance(scene_id, bool) or scene_id < 1:
            raise ValueError("Invalid scene identity")
        return self.put(f"scenes/revisions/scene_{scene_id}", code, ".py")

    def report(self, content: str) -> str:
        return self.put("artifacts/objects", content, ".json")
