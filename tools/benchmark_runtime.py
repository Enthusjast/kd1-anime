"""Deterministic operation counts plus diagnostic timings; never calls a model."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path


def measure(scene_count):
    import kd1_anime.agents.base as agent_module
    from kd1_anime.agents.base import BaseAgent
    from kd1_anime.config import settings
    from kd1_anime.domain.planning import ScenePlan
    from kd1_anime.orchestrator import Orchestrator, PipelineContext, RunPaths, SceneState, State
    from kd1_anime.rag.chunker import SourceChunk
    from kd1_anime.rag.store import RagIndex
    from kd1_anime.services.dependencies import DependencyService

    counts = {"dependency_digests": 0, "connections": 0, "vector_norms": 0}
    original_contract, original_teaching = DependencyService.contract, DependencyService.teaching

    def contract(*args):
        counts["dependency_digests"] += 1
        return original_contract(*args)

    def teaching(*args):
        counts["dependency_digests"] += 1
        return original_teaching(*args)

    DependencyService.contract = staticmethod(contract)
    DependencyService.teaching = staticmethod(teaching)

    root = Path(tempfile.mkdtemp(prefix="kd1-runtime-bench-"))
    run_root = root / "runs" / "20261019-120000-1234abcd"
    paths = RunPaths(
        run_root.name,
        run_root,
        run_root / "scenes",
        run_root / "logs",
        run_root / "videos",
        run_root / "output.mp4",
    )
    states = {}
    for sid in range(1, scene_count + 1):
        plan = ScenePlan(
            scene_id=sid,
            dependencies_complete=True,
            title=f"Scene {sid}",
            duration_seconds=4,
            purpose="benchmark",
            math_concept="circle",
            visual_design="fixed",
            camera_movement="fixed",
            visual_flow=["reveal"],
            key_moments=["end"],
            computation="r=1",
        )
        states[sid] = SceneState(plan=plan)
    context = PipelineContext("fixed benchmark", paths=paths, scene_states=states)
    workflow = Orchestrator()
    workflow._ctx = context

    class MeasuredLock:
        def __init__(self):
            self.lock = threading.RLock()
            self.wait = 0.0
            self.acquisitions = 0

        def __enter__(self):
            start = time.perf_counter()
            self.lock.acquire()
            self.wait += time.perf_counter() - start
            self.acquisitions += 1
            return self

        def __exit__(self, *args):
            self.lock.release()

    lock = MeasuredLock()
    workflow._state_lock = lock
    timings = []
    try:
        for _ in range(4):
            start = time.perf_counter()
            workflow._checkpoint(context, State.CODING)
            timings.append(time.perf_counter() - start)
    finally:
        DependencyService.contract = staticmethod(original_contract)
        DependencyService.teaching = staticmethod(original_teaching)

    chunks = [
        SourceChunk(
            path=Path(f"doc-{i}.md"),
            source_kind="manim_doc",
            text=f"fixed document {i}",
            ordinal=0,
            source_sha256="a" * 64,
            metadata={},
        )
        for i in range(scene_count * 10)
    ]
    vectors = [[1.0, float(i % 7), 2.0, 3.0] for i in range(len(chunks))]
    index = RagIndex(root / "index.sqlite3")
    RagIndex.build(index.path, chunks, vectors, embedding_model="fixed-benchmark")
    snapshot = index.load_verified()
    original_sqrt = math.sqrt

    def sqrt(value):
        counts["vector_norms"] += 1
        return original_sqrt(value)

    math.sqrt = sqrt
    start = time.perf_counter()
    try:
        for _ in range(10):
            index.search_verified(
                [1.0, 1.0, 2.0, 3.0],
                top_k=5,
                source_kinds=None,
                exclude_frameworks=None,
                info=snapshot.info,
                snapshot=snapshot,
            )
    finally:
        math.sqrt = original_sqrt
    rag_seconds = time.perf_counter() - start

    class FakeClient:
        def __init__(self, **kwargs):
            counts["connections"] += 1

        def close(self):
            pass

    original_client = agent_module.OpenAI
    agent_module.OpenAI = FakeClient
    try:
        try:
            from kd1_anime.clients import client_scope
        except ImportError:
            from contextlib import nullcontext

            client_scope = nullcontext
        profile = replace(settings.main_llm_profile(), api_key="fake", trust_env=True)
        with client_scope():
            for _ in range(scene_count):
                agent = BaseAgent(profile)
                _ = agent.client
                close = getattr(agent, "close", None)
                if close:
                    close()
    finally:
        agent_module.OpenAI = original_client
    return {
        "scenes": scene_count,
        "checkpoints": 4,
        "checkpoint_seconds": timings,
        "lock_wait_seconds": lock.wait,
        "lock_acquisitions": lock.acquisitions,
        "dependency_digest_calls": counts["dependency_digests"],
        "rag_seconds_10_queries": rag_seconds,
        "rag_sqrt_calls_10_queries": counts["vector_norms"],
        "client_connections": counts["connections"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [measure(count) for count in (10, 50, 100)]
    args.output.write_text(
        json.dumps(
            {
                "scope": "offline_runtime_operations",
                "timings_are_diagnostic": True,
                "scenarios": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
