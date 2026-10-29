"""Benchmark entry points retain measured operations and bound v11 runtime evidence."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from test_agent_vertical_slice import agent_boundary as agent_boundary

from kd1_anime.agent_runtime.runtime import AgentRuntime
from kd1_anime.commits import RunCommitter
from kd1_anime.config import settings
from kd1_anime.run_store import RunRepository


def load_tool(name):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark_runtime = load_tool("benchmark_runtime")
compare_codegen = load_tool("compare_codegen")


def test_operation_benchmark_publishes_four_v11_commits(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark_runtime.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    published = []
    commit = RunCommitter.commit

    def record_commit(self, command):
        result = commit(self, command)
        published.append(result)
        return result

    monkeypatch.setattr(RunCommitter, "commit", record_commit)
    result = benchmark_runtime.measure(2)

    assert result["scenes"] == 2
    assert result["checkpoints"] == len(result["checkpoint_seconds"]) == 4
    assert result["lock_acquisitions"] == 4
    assert result["lock_wait_seconds"] >= 0
    assert result["dependency_digest_calls"] == 0
    assert result["rag_sqrt_calls_10_queries"] == 10
    assert result["client_connections"] == 1
    assert [manifest.revision for manifest in published] == [1, 2, 3, 4]
    assert all(
        manifest.schema_version == 11 and manifest.agent is not None for manifest in published
    )
    saved = RunRepository(tmp_path).load(published[-1].run_id)
    assert saved.revision == 4 and len(saved.scenes) == 2
    assert saved.state == "CODING" and not saved.integrity_errors()


@pytest.mark.parametrize("dry_run", [False, True])
def test_model_comparison_executes_both_v11_modes(agent_boundary, monkeypatch, tmp_path, dry_run):
    boundary, dependencies = agent_boundary
    task_id = compare_codegen.fixed_tasks()[0].task_id
    report_path = tmp_path / "comparison.json"
    args = [
        "compare_codegen",
        "--run-models",
        "--task",
        task_id,
        "--backend",
        "slurm",
        "--output",
        str(report_path),
    ]
    if dry_run:
        args.append("--dry-run")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(
        compare_codegen, "AgentRuntime", lambda: AgentRuntime(dependencies=dependencies)
    )
    original_mode = settings.SCENE_CODEGEN_MODE

    compare_codegen.main()

    assert original_mode == settings.SCENE_CODEGEN_MODE
    report = json.loads(report_path.read_text())
    assert report["scope"] == "bound_runtime_runs"
    bindings = json.loads(report_path.with_suffix(".bindings.json").read_text())
    assert set(bindings) == {"llm", "compiler_first"}
    for mode, tasks in bindings.items():
        assert set(tasks) == {task_id}
        root = Path(tasks[task_id])
        manifest = RunRepository(boundary.workspace).load(root.name)
        assert manifest.schema_version == 11 and manifest.scene_codegen_mode == mode
        assert manifest.status == ("dry_run_complete" if dry_run else "completed")
        assert manifest.request_budget.reservations
        assert any(item.action == "propose_code_candidate" for item in manifest.agent.invocations)
        assert report["groups"][mode]["tasks"][0]["task_id"] == task_id
        assert report["groups"][mode]["tasks"][0]["success"] is not dry_run
    assert len(boundary.backend.sources) == (0 if dry_run else 2)


@pytest.mark.parametrize("failure_has_manifest", [False, True])
def test_model_comparison_keeps_failure_bindings_and_continues(
    monkeypatch, tmp_path, capsys, failure_has_manifest
):
    task_id = compare_codegen.fixed_tasks()[0].task_id
    report_path = tmp_path / "comparison.json"
    calls = []
    compared = []

    class Runtime:
        def generate(self, prompt, **kwargs):
            mode = settings.SCENE_CODEGEN_MODE
            calls.append((mode, prompt, kwargs))
            if mode == "llm" and not failure_has_manifest:
                raise OSError("before manifest publication")
            self.root = tmp_path / mode
            self.root.mkdir()
            (self.root / "manifest.json").write_text("{}")
            if mode == "llm":
                raise OSError("after manifest publication")

    def compare(bindings):
        compared.append(bindings)
        return {"scope": "bound_runtime_runs"}

    def no_offline_fallback():
        pytest.fail("--run-models must not fall back to offline compiler data")

    monkeypatch.setattr(compare_codegen, "AgentRuntime", Runtime)
    monkeypatch.setattr(compare_codegen, "compare_modes", compare)
    monkeypatch.setattr(compare_codegen, "offline_baseline", no_offline_fallback)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_codegen",
            "--run-models",
            "--task",
            task_id,
            "--backend",
            "local",
            "--dry-run",
            "--output",
            str(report_path),
        ],
    )
    original_mode = settings.SCENE_CODEGEN_MODE

    compare_codegen.main()

    assert original_mode == settings.SCENE_CODEGEN_MODE
    assert [mode for mode, _, _ in calls] == ["llm", "compiler_first"]
    assert all(prompt == compare_codegen.fixed_tasks()[0].prompt for _, prompt, _ in calls)
    assert all(
        kwargs == {"dry_run": True, "backend": "local", "approve_plan": True}
        for _, _, kwargs in calls
    )
    assert compared == [
        {
            "llm": {task_id: tmp_path / "llm"} if failure_has_manifest else {},
            "compiler_first": {task_id: tmp_path / "compiler_first"},
        }
    ]
    saved = json.loads(report_path.with_suffix(".bindings.json").read_text())
    assert bool(saved["llm"]) is failure_has_manifest
    assert saved["compiler_first"][task_id] == str(tmp_path / "compiler_first")
    assert json.loads(capsys.readouterr().out) == {
        "task": task_id,
        "mode": "llm",
        "outcome": "OSError",
    }


def test_bound_comparison_does_not_execute_models(monkeypatch, tmp_path, capsys):
    bindings_path = tmp_path / "bindings.json"
    bindings_path.write_text(json.dumps({"llm": {"task": str(tmp_path / "run")}}))
    compared = []

    def compare(bindings):
        compared.append(bindings)
        return {"scope": "bound_runtime_runs"}

    def no_runtime():
        pytest.fail("--bindings must not execute models")

    monkeypatch.setattr(compare_codegen, "compare_modes", compare)
    monkeypatch.setattr(compare_codegen, "AgentRuntime", no_runtime)
    monkeypatch.setattr(sys, "argv", ["compare_codegen", "--bindings", str(bindings_path)])
    compare_codegen.main()
    assert compared == [{"llm": {"task": tmp_path / "run"}}]
    assert json.loads(capsys.readouterr().out)["scope"] == "bound_runtime_runs"


def test_default_comparison_stays_explicitly_offline(monkeypatch, capsys):
    def no_runtime():
        pytest.fail("Default offline benchmark must not execute models")

    monkeypatch.setattr(compare_codegen, "AgentRuntime", no_runtime)
    monkeypatch.setattr(
        compare_codegen, "offline_baseline", lambda: {"scope": "offline_compiler_contract"}
    )
    monkeypatch.setattr(sys, "argv", ["compare_codegen"])
    compare_codegen.main()
    assert json.loads(capsys.readouterr().out)["scope"] == "offline_compiler_contract"
