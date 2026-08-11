"""批量并行处理模块测试。"""

import json
from types import SimpleNamespace

import pytest

import kd1_anime.orchestrator as orchestrator_module
from kd1_anime.batch import (
    BatchConfig,
    BatchProcessor,
    BatchTask,
    load_prompts_from_file,
)


@pytest.fixture
def tmp_prompts_file(tmp_path):
    """创建临时 prompts 文件。"""
    prompts = [
        "Explain Euler's formula",
        "Show the Pythagorean theorem",
        "Visualize Fourier transform",
    ]
    file_path = tmp_path / "prompts.txt"
    file_path.write_text("\n".join(prompts), encoding="utf-8")
    return file_path


@pytest.fixture
def tmp_json_prompts_file(tmp_path):
    """创建临时 JSON prompts 文件。"""
    data = {
        "prompts": [
            "Explain Euler's formula",
            "Show the Pythagorean theorem",
            "Visualize Fourier transform",
        ]
    }
    file_path = tmp_path / "prompts.json"
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return file_path


class TestLoadPrompts:
    """测试 prompts 加载功能。"""

    def test_load_from_text_file(self, tmp_prompts_file):
        """测试从纯文本文件加载 prompts。"""
        prompts = load_prompts_from_file(tmp_prompts_file)
        assert len(prompts) == 3
        assert prompts[0] == "Explain Euler's formula"

    def test_load_from_json_file(self, tmp_json_prompts_file):
        """测试从 JSON 文件加载 prompts。"""
        prompts = load_prompts_from_file(tmp_json_prompts_file)
        assert len(prompts) == 3
        assert prompts[0] == "Explain Euler's formula"

    def test_load_with_comments(self, tmp_path):
        """测试加载带注释的文件。"""
        content = """# 这是注释
Explain Euler's formula
# 另一个注释
Show the Pythagorean theorem
"""
        file_path = tmp_path / "prompts.txt"
        file_path.write_text(content, encoding="utf-8")
        prompts = load_prompts_from_file(file_path)
        assert len(prompts) == 2

    def test_load_empty_file(self, tmp_path):
        """测试加载空文件。"""
        file_path = tmp_path / "empty.txt"
        file_path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="未包含有效任务"):
            load_prompts_from_file(file_path)


class TestBatchConfig:
    """测试批量配置。"""

    def test_default_config(self):
        """测试默认配置。"""
        config = BatchConfig()
        assert config.max_parallel == 3
        assert config.dry_run is False
        assert config.output_dir is None
        assert config.incremental is False

    def test_custom_config(self, tmp_path):
        """测试自定义配置。"""
        config = BatchConfig(
            max_parallel=5,
            dry_run=True,
            output_dir=tmp_path / "output",
        )
        assert config.max_parallel == 5
        assert config.dry_run is True
        assert config.output_dir == tmp_path / "output"

    def test_string_base_run_ids_are_normalized(self):
        config = BatchConfig(base_run_ids={"1": "20260728-120000-1234abcd"})
        assert config.base_run_ids == {1: "20260728-120000-1234abcd"}

    def test_rejects_invalid_parallelism(self):
        with pytest.raises(ValueError, match="max_parallel"):
            BatchConfig(max_parallel=0)

        with pytest.raises(ValueError, match="max_parallel"):
            BatchConfig(max_parallel=True)

    def test_rejects_invalid_base_run_id(self):
        with pytest.raises(ValueError, match="无效 run-id"):
            BatchConfig(base_run_ids={1: "not-a-run"})


class TestBatchProcessor:
    """测试批量处理器。"""

    def test_add_task(self):
        """测试添加任务。"""
        processor = BatchProcessor()
        task = processor.add_task("Test prompt")
        assert len(processor.tasks) == 1
        assert task.task_id == 1
        assert task.prompt == "Test prompt"
        assert task.status == "pending"

    def test_add_multiple_tasks(self):
        """测试添加多个任务。"""
        processor = BatchProcessor()
        processor.add_task("Task 1")
        processor.add_task("Task 2")
        processor.add_task("Task 3")
        assert len(processor.tasks) == 3
        assert processor.tasks[0].task_id == 1
        assert processor.tasks[2].task_id == 3

    def test_load_tasks_from_file(self, tmp_prompts_file):
        """测试从文件加载任务。"""
        processor = BatchProcessor()
        tasks = processor.load_tasks_from_file(tmp_prompts_file)
        assert len(tasks) == 3
        assert tasks[0].prompt == "Explain Euler's formula"
        assert tasks[1].prompt == "Show the Pythagorean theorem"

    def test_generate_summary(self):
        """测试生成摘要。"""
        processor = BatchProcessor()
        tasks = [
            BatchTask(task_id=1, prompt="Task 1", status="completed"),
            BatchTask(task_id=2, prompt="Task 2", status="failed", error="Test error"),
        ]
        summary = processor.generate_summary(tasks)
        assert "成功: 1" in summary
        assert "失败: 1" in summary

    def test_duplicate_output_targets_are_rejected(self, tmp_path):
        processor = BatchProcessor()
        output = tmp_path / "same.mp4"
        processor.add_task("one", output)
        processor.add_task("two", output)

        with pytest.raises(ValueError, match="重复输出路径"):
            processor._validate_output_targets()

    def test_output_validation_does_not_change_existing_parent_permissions(self, tmp_path):
        output_dir = tmp_path / "shared"
        output_dir.mkdir(mode=0o755)
        processor = BatchProcessor()
        processor.add_task("one", output_dir / "one.mp4")

        processor._validate_output_targets()

        assert output_dir.stat().st_mode & 0o777 == 0o755

    def test_dry_run_does_not_report_nonexistent_output(self, monkeypatch, tmp_path):
        class FakeOrchestrator:
            def __init__(self, resource_coordinator=None):
                self._ctx = None

            def run(self, *args, **kwargs):
                return None

        monkeypatch.setattr(orchestrator_module, "Orchestrator", FakeOrchestrator)
        processor = BatchProcessor(BatchConfig(dry_run=True))
        task = processor.add_task("demo", tmp_path / "not-created.mp4")

        processor._execute_single_task(task)

        assert task.status == "completed"
        assert task.output is None

    def test_failed_task_keeps_run_id_for_diagnostics(self, monkeypatch, tmp_path):
        class FakeOrchestrator:
            def __init__(self, resource_coordinator=None):
                self._ctx = SimpleNamespace(
                    paths=SimpleNamespace(run_id="20260728-120000-1234abcd")
                )

            def run(self, *args, **kwargs):
                raise RuntimeError("render failed")

        monkeypatch.setattr(orchestrator_module, "Orchestrator", FakeOrchestrator)
        processor = BatchProcessor(BatchConfig(dry_run=True))
        task = processor.add_task("demo", tmp_path / "not-created.mp4")

        processor._execute_single_task(task)

        assert task.status == "failed"
        assert task.run_id == "20260728-120000-1234abcd"


class TestBatchTask:
    """测试批量任务。"""

    def test_task_initialization(self):
        """测试任务初始化。"""
        task = BatchTask(task_id=1, prompt="Test")
        assert task.task_id == 1
        assert task.prompt == "Test"
        assert task.status == "pending"
        assert task.output is None
        assert task.error is None

    def test_task_with_output(self, tmp_path):
        """测试带输出的任务。"""
        output = tmp_path / "output.mp4"
        task = BatchTask(task_id=1, prompt="Test", output=output)
        assert task.output == output


def test_invalid_json_prompt_file_is_rejected(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="格式无效"):
        load_prompts_from_file(path)


def test_json_prompt_entries_must_be_strings(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"prompts": ["ok", {"prompt": "bad"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="字符串数组"):
        load_prompts_from_file(path)
