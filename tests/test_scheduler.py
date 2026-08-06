"""场景级并行调度 (per-scene pipeline) 测试。

每个 Scene 独立推进 分镜→编码→审查→提交→渲染→修复, 互不等待。
"""
import threading
import time
from pathlib import Path

import kd1_anime.orchestrator as module
from kd1_anime.agents.planner import SceneOutline, ScenePlan
from kd1_anime.agents.reviewer import ReviewResult
from kd1_anime.agents.validator import CodeValidationResult
from kd1_anime.cluster.slurm import SlurmJob
from kd1_anime.config import settings
from kd1_anime.orchestrator import Orchestrator, PipelineContext, RunPaths, SceneState, State

CODE = "from manim import *\nclass Demo(Scene):\n    def construct(self): self.wait()\n"


def make_outline(scene_id: int, title: str = "demo") -> SceneOutline:
    return SceneOutline(
        scene_id=scene_id,
        title=title,
        duration_seconds=10,
        purpose="test",
        math_concept="circle",
    )


def make_plan(outline: SceneOutline) -> ScenePlan:
    return ScenePlan(
        scene_id=outline.scene_id,
        title=outline.title,
        duration_seconds=outline.duration_seconds,
        purpose=outline.purpose,
        math_concept=outline.math_concept,
        visual_design="v",
        camera_movement="c",
        visual_flow=["f"],
        key_moments=["k"],
        computation="comp",
    )


def make_paths(tmp_path: Path) -> RunPaths:
    root = tmp_path / "run"
    for directory in (root, root / "scenes", root / "logs", root / "videos"):
        directory.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        "20260802-120000-1234abcd",
        root,
        root / "scenes",
        root / "logs",
        root / "videos",
        root / "out.mp4",
    )


class FakePlanner:
    """plan_detail 返回固定 plan; 可记录调用并模拟慢速。"""

    def __init__(self, detail_delay: float = 0.0, events: list | None = None):
        self.detail_delay = detail_delay
        self.events = events

    def plan_detail(self, outline, all_outlines, user_prompt, *, stream=False):
        if self.events is not None:
            self.events.append(("detail_start", outline.scene_id, time.time()))
        if self.detail_delay:
            time.sleep(self.detail_delay)
        if self.events is not None:
            self.events.append(("detail_end", outline.scene_id, time.time()))
        return make_plan(outline)


class FakeCoder:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []  # (feedback, previous_code)

    def generate_code(self, scene_plan, feedback="", previous_code="", *, stream=True):
        self.calls.append((feedback, previous_code))
        return CODE


class FakeReviewer:
    """按顺序返回给定结果; 默认全部有效。"""

    def __init__(self, results=None):
        self.results = list(results) if results else []
        self.calls = 0

    def review(self, code, scene_plan):
        self.calls += 1
        if self.results:
            return self.results.pop(0)
        return ReviewResult(is_valid=True)


class FakeAutoFixer:
    def __init__(self, infra: bool = False):
        self.infra = infra
        self.fix_calls = 0

    def is_infrastructure_error(self, error_log):
        return self.infra

    def fix(self, code, error_log):
        self.fix_calls += 1
        return CODE


class FakeSlurm:
    """submit 立即返回; poll 根据 status_map 返回状态。"""

    def __init__(self, run_paths: RunPaths, status_map=None, error_log: str = ""):
        self.run_paths = run_paths
        self.status_map = status_map or {}
        self.submitted: list[int] = []
        self.jobs: dict[str, SlurmJob] = {}
        self._n = 0
        self.error_log = error_log
        self.events: list[tuple] = []

    def submit_scene(self, scene_id, python_file, scene_class_name="Scene", **kw):
        self._n += 1
        job_id = str(self._n)
        job = SlurmJob(
            job_id=job_id,
            scene_id=scene_id,
            script_path=self.run_paths.scenes / f"render_{scene_id}.sh",
            log_out=self.run_paths.logs / f"scene_{scene_id}.out",
            log_err=self.run_paths.logs / f"scene_{scene_id}.err",
            media_dir=self.run_paths.videos / f"scene_{scene_id}",
            scene_class_name=scene_class_name,
            submitted_at=time.time(),
        )
        self.jobs[job_id] = job
        self.submitted.append(scene_id)
        self.events.append(("submit", scene_id, time.time()))
        return job

    def poll_all_statuses(self, job_ids):
        out = {}
        for job_id in job_ids:
            if callable(self.status_map):
                status = self.status_map(job_id)
            else:
                status = self.status_map.get(job_id, "COMPLETED")
            out[job_id] = status(job_id) if callable(status) else status
        return out

    def _forward_log(self, job, positions):
        pass

    def cancel_job(self, job_id):
        return False

    def get_error_log(self, job):
        return self.error_log


class _CallableAgent:
    """让 PlannerAgent()/CoderAgent()/ReviewerAgent() 返回测试实例。"""

    def __init__(self, instance):
        self._instance = instance

    def __call__(self):
        return self._instance


def make_orchestrator(monkeypatch, tmp_path, run_paths, *, slurm=None, planner=None,
                      coder=None, reviewer=None, autofixer=None):
    orchestrator = Orchestrator()
    monkeypatch.setattr(
        orchestrator,
        "_validate",
        lambda value: CodeValidationResult(True, scene_classes=["Demo"]),
    )
    orchestrator.slurm = slurm or FakeSlurm(run_paths)
    if autofixer is not None:
        orchestrator.auto_fixer = autofixer
    monkeypatch.setattr(module, "PlannerAgent", _CallableAgent(planner or FakePlanner()))
    monkeypatch.setattr(module, "CoderAgent", _CallableAgent(coder or FakeCoder()))
    monkeypatch.setattr(module, "ReviewerAgent", _CallableAgent(reviewer or FakeReviewer()))
    monkeypatch.setattr(settings, "MONITOR_POLL_INTERVAL", 1)
    return orchestrator


# ---------------------------------------------------------------------------
# 1) 已就绪场景立即提交渲染, 不等待其他场景的 LLM 阶段
# ---------------------------------------------------------------------------
def test_ready_scene_submits_while_other_scene_still_detailing(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    events: list[tuple] = []
    detail_done = threading.Event()

    class SlowPlanner(FakePlanner):
        def plan_detail(self, outline, all_outlines, user_prompt, *, stream=False):
            if outline.scene_id == 2:
                events.append(("detail_start", time.time()))
                time.sleep(0.3)
                events.append(("detail_end", time.time()))
                detail_done.set()
            return make_plan(outline)

    slurm = FakeSlurm(run_paths)
    orchestrator = make_orchestrator(
        monkeypatch, tmp_path, run_paths, slurm=slurm, planner=SlowPlanner()
    )

    ctx = PipelineContext("x", paths=run_paths, outlines=[make_outline(1), make_outline(2)])
    # Scene 1: 分镜/编码/审查全部完成 → 立即可提交
    ctx.scene_states[1] = SceneState(
        plan=make_plan(make_outline(1)), code=CODE, class_name="Demo",
        plan_ready=True, reviewed=True,
    )
    (run_paths.scenes / "scene_1.py").write_text(CODE, encoding="utf-8")
    # Scene 2: 只有占位 plan, 需要先 detail
    ctx.scene_states[2] = SceneState(plan=make_plan(make_outline(2)))

    orchestrator._run_scheduler(ctx)

    # Scene 1 在 Scene 2 的 detail 完成之前就已提交
    detail_end_time = next(t for e, t in events if e == "detail_end")
    scene1_submit_time = next(
        t for e, sid, t in slurm.events if e == "submit" and sid == 1
    )
    assert scene1_submit_time < detail_end_time
    assert ctx.scene_states[1].rendered is True
    assert ctx.scene_states[2].rendered is True
    assert sorted(slurm.submitted) == [1, 2]


# ---------------------------------------------------------------------------
# 2) 完整多场景流水线: detail→code→review→submit→render
# ---------------------------------------------------------------------------
def test_multiple_scenes_complete_independently(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    slurm = FakeSlurm(run_paths)
    orchestrator = make_orchestrator(monkeypatch, tmp_path, run_paths, slurm=slurm)

    ctx = PipelineContext("x", paths=run_paths, outlines=[make_outline(1), make_outline(2)])
    ctx.scene_states = {i: SceneState(plan=make_plan(make_outline(i))) for i in (1, 2)}

    orchestrator._run_scheduler(ctx)

    for sid in (1, 2):
        state = ctx.scene_states[sid]
        assert state.plan_ready is True
        assert state.reviewed is True
        assert state.rendered is True
    assert sorted(slurm.submitted) == [1, 2]


# ---------------------------------------------------------------------------
# 3) major 审查反馈 → 排队重写 → 再次审查通过
# ---------------------------------------------------------------------------
def test_major_review_queues_rewrite_then_passes(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    coder = FakeCoder()
    reviewer = FakeReviewer(
        [
            ReviewResult(
                is_valid=False,
                severity="major",
                feedback="需要重写",
                fixes=[{"find": "x", "replace": "y", "reason": "demo"}],
            ),
            ReviewResult(is_valid=True),
        ]
    )
    orchestrator = make_orchestrator(
        monkeypatch, tmp_path, run_paths,
        coder=coder, reviewer=reviewer,
    )

    ctx = PipelineContext("x", paths=run_paths, outlines=[make_outline(1)])
    ctx.scene_states[1] = SceneState(
        plan=make_plan(make_outline(1)), plan_ready=True,
    )

    orchestrator._run_scheduler(ctx)

    state = ctx.scene_states[1]
    assert state.reviewed is True
    assert state.rendered is True
    # 第一次生成 + 审查失败后带反馈重写
    assert len(coder.calls) == 2
    assert coder.calls[0][0] == ""
    assert "Reviewer" in coder.calls[1][0]


# ---------------------------------------------------------------------------
# 4) 渲染失败 → 自动修复 → 重新提交 → 成功
# ---------------------------------------------------------------------------
def test_render_failure_triggers_fix_and_resubmit(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)

    submitted_count = {"n": 0}

    def status_for(job_id):
        # 第一次提交失败, 修复后的新作业成功
        submitted_count["n"] += 1
        return "FAILED" if submitted_count["n"] == 1 else "COMPLETED"

    slurm = FakeSlurm(run_paths, status_map=status_for, error_log="render boom\n")
    autofixer = FakeAutoFixer()
    orchestrator = make_orchestrator(
        monkeypatch, tmp_path, run_paths, slurm=slurm, autofixer=autofixer,
    )

    ctx = PipelineContext("x", paths=run_paths, outlines=[make_outline(1)])
    ctx.scene_states[1] = SceneState(
        plan=make_plan(make_outline(1)), code=CODE, class_name="Demo",
        plan_ready=True, reviewed=True,
    )
    (run_paths.scenes / "scene_1.py").write_text(CODE, encoding="utf-8")

    orchestrator._run_scheduler(ctx)

    state = ctx.scene_states[1]
    assert autofixer.fix_calls == 1
    assert state.rendered is True
    assert state.fix_attempts == 1
    assert len(slurm.submitted) == 2  # 首次 + 修复后重新提交


# ---------------------------------------------------------------------------
# 5) 不可修复的失败状态 → 直接放弃, 不调用 autofixer
# ---------------------------------------------------------------------------
def test_non_fixable_failure_gives_up(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    slurm = FakeSlurm(run_paths, status_map={"1": "CANCELLED"})
    autofixer = FakeAutoFixer()
    orchestrator = make_orchestrator(
        monkeypatch, tmp_path, run_paths, slurm=slurm, autofixer=autofixer,
    )

    ctx = PipelineContext("x", paths=run_paths, outlines=[make_outline(1)])
    ctx.scene_states[1] = SceneState(
        plan=make_plan(make_outline(1)), code=CODE, class_name="Demo",
        plan_ready=True, reviewed=True,
    )
    (run_paths.scenes / "scene_1.py").write_text(CODE, encoding="utf-8")

    orchestrator._run_scheduler(ctx)

    state = ctx.scene_states[1]
    assert state.give_up is True
    assert autofixer.fix_calls == 0
    assert state.rendered is False


# ---------------------------------------------------------------------------
# 6) SLURM_MAX_IN_FLIGHT 限制在飞作业数
# ---------------------------------------------------------------------------
def test_in_flight_limit_is_respected(monkeypatch, tmp_path):
    from kd1_anime.config import settings as cfg

    monkeypatch.setattr(cfg, "SLURM_MAX_IN_FLIGHT", 1)
    run_paths = make_paths(tmp_path)
    slurm = FakeSlurm(run_paths)
    orchestrator = make_orchestrator(monkeypatch, tmp_path, run_paths, slurm=slurm)

    ctx = PipelineContext("x", paths=run_paths, outlines=[make_outline(1), make_outline(2)])
    ctx.scene_states = {
        i: SceneState(plan=make_plan(make_outline(i)), code=CODE, class_name="Demo",
                      plan_ready=True, reviewed=True)
        for i in (1, 2)
    }
    for i in (1, 2):
        (run_paths.scenes / f"scene_{i}.py").write_text(CODE, encoding="utf-8")

    orchestrator._run_scheduler(ctx)

    # 两个场景最终都完成; 提交按顺序进行 (第二个等待第一个释放名额)
    assert ctx.scene_states[1].rendered is True
    assert ctx.scene_states[2].rendered is True
    assert slurm.submitted == [1, 2]


# ---------------------------------------------------------------------------
# 7) dry-run: 完成 LLM 阶段后结束, 不提交
# ---------------------------------------------------------------------------
def test_dry_run_completes_without_submission(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    slurm = FakeSlurm(run_paths)
    orchestrator = make_orchestrator(monkeypatch, tmp_path, run_paths, slurm=slurm)
    ctx = PipelineContext("x", paths=run_paths, outlines=[make_outline(1)], dry_run=True)
    ctx.scene_states[1] = SceneState(plan=make_plan(make_outline(1)))

    orchestrator._run_scheduler(ctx)

    assert slurm.submitted == []
    assert ctx.scene_states[1].plan_ready is True
    assert ctx.scene_states[1].reviewed is True
    assert ctx.scene_states[1].rendered is False


# ---------------------------------------------------------------------------
# 8) _execute 端到端: dry-run 完整流水线 (init→outline→detail→code→review→DONE)
# ---------------------------------------------------------------------------
def test_execute_dry_run_end_to_end(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)

    class FullPlanner(FakePlanner):
        def plan_outline(self, user_prompt):
            return [make_outline(1), make_outline(2)]

    orchestrator = make_orchestrator(monkeypatch, tmp_path, run_paths, planner=FullPlanner())
    # _plan_outline 走的是 orchestrator.planner 实例, 需要一并替换
    orchestrator.planner = FullPlanner()
    ctx = PipelineContext("prompt", paths=run_paths, dry_run=True)

    result = orchestrator._execute(ctx, State.INIT)

    assert result is None
    assert all(
        st.plan_ready and st.reviewed for st in ctx.scene_states.values()
    )
    manifest = (run_paths.root / "manifest.json")
    assert manifest.is_file()
    import json
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["status"] == "dry_run_complete"


# ---------------------------------------------------------------------------
# 9) _execute 端到端: 渲染 + 合并 → 返回最终视频
# ---------------------------------------------------------------------------
def test_execute_full_pipeline_with_merge(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    slurm = FakeSlurm(run_paths)

    class FakeMerger:
        def merge_jobs(self, rendered_jobs, output_path=None):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x00" * 100)
            return output_path

        def collect_incremental_videos(self, *args, **kwargs):
            return []

    orchestrator = make_orchestrator(monkeypatch, tmp_path, run_paths, slurm=slurm)
    orchestrator.merger = FakeMerger()
    ctx = PipelineContext("prompt", paths=run_paths)
    ctx.outlines = [make_outline(1)]
    ctx.scene_states = {1: SceneState(plan=make_plan(make_outline(1)))}

    result = orchestrator._execute(ctx, State.INIT)

    assert result is not None
    assert result.exists()
    assert ctx.scene_states[1].rendered is True
    import json
    data = json.loads((run_paths.root / "manifest.json").read_text(encoding="utf-8"))
    assert data["status"] == "completed"


# ---------------------------------------------------------------------------
# 10) 连续相同渲染错误 → 判定环境问题提前放弃, 并附错误日志尾部
# ---------------------------------------------------------------------------
def test_identical_render_error_gives_up_early(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    error_log = (
        "Rendering:   0%|          | 0/60 [00:00<?, ?it/s]\n"
        "Traceback (most recent call last):\n"
        "  File \"scene_1.py\", line 42, in construct\n"
        "ValueError: something deterministic\n"
    )
    slurm = FakeSlurm(
        run_paths,
        status_map=lambda job_id: "FAILED",
        error_log=error_log,
    )
    autofixer = FakeAutoFixer()
    orchestrator = make_orchestrator(
        monkeypatch, tmp_path, run_paths, slurm=slurm, autofixer=autofixer,
    )
    monkeypatch.setattr(module.settings, "MAX_FIX_IDENTICAL_ERRORS", 2)

    ctx = PipelineContext("x", paths=run_paths, outlines=[make_outline(1)])
    ctx.scene_states[1] = SceneState(
        plan=make_plan(make_outline(1)), code=CODE, class_name="Demo",
        plan_ready=True, reviewed=True,
    )
    (run_paths.scenes / "scene_1.py").write_text(CODE, encoding="utf-8")

    orchestrator._run_scheduler(ctx)

    state = ctx.scene_states[1]
    assert state.give_up is True
    # 相同错误至少要修 2 次 (fix_attempts>=2 门槛), 第 3 次相同才提前放弃,
    # 避免一次修复失败就被误判为环境问题。
    assert autofixer.fix_calls == 2
    assert "连续 3 次渲染错误完全相同" in state.failure_reason
    # 放弃原因里带错误日志尾部, 方便直接定位根因
    assert "ValueError: something deterministic" in state.failure_reason


def test_error_fingerprint_normalizes_digits(monkeypatch, tmp_path):
    from kd1_anime.orchestrator import Orchestrator

    fp1 = Orchestrator._error_fingerprint(
        "Traceback line 42: ValueError at frame 1234"
    )
    fp2 = Orchestrator._error_fingerprint(
        "Traceback line 99: ValueError at frame 9999"
    )
    assert fp1 == fp2  # 数字不同 → 指纹相同

    fp3 = Orchestrator._error_fingerprint(
        "Traceback line 42: KeyError at frame 1234"
    )
    assert fp1 != fp3  # 错误类型不同 → 指纹不同
