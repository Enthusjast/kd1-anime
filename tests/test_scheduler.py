"""分阶段场景调度测试。

分镜并行，代码按场景顺序交接，渲染任务并行执行。
"""

import threading
import time
from pathlib import Path

import kd1_anime.orchestrator as module
from kd1_anime.agents.continuity import ContinuityReviewResult
from kd1_anime.agents.plan_reviewer import PlanReviewResult
from kd1_anime.agents.planner import ContinuityBible, SceneOutline, ScenePlan
from kd1_anime.agents.reviewer import ReviewResult
from kd1_anime.agents.technical_planner import TechnicalSpec
from kd1_anime.agents.validator import CodeValidationResult
from kd1_anime.cluster.slurm import SlurmJob
from kd1_anime.config import settings
from kd1_anime.orchestrator import Orchestrator, PipelineContext, RunPaths, SceneState, State
from kd1_anime.rendering import VideoMetadata
from kd1_anime.run_store import sha256_text

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

    def plan_detail(self, outline, all_outlines, user_prompt, *, stream=False, renderer=None):
        if self.events is not None:
            self.events.append(("detail_start", outline.scene_id, time.time()))
        if self.detail_delay:
            time.sleep(self.detail_delay)
        if self.events is not None:
            self.events.append(("detail_end", outline.scene_id, time.time()))
        plan = make_plan(outline)
        plan.opening_state = ["核心对象进入画面"]
        plan.closing_state = ["核心对象保留到场景结束"]
        plan.transition_in = "核心对象从初始状态接入"
        plan.transition_out = "保留核心对象交给下一场景"
        return plan


class ContinuityPlanner(FakePlanner):
    """支持全片连续性阶段的测试 Planner。"""

    def __init__(self):
        super().__init__()
        self.bible_calls = 0
        self.detail_calls = 0

    def plan_continuity_bible(self, user_prompt, outlines, *, stream=False, renderer=None):
        self.bible_calls += 1
        return ContinuityBible()

    def plan_detail(
        self,
        outline,
        all_outlines,
        user_prompt,
        *,
        stream=False,
        renderer=None,
        continuity_bible=None,
        continuity_feedback="",
    ):
        self.detail_calls += 1
        plan = make_plan(outline)
        plan.opening_state = ["核心公式 x=1"]
        plan.closing_state = ["核心公式 x=1"]
        plan.transition_in = "核心公式从上一状态变换接入"
        plan.transition_out = "保留核心公式并交给下一场景"
        plan.continuity_references = ["背景 #1C1C1C", "x 使用蓝色"]
        return plan


class PassingContinuityReviewer:
    calls = 0

    def review(self, *args, **kwargs):
        self.calls += 1
        return ContinuityReviewResult(is_valid=True, summary="通过")


class PassingPlanReviewer:
    calls = 0

    def review(self, *args, **kwargs):
        self.calls += 1
        return PlanReviewResult(is_valid=True, summary="计划通过")


class RejectingPlanReviewer:
    def __init__(self):
        self.calls = 0

    def review(self, *args, **kwargs):
        self.calls += 1
        return PlanReviewResult(
            is_valid=False,
            severity="major",
            summary="不可实现",
            issues=[
                {
                    "category": "feasibility",
                    "severity": "major",
                    "field": "visual_flow",
                    "message": "方案无法实现",
                    "fix_instruction": "删除不可实现的步骤",
                }
            ],
        )


class OneRoundContinuityReviewer:
    def __init__(self):
        self.calls = 0

    def review(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return ContinuityReviewResult(
                is_valid=False,
                summary="需要修正",
                issues=[
                    {
                        "scene_ids": [1, 2],
                        "category": "transition",
                        "severity": "major",
                        "message": "两个场景的交接对象不一致",
                        "fix_instruction": "两场景都使用核心公式 x=1 作为交接对象",
                    }
                ],
            )
        return ContinuityReviewResult(is_valid=True, summary="通过")


class ContextAwareContinuityPlanner(ContinuityPlanner):
    def __init__(self):
        super().__init__()
        self.continuity_contexts: list[str] = []

    def plan_detail(
        self,
        outline,
        all_outlines,
        user_prompt,
        *,
        stream=False,
        renderer=None,
        continuity_bible=None,
        continuity_feedback="",
        continuity_context="",
    ):
        self.continuity_contexts.append(continuity_context)
        return super().plan_detail(
            outline,
            all_outlines,
            user_prompt,
            stream=stream,
            renderer=renderer,
            continuity_bible=continuity_bible,
            continuity_feedback=continuity_feedback,
        )


class FakeCoder:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []  # (feedback, previous_code)

    def generate_code(
        self,
        scene_plan,
        feedback="",
        previous_code="",
        *,
        stream=True,
        renderer=None,
        continuity_bible=None,
    ):
        self.calls.append((feedback, previous_code))
        return CODE


class FakeReviewer:
    """按顺序返回给定结果; 默认全部有效。"""

    def __init__(self, results=None):
        self.results = list(results) if results else []
        self.calls = 0

    def review(self, code, scene_plan, *, renderer=None, continuity_bible=None):
        self.calls += 1
        if self.results:
            return self.results.pop(0)
        return ReviewResult(is_valid=True)


class FakeTechnicalPlanner:
    """测试用技术计划器；无元素的简化计划无需额外技术事件。"""

    def __init__(self):
        self.calls = 0

    def plan(self, scene_plan, *, renderer=None, **kwargs):
        self.calls += 1
        return TechnicalSpec(scene_id=scene_plan.scene_id, renderer=renderer or "cairo")


class FakeAutoFixer:
    def __init__(self, infra: bool = False):
        self.infra = infra
        self.fix_calls = 0

    def is_infrastructure_error(self, error_log):
        return self.infra

    def fix(self, code, error_log, *, renderer=None):
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
            code_sha256=kw.get("code_sha256", ""),
            render_profile=kw.get("render_profile") or PipelineContext("x").render_profile,
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

    def validate_completed_job(self, job):
        job.media_dir.mkdir(parents=True, exist_ok=True)
        video = job.media_dir / f"{job.scene_class_name}.mp4"
        video.write_bytes(b"fake-video")
        job.output_path = video
        job.output_metadata = VideoMetadata(
            size_bytes=video.stat().st_size,
            duration_seconds=1,
            width=job.render_profile.pixel_width,
            height=job.render_profile.pixel_height,
            frame_rate=job.render_profile.frame_rate,
        )
        return True

    def _classify_gone(self, job):
        if any(job.media_dir.rglob(f"{job.scene_class_name}.mp4")):
            self.validate_completed_job(job)
            return "COMPLETED"
        return None

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


def make_orchestrator(
    monkeypatch,
    tmp_path,
    run_paths,
    *,
    slurm=None,
    planner=None,
    plan_reviewer=None,
    coder=None,
    reviewer=None,
    autofixer=None,
    technical_planner=None,
):
    orchestrator = Orchestrator()
    monkeypatch.setattr(
        orchestrator,
        "_validate",
        lambda value, **kwargs: CodeValidationResult(True, scene_classes=["Demo"]),
    )
    orchestrator.slurm = slurm or FakeSlurm(run_paths)
    monkeypatch.setattr(module, "PlannerAgent", _CallableAgent(planner or FakePlanner()))
    monkeypatch.setattr(
        module,
        "PlanReviewerAgent",
        _CallableAgent(plan_reviewer or PassingPlanReviewer()),
    )
    monkeypatch.setattr(module, "CoderAgent", _CallableAgent(coder or FakeCoder()))
    monkeypatch.setattr(module, "ReviewerAgent", _CallableAgent(reviewer or FakeReviewer()))
    monkeypatch.setattr(
        module,
        "TechnicalPlannerAgent",
        _CallableAgent(technical_planner or FakeTechnicalPlanner()),
    )
    monkeypatch.setattr(module, "AutoFixerAgent", _CallableAgent(autofixer or FakeAutoFixer()))
    monkeypatch.setattr(settings, "MONITOR_POLL_INTERVAL", 1)
    return orchestrator


# ---------------------------------------------------------------------------
# 1) 为了代码级连续性，编码/提交等待所有场景分镜完成
# ---------------------------------------------------------------------------
def test_code_barrier_waits_for_other_scene_detailing(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    events: list[tuple] = []
    detail_done = threading.Event()

    class SlowPlanner(FakePlanner):
        def plan_detail(self, outline, all_outlines, user_prompt, *, stream=False, renderer=None):
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
        plan=make_plan(make_outline(1)),
        code=CODE,
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
    )
    (run_paths.scenes / "scene_1.py").write_text(CODE, encoding="utf-8")
    # Scene 2: 只有占位 plan, 需要先 detail
    ctx.scene_states[2] = SceneState(plan=make_plan(make_outline(2)))

    orchestrator._run_scheduler(ctx)

    # Scene 1 的提交必须发生在 Scene 2 detail 完成之后，确保顺序编码前
    # 每个场景都拥有真实的 ScenePlan。
    detail_end_time = next(t for e, t in events if e == "detail_end")
    scene1_submit_time = next(t for e, sid, t in slurm.events if e == "submit" and sid == 1)
    assert scene1_submit_time >= detail_end_time
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
    emitted: list[tuple[str, int | None]] = []
    orchestrator._callback = lambda event, data: emitted.append((event, data.get("scene_id")))

    ctx = PipelineContext("x", paths=run_paths, outlines=[make_outline(1), make_outline(2)])
    ctx.scene_states = {i: SceneState(plan=make_plan(make_outline(i))) for i in (1, 2)}

    orchestrator._run_scheduler(ctx)

    for sid in (1, 2):
        state = ctx.scene_states[sid]
        assert state.plan_ready is True
        assert state.reviewed is True
        assert state.rendered is True
    assert sorted(slurm.submitted) == [1, 2]
    for scene_id in (1, 2):
        scene_events = [event for event, sid in emitted if sid == scene_id]
        assert "scene_detailing" in scene_events
        assert "scene_coding" in scene_events
        assert "scene_reviewing" in scene_events
        assert "scene_rendered" in scene_events


def test_coder_receives_previous_scene_export_in_scene_order(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)

    class ContextCoder(FakeCoder):
        def __init__(self):
            super().__init__()
            self.inherited: list[str] = []

        def generate_code(
            self,
            scene_plan,
            feedback="",
            previous_code="",
            *,
            stream=True,
            renderer=None,
            continuity_bible=None,
            inherited_elements_code="",
            inherited_elements=None,
            elements_to_remove=None,
        ):
            self.inherited.append(inherited_elements_code)
            self.calls.append((feedback, previous_code))
            return (
                "from manim import *\n"
                "class Demo(Scene):\n"
                "    def construct(self):\n"
                "        # KD1_CONTINUITY_EXPORT_BEGIN\n"
                "        formula = MathTex(r'x^2')\n"
                "        # KD1_CONTINUITY_EXPORT_END\n"
                "        self.add(formula)\n"
            )

    class ContextReviewer(FakeReviewer):
        def review(self, code, scene_plan, *, renderer=None, continuity_bible=None, **kwargs):
            return ReviewResult(is_valid=True)

    coder = ContextCoder()
    orchestrator = make_orchestrator(
        monkeypatch,
        tmp_path,
        run_paths,
        coder=coder,
        reviewer=ContextReviewer(),
    )
    ctx = PipelineContext(
        "x",
        paths=run_paths,
        dry_run=True,
        outlines=[make_outline(1), make_outline(2)],
        scene_states={
            1: SceneState(plan=make_plan(make_outline(1))),
            2: SceneState(plan=make_plan(make_outline(2))),
        },
    )

    orchestrator._run_scheduler(ctx)

    assert coder.inherited[0] == ""
    assert "formula = MathTex" in coder.inherited[1]


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
        monkeypatch,
        tmp_path,
        run_paths,
        coder=coder,
        reviewer=reviewer,
    )

    ctx = PipelineContext("x", paths=run_paths, outlines=[make_outline(1)])
    ctx.scene_states[1] = SceneState(
        plan=make_plan(make_outline(1)),
        plan_ready=True,
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
    reviewer = FakeReviewer()
    orchestrator = make_orchestrator(
        monkeypatch,
        tmp_path,
        run_paths,
        slurm=slurm,
        autofixer=autofixer,
        reviewer=reviewer,
    )

    ctx = PipelineContext("x", paths=run_paths, outlines=[make_outline(1)])
    ctx.scene_states[1] = SceneState(
        plan=make_plan(make_outline(1)),
        code=CODE,
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
    )
    (run_paths.scenes / "scene_1.py").write_text(CODE, encoding="utf-8")

    orchestrator._run_scheduler(ctx)

    state = ctx.scene_states[1]
    assert autofixer.fix_calls == 1
    assert state.rendered is True
    assert state.fix_attempts == 1
    assert len(slurm.submitted) == 2  # 首次 + 修复后重新提交
    assert reviewer.calls == 1  # 初始代码已审查；AutoFix 代码必须重新审查


def test_infrastructure_failure_requeues_without_autofix(monkeypatch, tmp_path):
    """节点终态应重排队，不应依赖 AutoFix 开关或调用 LLM。"""
    run_paths = make_paths(tmp_path)

    class InfraSlurm(FakeSlurm):
        def poll_all_statuses(self, job_ids):
            status = "NODE_FAIL" if len(self.submitted) == 1 else "COMPLETED"
            return {job_id: status for job_id in job_ids}

    slurm = InfraSlurm(run_paths)
    orchestrator = make_orchestrator(monkeypatch, tmp_path, run_paths, slurm=slurm)
    monkeypatch.setattr(settings, "MAX_INFRA_RETRIES", 1)

    ctx = PipelineContext(
        "x",
        paths=run_paths,
        auto_fix=False,
        outlines=[make_outline(1)],
    )
    ctx.scene_states[1] = SceneState(
        plan=make_plan(make_outline(1)),
        code=CODE,
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
    )
    (run_paths.scenes / "scene_1.py").write_text(CODE, encoding="utf-8")

    orchestrator._run_scheduler(ctx)

    state = ctx.scene_states[1]
    assert state.rendered is True
    assert state.failed is False
    assert state.give_up is False
    assert state.infra_retries == 1
    assert slurm.submitted == [1, 1]


# ---------------------------------------------------------------------------
# 5) 不可修复的失败状态 → 直接放弃, 不调用 autofixer
# ---------------------------------------------------------------------------
def test_non_fixable_failure_gives_up(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    slurm = FakeSlurm(run_paths, status_map={"1": "CANCELLED"})
    autofixer = FakeAutoFixer()
    orchestrator = make_orchestrator(
        monkeypatch,
        tmp_path,
        run_paths,
        slurm=slurm,
        autofixer=autofixer,
    )

    ctx = PipelineContext("x", paths=run_paths, outlines=[make_outline(1)])
    ctx.scene_states[1] = SceneState(
        plan=make_plan(make_outline(1)),
        code=CODE,
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
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
        i: SceneState(
            plan=make_plan(make_outline(i)),
            code=CODE,
            class_name="Demo",
            plan_ready=True,
            reviewed=True,
        )
        for i in (1, 2)
    }
    for i in (1, 2):
        (run_paths.scenes / f"scene_{i}.py").write_text(CODE, encoding="utf-8")

    orchestrator._run_scheduler(ctx)

    # 两个场景最终都完成; 提交按顺序进行 (第二个等待第一个释放名额)
    assert ctx.scene_states[1].rendered is True
    assert ctx.scene_states[2].rendered is True
    assert slurm.submitted == [1, 2]


def test_restored_job_reserves_slot_before_new_submission(monkeypatch, tmp_path):
    from kd1_anime.config import settings as cfg

    monkeypatch.setattr(cfg, "SLURM_MAX_IN_FLIGHT", 1)
    run_paths = make_paths(tmp_path)

    class CountingSlurm(FakeSlurm):
        def __init__(self, paths):
            super().__init__(paths)
            self.remote_active = 1
            self.max_remote_active = 1
            self._active_lock = threading.Lock()

        def submit_scene(self, *args, **kwargs):
            with self._active_lock:
                self.remote_active += 1
                self.max_remote_active = max(self.max_remote_active, self.remote_active)
            return super().submit_scene(*args, **kwargs)

        def poll_all_statuses(self, job_ids):
            if "99" in job_ids:
                time.sleep(0.1)
            return {job_id: "COMPLETED" for job_id in job_ids}

        def validate_completed_job(self, job):
            valid = super().validate_completed_job(job)
            with self._active_lock:
                self.remote_active -= 1
            return valid

    slurm = CountingSlurm(run_paths)
    existing = SlurmJob(
        job_id="99",
        scene_id=1,
        script_path=run_paths.scenes / "render_1.sh",
        log_out=run_paths.logs / "scene_1.out",
        log_err=run_paths.logs / "scene_1.err",
        media_dir=run_paths.videos / "scene_1",
        scene_class_name="Demo",
        submitted_at=time.time(),
        code_sha256=sha256_text(CODE),
        render_profile=PipelineContext("x").render_profile,
    )
    orchestrator = make_orchestrator(monkeypatch, tmp_path, run_paths, slurm=slurm)
    ctx = PipelineContext("x", paths=run_paths, outlines=[make_outline(1), make_outline(2)])
    ctx.scene_states = {
        1: SceneState(
            plan=make_plan(make_outline(1)),
            code=CODE,
            class_name="Demo",
            plan_ready=True,
            reviewed=True,
            slurm_job=existing,
        ),
        2: SceneState(
            plan=make_plan(make_outline(2)),
            code=CODE,
            class_name="Demo",
            plan_ready=True,
            reviewed=True,
        ),
    }
    for scene_id in (1, 2):
        (run_paths.scenes / f"scene_{scene_id}.py").write_text(CODE, encoding="utf-8")

    orchestrator._run_scheduler(ctx)

    assert slurm.max_remote_active == 1
    assert slurm.submitted == [2]
    assert all(state.rendered for state in ctx.scene_states.values())


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


def test_plan_review_blocks_coding_when_plan_is_invalid(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    plan_reviewer = RejectingPlanReviewer()
    orchestrator = make_orchestrator(
        monkeypatch,
        tmp_path,
        run_paths,
        plan_reviewer=plan_reviewer,
    )
    monkeypatch.setattr(settings, "MAX_PLAN_REVIEW_ROUNDS", 1)
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        dry_run=True,
        outlines=[make_outline(1)],
        continuity_bible=ContinuityBible(),
        plan_review_status="pending",
        scene_states={
            1: SceneState(
                plan=make_plan(make_outline(1)),
                plan_ready=True,
            )
        },
    )

    orchestrator._run_scheduler(ctx)

    state = ctx.scene_states[1]
    assert plan_reviewer.calls == 1
    assert state.failed is True
    assert state.plan_reviewed is False
    assert state.reviewed is False
    assert state.failure_category == "planning"
    assert state.code == ""


def test_plan_review_passes_before_code_review(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    events = []

    class OneRoundPlanReviewer:
        calls = 0

        def review(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return PlanReviewResult(
                    is_valid=False,
                    severity="major",
                    issues=[
                        {
                            "category": "math",
                            "confidence": "high",
                            "evidence_type": "calculation",
                            "evidence": "a²+b²=25 与 a²+b²=26",
                            "field": "computation",
                            "message": "公式两侧确定不等价",
                            "fix_instruction": "修正右侧表达式",
                        }
                    ],
                )
            return PlanReviewResult(is_valid=True)

    planner = ContinuityPlanner()
    plan_reviewer = OneRoundPlanReviewer()
    coder = FakeCoder()
    reviewer = FakeReviewer()
    orchestrator = make_orchestrator(
        monkeypatch,
        tmp_path,
        run_paths,
        planner=planner,
        plan_reviewer=plan_reviewer,
        coder=coder,
        reviewer=reviewer,
    )
    orchestrator.planner = planner
    orchestrator._callback = lambda event, data: events.append(event)
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        dry_run=True,
        outlines=[make_outline(1)],
        continuity_bible=ContinuityBible(),
        plan_review_status="pending",
        continuity_review_status="passed",
        scene_states={
            1: SceneState(
                plan=planner.plan_detail(make_outline(1), [make_outline(1)], "prompt"),
                plan_ready=True,
            )
        },
    )

    orchestrator._run_scheduler(ctx)

    assert plan_reviewer.calls == 2
    assert coder.calls
    assert ctx.scene_states[1].plan_reviewed is True
    assert ctx.scene_states[1].reviewed is True
    assert events.index("scene_plan_review_pass") < events.index("scene_coding")


def test_technical_spec_is_ready_before_code_generation(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    technical_planner = FakeTechnicalPlanner()
    events = []
    orchestrator = make_orchestrator(
        monkeypatch,
        tmp_path,
        run_paths,
        technical_planner=technical_planner,
    )
    orchestrator._callback = lambda event, data: events.append(event)
    outline = make_outline(1)
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        dry_run=True,
        outlines=[outline],
        continuity_bible=ContinuityBible(),
        plan_review_status="passed",
        continuity_review_status="passed",
        scene_states={1: SceneState(plan=make_plan(outline))},
    )

    orchestrator._run_scheduler(ctx)

    state = ctx.scene_states[1]
    assert technical_planner.calls == 1
    assert state.technical_status == "passed"
    assert state.technical_spec is not None
    assert events.index("scene_technical_ready") < events.index("scene_coding")


def test_invalid_technical_spec_is_regenerated_with_compile_feedback(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    calls: list[str] = []

    class RetryingTechnicalPlanner:
        def plan(self, scene_plan, *, renderer=None, feedback="", **kwargs):
            calls.append(feedback)
            if len(calls) == 1:
                return TechnicalSpec(scene_id=999, renderer=renderer or "cairo")
            return TechnicalSpec(scene_id=scene_plan.scene_id, renderer=renderer or "cairo")

    monkeypatch.setattr(settings, "MAX_TECHNICAL_SPEC_ATTEMPTS", 2)
    orchestrator = make_orchestrator(
        monkeypatch,
        tmp_path,
        run_paths,
        technical_planner=RetryingTechnicalPlanner(),
    )
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        dry_run=True,
        outlines=[make_outline(1)],
        continuity_bible=ContinuityBible(),
        plan_review_status="passed",
        continuity_review_status="passed",
        scene_states={1: SceneState(plan=make_plan(make_outline(1)))},
    )

    orchestrator._run_scheduler(ctx)

    assert len(calls) == 2
    assert calls[0] == ""
    assert "scene_id" in calls[1]
    assert ctx.scene_states[1].technical_status == "passed"


def test_continuity_review_is_a_barrier_before_coding(monkeypatch, tmp_path):
    """所有 Detail 完成后才审查连续性，审查通过后才进入编码。"""

    run_paths = make_paths(tmp_path)
    planner = ContinuityPlanner()
    continuity_reviewer = PassingContinuityReviewer()
    orchestrator = make_orchestrator(monkeypatch, tmp_path, run_paths, planner=planner)
    orchestrator.planner = planner
    monkeypatch.setattr(module, "ContinuityReviewerAgent", lambda: continuity_reviewer)
    events = []
    orchestrator._callback = lambda event, data: events.append(event)

    outlines = [make_outline(1), make_outline(2)]
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        dry_run=True,
        outlines=outlines,
        continuity_bible=ContinuityBible(),
        continuity_review_status="pending",
        scene_states={
            sid: SceneState(plan=make_plan(outline)) for sid, outline in enumerate(outlines, 1)
        },
    )
    planner.detail_calls = 0

    orchestrator._run_scheduler(ctx)

    assert planner.detail_calls == 2
    assert continuity_reviewer.calls == 1
    assert ctx.continuity_review_status == "passed"
    assert all(state.plan_ready and state.reviewed for state in ctx.scene_states.values())
    assert max(index for index, event in enumerate(events) if event == "scene_detailed") < min(
        index for index, event in enumerate(events) if event == "continuity_reviewing"
    )


def test_continuity_review_replans_only_affected_scenes(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    planner = ContinuityPlanner()
    continuity_reviewer = OneRoundContinuityReviewer()
    orchestrator = make_orchestrator(monkeypatch, tmp_path, run_paths, planner=planner)
    orchestrator.planner = planner
    monkeypatch.setattr(module, "ContinuityReviewerAgent", lambda: continuity_reviewer)

    outlines = [make_outline(1), make_outline(2), make_outline(3)]
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        dry_run=True,
        outlines=outlines,
        continuity_bible=ContinuityBible(),
        continuity_review_status="pending",
        scene_states={
            sid: SceneState(plan=planner.plan_detail(outline, outlines, "prompt"), plan_ready=True)
            for sid, outline in enumerate(outlines, 1)
        },
    )
    planner.detail_calls = 0

    orchestrator._run_scheduler(ctx)

    assert continuity_reviewer.calls == 2
    assert planner.detail_calls == 2
    assert ctx.continuity_review_status == "passed"
    assert all(state.reviewed for state in ctx.scene_states.values())


def test_continuity_replan_passes_latest_neighbor_snapshot(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    planner = ContextAwareContinuityPlanner()
    continuity_reviewer = OneRoundContinuityReviewer()
    orchestrator = make_orchestrator(monkeypatch, tmp_path, run_paths, planner=planner)
    orchestrator.planner = planner
    monkeypatch.setattr(module, "ContinuityReviewerAgent", lambda: continuity_reviewer)

    outlines = [make_outline(1), make_outline(2)]
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        dry_run=True,
        outlines=outlines,
        continuity_bible=ContinuityBible(),
        continuity_review_status="pending",
        scene_states={
            sid: SceneState(plan=planner.plan_detail(outline, outlines, "prompt"), plan_ready=True)
            for sid, outline in enumerate(outlines, 1)
        },
    )
    planner.continuity_contexts.clear()

    orchestrator._run_scheduler(ctx)

    assert ctx.continuity_review_status == "passed"
    assert len(planner.continuity_contexts) == 2
    assert all('"scene_id": 1' in context for context in planner.continuity_contexts)
    assert all('"scene_id": 2' in context for context in planner.continuity_contexts)


def test_continuity_replan_round_is_not_reset_by_contract_normalization(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    planner = ContinuityPlanner()

    class AlwaysRejectingContinuityReviewer:
        calls = 0

        def review(self, *args, **kwargs):
            self.calls += 1
            return ContinuityReviewResult(
                is_valid=False,
                summary="仍需修正",
                issues=[
                    {
                        "scene_ids": [1, 2],
                        "category": "transition",
                        "severity": "major",
                        "message": "交接对象仍不一致",
                        "fix_instruction": "统一交接对象",
                    }
                ],
            )

    continuity_reviewer = AlwaysRejectingContinuityReviewer()
    orchestrator = make_orchestrator(monkeypatch, tmp_path, run_paths, planner=planner)
    orchestrator.planner = planner
    monkeypatch.setattr(module, "ContinuityReviewerAgent", lambda: continuity_reviewer)
    monkeypatch.setattr(settings, "MAX_CONTINUITY_FIX_ROUNDS", 1)
    events = []
    orchestrator._callback = lambda event, data: events.append((event, data))

    outlines = [make_outline(1), make_outline(2)]
    ctx = PipelineContext(
        "prompt",
        paths=run_paths,
        dry_run=True,
        outlines=outlines,
        continuity_bible=ContinuityBible(),
        continuity_review_status="pending",
        scene_states={
            sid: SceneState(plan=planner.plan_detail(outline, outlines, "prompt"), plan_ready=True)
            for sid, outline in enumerate(outlines, 1)
        },
    )

    orchestrator._run_scheduler(ctx)

    assert continuity_reviewer.calls == 2
    assert ctx.continuity_review_round == 2
    assert ctx.continuity_review_status == "warning"
    assert all(state.reviewed for state in ctx.scene_states.values())
    assert not orchestrator._stop_event.is_set()
    assert any(event == "continuity_review_exhausted" for event, _ in events)
    assert any(event == "continuity_review_accepted_with_warning" for event, _ in events)


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
    assert all(st.plan_ready and st.reviewed for st in ctx.scene_states.values())
    manifest = run_paths.root / "manifest.json"
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
        def merge(self, videos, output_path, *, replace_existing=False, render_profile=None):
            # run-local 输出属于当前私有 run；合并器必须能够在校验通过后
            # 原子替换之前的结果，而不是因为旧文件存在而直接拒绝。
            assert replace_existing is True
            assert render_profile == ctx.render_profile
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x00" * 100)
            return output_path

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
# 9.5) resume 时核对恢复的 Slurm 作业: 已消失的清除重跑, 已完成的复用
# ---------------------------------------------------------------------------
def test_reconcile_restored_jobs_clears_gone_and_reuses_completed(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    slurm = FakeSlurm(
        run_paths,
        status_map={"1": "GONE", "2": "COMPLETED", "3": "RUNNING", "4": "UNKNOWN", "5": "UNKNOWN"},
    )
    orchestrator = make_orchestrator(monkeypatch, tmp_path, run_paths, slurm=slurm)
    ctx = PipelineContext("x", paths=run_paths)

    def make_job(sid, job_id, media_dir):
        return SlurmJob(
            job_id=job_id,
            scene_id=sid,
            script_path=run_paths.scenes / f"scene_{sid}.py",
            log_out=run_paths.logs / f"scene_{sid}.out",
            log_err=run_paths.logs / f"scene_{sid}.err",
            media_dir=media_dir,
            scene_class_name="Demo",
            submitted_at=0,
            code_sha256=sha256_text(CODE),
            render_profile=ctx.render_profile,
        )

    # 场景1: 作业已从集群消失 (GONE) → 清空引用, resume 后重新提交
    ctx.scene_states[1] = SceneState(
        plan=make_plan(make_outline(1)),
        code=CODE,
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
        slurm_job=make_job(1, "1", run_paths.videos / "s1"),
    )
    # 场景2: 上次已完成且视频存在 → 直接复用
    video_dir = run_paths.videos / "s2"
    video_dir.mkdir(parents=True)
    (video_dir / "Demo.mp4").write_bytes(b"fake")
    ctx.scene_states[2] = SceneState(
        plan=make_plan(make_outline(2)),
        code=CODE,
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
        slurm_job=make_job(2, "2", video_dir),
    )
    # 场景3: 仍在运行 → 保留继续监控
    ctx.scene_states[3] = SceneState(
        plan=make_plan(make_outline(3)),
        code=CODE,
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
        slurm_job=make_job(3, "3", run_paths.videos / "s3"),
    )
    # 场景4: 作业不可见 (UNKNOWN) 且无视频 → 保留引用，禁止重复提交
    ctx.scene_states[4] = SceneState(
        plan=make_plan(make_outline(4)),
        code=CODE,
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
        slurm_job=make_job(4, "4", run_paths.videos / "s4"),
    )
    # 场景5: UNKNOWN 即使已有文件也不越过调度器身份验证
    video_dir5 = run_paths.videos / "s5"
    video_dir5.mkdir(parents=True)
    (video_dir5 / "Demo.mp4").write_bytes(b"fake")
    ctx.scene_states[5] = SceneState(
        plan=make_plan(make_outline(5)),
        code=CODE,
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
        slurm_job=make_job(5, "5", video_dir5),
    )

    orchestrator._reconcile_restored_jobs(ctx)

    assert ctx.scene_states[1].slurm_job is None
    assert ctx.scene_states[2].rendered is True
    assert ctx.scene_states[2].slurm_job is not None
    assert ctx.scene_states[3].slurm_job is not None
    assert ctx.scene_states[3].rendered is False
    # UNKNOWN 一律保留 Job ID，直到确认终态或成功取消。
    assert ctx.scene_states[4].slurm_job is not None
    assert ctx.scene_states[4].rendered is False
    assert ctx.scene_states[5].slurm_job is not None
    assert ctx.scene_states[5].rendered is False


# ---------------------------------------------------------------------------
# 10) 连续相同渲染错误 → 判定环境问题提前放弃, 并附错误日志尾部
# ---------------------------------------------------------------------------
def test_identical_render_error_gives_up_early(monkeypatch, tmp_path):
    run_paths = make_paths(tmp_path)
    error_log = (
        "Rendering:   0%|          | 0/60 [00:00<?, ?it/s]\n"
        "Traceback (most recent call last):\n"
        '  File "scene_1.py", line 42, in construct\n'
        "ValueError: something deterministic\n"
    )
    slurm = FakeSlurm(
        run_paths,
        status_map=lambda job_id: "FAILED",
        error_log=error_log,
    )
    autofixer = FakeAutoFixer()
    orchestrator = make_orchestrator(
        monkeypatch,
        tmp_path,
        run_paths,
        slurm=slurm,
        autofixer=autofixer,
    )
    monkeypatch.setattr(module.settings, "MAX_FIX_IDENTICAL_ERRORS", 2)

    ctx = PipelineContext("x", paths=run_paths, outlines=[make_outline(1)])
    ctx.scene_states[1] = SceneState(
        plan=make_plan(make_outline(1)),
        code=CODE,
        class_name="Demo",
        plan_ready=True,
        reviewed=True,
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

    fp1 = Orchestrator._error_fingerprint("Traceback line 42: ValueError at frame 1234")
    fp2 = Orchestrator._error_fingerprint("Traceback line 99: ValueError at frame 9999")
    assert fp1 == fp2  # 数字不同 → 指纹相同

    fp3 = Orchestrator._error_fingerprint("Traceback line 42: KeyError at frame 1234")
    assert fp1 != fp3  # 错误类型不同 → 指纹不同


def test_error_fingerprint_normalizes_attempt_paths_and_hex_names():
    first = Orchestrator._error_fingerprint(
        "ValueError: failed in workspace/runs/20260811-120000-abcdef12/"
        "videos/scene_1/attempt_0123456789ab/Tex/aa11bb22cc33dd44.svg"
    )
    second = Orchestrator._error_fingerprint(
        "ValueError: failed in workspace/runs/20260812-130000-1234abcd/"
        "videos/scene_1/attempt_fedcba987654/Tex/ffeeddccbbaa9988.svg"
    )

    assert first == second
