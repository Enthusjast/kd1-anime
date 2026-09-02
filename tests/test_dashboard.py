"""SceneDashboard 仪表盘测试。"""

from unittest.mock import MagicMock

from kd1_anime.dashboard import SceneDashboard, SceneStatus, suppress_agent_logs


class TestSceneStatus:
    def test_icon_mapping(self):
        assert SceneStatus(scene_id=1, state="pending").icon == "⏳"
        assert SceneStatus(scene_id=1, state="running").icon == "⟳"
        assert SceneStatus(scene_id=1, state="completed").icon == "✓"
        assert SceneStatus(scene_id=1, state="failed").icon == "✗"

    def test_color_mapping(self):
        assert SceneStatus(scene_id=1, state="pending").color == "dim"
        assert SceneStatus(scene_id=1, state="running").color == "cyan"
        assert SceneStatus(scene_id=1, state="completed").color == "green"
        assert SceneStatus(scene_id=1, state="failed").color == "red"

    def test_invalidate_keeps_legacy_code_review_stage_alias(self):
        status = SceneStatus(scene_id=1, done=["分镜", "编码", "审查", "渲染"])

        status.invalidate_from("审查", ("分镜", "编码", "审查", "渲染"))

        assert status.done == ["分镜", "编码"]


class TestSceneDashboard:
    def test_event_mapping(self):
        dash = SceneDashboard()
        dash.live = MagicMock()  # 模拟 Live 已激活

        # plan_complete 初始化场景
        scene1 = MagicMock(scene_id=1, title="Scene 1")
        scene2 = MagicMock(scene_id=2, title="Scene 2")
        dash.on_event("plan_complete", {"scenes": [scene1, scene2]})
        assert dash.total == 2
        assert dash.scenes[1].title == "Scene 1"

        # coding 状态
        dash.on_event("scene_coding", {"scene_id": 1, "title": "Scene 1"})
        assert dash.scenes[1].state == "running"
        assert dash.scenes[1].stage == "编码"

        # coded 完成: 场景仍未完成 (只有渲染完成才置绿)
        dash.on_event("scene_coded", {"scene_id": 1, "file_path": "s1.py"})
        assert dash.scenes[1].state == "running"
        assert "编码" in dash.scenes[1].done

        # review pass
        dash.on_event("scene_review_pass", {"scene_id": 1})
        assert "审查" in dash.scenes[1].done

        # submitted
        dash.on_event("scene_submitted", {"scene_id": 1, "job_id": "123"})
        assert dash.scenes[1].stage == "渲染"

        # rendered
        dash.on_event("scene_rendered", {"scene_id": 1})
        assert dash.scenes[1].state == "completed"

    def test_scene_stays_in_progress_until_rendered(self):
        """分镜/编码/审查完成都不应让场景变绿, 只有渲染完成才算完成。"""
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event("plan_complete", {"scenes": [MagicMock(scene_id=1, title="S1")]})
        dash.on_event("scene_detailing", {"scene_id": 1})
        dash.on_event("scene_detailed", {"scene_id": 1})
        assert dash.scenes[1].state == "running"
        assert dash.scenes[1].done == ["分镜"]

        dash.on_event("scene_coding", {"scene_id": 1})
        dash.on_event("scene_coded", {"scene_id": 1})
        assert dash.scenes[1].state == "running"
        assert dash.scenes[1].done == ["分镜", "编码"]

        dash.on_event("scene_review_pass", {"scene_id": 1})
        assert dash.scenes[1].state == "running"
        assert dash.scenes[1].done == ["分镜", "编码", "审查"]

        dash.on_event("scene_submitted", {"scene_id": 1, "job_id": "9"})
        assert dash.scenes[1].state == "running"
        assert dash.scenes[1].stage == "渲染"

        dash.on_event("scene_rendered", {"scene_id": 1})
        assert dash.scenes[1].state == "completed"
        assert dash.scenes[1].done == ["分镜", "编码", "审查", "渲染"]

    def test_plan_review_and_code_review_are_distinct_stages(self):
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event("plan_complete", {"scenes": [MagicMock(scene_id=1, title="S1")]})
        dash.on_event("plan_reviewing", {"scene_count": 1})
        assert dash.stage_label == "计划正确性审查"
        dash.on_event("scene_detailed", {"scene_id": 1})
        dash.on_event("scene_plan_reviewing", {"scene_id": 1})
        assert dash.scenes[1].stage == "计划审查"
        assert "计划审查⟳" in str(dash.scenes[1].render_row()[2])

        dash.on_event("scene_plan_review_pass", {"scene_id": 1})
        assert "计划审查" in dash.scenes[1].done
        dash.on_event("stage_start", {"stage": "technical"})
        assert dash.stage_label == "技术实现设计"
        dash.on_event("scene_technical_planning", {"scene_id": 1})
        assert dash.scenes[1].stage == "技术设计"
        assert "技术设计⟳" in str(dash.scenes[1].render_row()[2])
        dash.on_event("scene_technical_ready", {"scene_id": 1})
        assert "技术设计" in dash.scenes[1].done
        dash.on_event("scene_coding", {"scene_id": 1})
        dash.on_event("scene_coded", {"scene_id": 1})
        dash.on_event("scene_reviewing", {"scene_id": 1})
        assert dash.scenes[1].stage == "代码审查"
        assert "代码审查⟳" in str(dash.scenes[1].render_row()[2])

    def test_visual_enabled_scene_stays_in_progress_until_visual_gate_accepts(self):
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event(
            "plan_complete",
            {
                "scenes": [MagicMock(scene_id=1, title="S1")],
                "visual_enabled": True,
            },
        )

        dash.on_event("scene_rendered", {"scene_id": 1})
        assert dash.scenes[1].state == "running"
        assert dash.scenes[1].done == ["渲染"]
        assert "视觉" in dash.stages

        dash.on_event("scene_visual_evaluating", {"scene_id": 1})
        assert dash.scenes[1].stage == "视觉"
        dash.on_event("scene_visual_pass", {"scene_id": 1, "score": 4.2})
        assert dash.scenes[1].state == "completed"
        assert dash.scenes[1].done == ["渲染", "视觉"]

    def test_visual_unknown_is_an_accepted_warning_terminal(self):
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event(
            "plan_complete",
            {
                "scenes": [MagicMock(scene_id=1, title="S1")],
                "visual_enabled": True,
            },
        )
        dash.on_event("scene_rendered", {"scene_id": 1})
        dash.on_event("scene_visual_unknown", {"scene_id": 1, "reason": "timeout"})

        assert dash.scenes[1].state == "warning"
        assert dash.scenes[1].icon == "⚠"
        assert "未知" in dash.scenes[1].message

    def test_safe_fallback_is_visible_without_marking_scene_complete(self):
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event("plan_complete", {"scenes": [MagicMock(scene_id=1, title="S1")]})
        dash.on_event(
            "scene_safe_fallback",
            {"scene_id": 1, "reason": "几何方案无法验证"},
        )

        assert dash.scenes[1].safe_fallback_used is True
        assert dash.scenes[1].state == "warning"
        assert dash.scenes[1].icon == "⚠"
        assert "保守方案" in dash.scenes[1].render_row()[3].plain

        dash.on_event("scene_coding", {"scene_id": 1})
        assert dash.scenes[1].state == "running"
        assert "保守方案" in dash.scenes[1].render_row()[3].plain

    def test_continuity_exhaustion_is_warning_and_does_not_fail_scene(self):
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event(
            "plan_complete",
            {"scenes": [MagicMock(scene_id=1, title="S1"), MagicMock(scene_id=2, title="S2")]},
        )
        dash.on_event(
            "continuity_review_exhausted",
            {
                "scene_ids": [1, 2],
                "reason": "达到最大连续性修正轮数，已继续生成",
                "round": 3,
                "max_rounds": 2,
            },
        )

        assert dash.stage_label == "连续性审查达到上限，继续生成"
        assert all(scene.state == "warning" for scene in dash.scenes.values())
        assert all(scene.icon == "⚠" for scene in dash.scenes.values())

        dash.on_event("scene_coding", {"scene_id": 1})
        assert dash.scenes[1].state == "running"
        assert dash.scenes[2].state == "warning"

    def test_continuity_resume_recheck_has_explicit_stage(self):
        dash = SceneDashboard()
        dash.live = MagicMock()

        dash.on_event("continuity_review_resume_recheck", {"max_rechecks": 1})

        assert dash.stage == "continuity"
        assert dash.stage_label == "恢复：重新检查连续性"

    def test_pipeline_row_shows_done_and_current_stages(self):
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event("plan_complete", {"scenes": [MagicMock(scene_id=1, title="S1")]})
        dash.on_event("scene_detailed", {"scene_id": 1})
        dash.on_event("scene_coding", {"scene_id": 1})
        row = dash.scenes[1].render_row()
        pipeline = str(row[2])
        assert "分镜✓" in pipeline
        assert "编码⟳" in pipeline
        assert "渲染·" in pipeline
        assert "审查·" in pipeline

    def test_regeneration_clears_stale_downstream_checkmarks(self):
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event("plan_complete", {"scenes": [MagicMock(scene_id=1, title="S1")]})
        dash.on_event("scene_detailed", {"scene_id": 1})
        dash.on_event("scene_coded", {"scene_id": 1})
        dash.on_event("scene_review_pass", {"scene_id": 1})
        dash.on_event("scene_rendered", {"scene_id": 1})

        dash.on_event("scene_coding", {"scene_id": 1, "title": "S1"})

        assert dash.scenes[1].state == "running"
        assert dash.scenes[1].stage == "编码"
        assert dash.scenes[1].done == ["分镜"]
        pipeline = str(dash.scenes[1].render_row()[2])
        assert "编码⟳" in pipeline

    def test_visual_plan_repair_is_shown_as_plan_review_not_code_repair(self):
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event(
            "plan_complete",
            {"scenes": [MagicMock(scene_id=1, title="S1")], "visual_enabled": True},
        )
        dash.on_event("scene_detailed", {"scene_id": 1})
        dash.on_event("scene_plan_review_pass", {"scene_id": 1})
        dash.on_event("scene_technical_ready", {"scene_id": 1})
        dash.on_event("scene_coded", {"scene_id": 1})
        dash.on_event("scene_review_pass", {"scene_id": 1})
        dash.on_event("scene_rendered", {"scene_id": 1})
        dash.on_event("scene_visual_evaluating", {"scene_id": 1})

        dash.on_event(
            "scene_visual_plan_fixing",
            {"scene_id": 1, "target": "Planner", "attempt": 1, "max_attempts": 2},
        )

        assert dash.scenes[1].state == "running"
        assert dash.scenes[1].stage == "计划审查"
        assert dash.scenes[1].done == ["分镜"]
        assert "计划审查⟳" in str(dash.scenes[1].render_row()[2])

        dash.on_event("scene_plan_repair_requested", {"scene_id": 1, "target": "planner"})
        assert dash.scenes[1].stage == "计划审查"

    def test_failed_and_give_up(self):
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event("plan_complete", {"scenes": [MagicMock(scene_id=1, title="S1")]})
        dash.on_event("scene_failed", {"scene_id": 1, "reason": "LLM timeout"})
        assert dash.scenes[1].state == "failed"
        assert dash.scenes[1].stage == ""
        assert "LLM timeout" in dash.scenes[1].message

        dash.on_event("scene_give_up", {"scene_id": 1})
        assert dash.scenes[1].message == "已放弃"

    def test_inactive_dashboard_ignores_events(self):
        dash = SceneDashboard()  # live 为 None（未激活），事件被忽略不崩溃
        dash.on_event("stage_start", {"stage": "coding"})
        dash.on_event("scene_coding", {"scene_id": 1, "title": "S1"})
        assert dash.stage == ""
        assert 1 not in dash.scenes

    def test_render_without_live(self):
        dash = SceneDashboard()
        panel = dash._render()
        assert panel is not None


def test_suppress_flag_off_by_default():
    assert suppress_agent_logs() is False


def test_live_uses_dynamic_render_callback_for_elapsed_time(monkeypatch):
    """Rich 自动刷新必须重新调用 _render，而不是复用静态 Panel。"""
    import io
    import types

    from rich.console import Console as RichConsole

    import kd1_anime.dashboard as dashboard_module

    class FakeLive:
        def __init__(self, *, get_renderable, **_kwargs):
            self.get_renderable = get_renderable
            self.refresh_count = 0

        def start(self):
            return None

        def stop(self):
            return None

        def refresh(self):
            self.refresh_count += 1
            # 模拟 Rich 的自动刷新线程：每次刷新都重新取得 renderable。
            self.get_renderable()

    monkeypatch.setattr(dashboard_module, "Live", FakeLive)
    monkeypatch.setattr(
        dashboard_module,
        "sys",
        types.SimpleNamespace(stdout=types.SimpleNamespace(isatty=lambda: True)),
    )
    dash = SceneDashboard()
    assert dash.start() is True
    try:
        live = dash.live
        assert isinstance(live, FakeLive)
        assert getattr(live.get_renderable, "__self__", None) is dash
        assert getattr(live.get_renderable, "__func__", None) is SceneDashboard._render

        dash.on_event("plan_complete", {"scenes": [MagicMock(scene_id=1, title="S1")]})
        dash.on_event("scene_coding", {"scene_id": 1})
        dash.scenes[1].started_at = 100.0

        now = [112.0]
        monkeypatch.setattr(dashboard_module.time, "time", lambda: now[0])
        first_buffer = io.StringIO()
        RichConsole(file=first_buffer, width=120, force_terminal=False).print(live.get_renderable())
        assert "12s" in first_buffer.getvalue()

        now[0] = 119.0
        second_buffer = io.StringIO()
        RichConsole(file=second_buffer, width=120, force_terminal=False).print(
            live.get_renderable()
        )
        assert "19s" in second_buffer.getvalue()
        assert live.refresh_count >= 2  # plan_complete + scene_coding
    finally:
        dash.stop()


class TestSceneDashboardEvents:
    def test_reused_scene(self):
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event("plan_complete", {"scenes": [MagicMock(scene_id=1, title="S1")]})
        dash.on_event("scene_reused", {"scene_id": 1})
        assert dash.scenes[1].state == "completed"
        assert dash.scenes[1].stage == "渲染"
        assert dash.scenes[1].message == "复用旧视频"

    def test_reused_scene_is_emitted_after_review(self):
        """增量复用必须在审查事件之后落地, 不得把终态又刷回进行中。"""
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event("plan_complete", {"scenes": [MagicMock(scene_id=1, title="S1")]})
        dash.on_event("scene_review_pass", {"scene_id": 1})
        dash.on_event("scene_reused", {"scene_id": 1})

        assert dash.scenes[1].state == "completed"
        assert dash.scenes[1].done == ["审查", "渲染"]

    def test_run_started_sets_run_id_and_timer(self):
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event("run_started", {"run_id": "20260802-000000-1234abcd"})
        assert dash.run_id == "20260802-000000-1234abcd"
        assert dash.started_at > 0

    def test_running_phase_aggregation(self):
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event(
            "plan_complete",
            {"scenes": [MagicMock(scene_id=i, title=f"S{i}") for i in (1, 2, 3)]},
        )
        dash.on_event("scene_detailing", {"scene_id": 1, "title": "S1"})
        dash.on_event("scene_coding", {"scene_id": 2, "title": "S2"})
        dash.on_event("scene_coding", {"scene_id": 3, "title": "S3"})
        assert dash._running_phases() == {"分镜": 1, "编码": 2}

    def test_header_shows_phase_aggregation(self):
        import io

        from rich.console import Console as RichConsole

        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event(
            "plan_complete",
            {"scenes": [MagicMock(scene_id=i, title=f"S{i}") for i in (1, 2)]},
        )
        dash.on_event("scene_coding", {"scene_id": 1, "title": "S1"})
        dash.on_event("scene_rendered", {"scene_id": 2})
        panel = dash._render()
        buf = io.StringIO()
        RichConsole(file=buf, width=120, force_terminal=False).print(panel)
        text = buf.getvalue()
        assert "运行中" in text
        assert "编码×1" in text
        assert "完成 1/2" in text

    def test_active_rag_with_historical_warning_is_not_displayed_as_degraded(self):
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event(
            "rag_status",
            {
                "status": "active",
                "warning": "恢复运行：RAG 索引或模型已变化，将重新检索",
                "embedding_model": "embedding",
                "reranker_model": "reranker",
            },
        )

        assert dash.rag_status == "active"
        assert dash.rag_models == "E:embedding R:reranker"
        assert "degraded" not in str(dash._render())

        dash.on_event(
            "rag_status",
            {
                "status": "degraded",
                "warning": "索引不存在",
                "embedding_model": "embedding",
                "reranker_model": "reranker",
            },
        )
        assert "degraded" in dash.rag_models

    def test_rendering_scene_shows_elapsed(self):
        import time

        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event("plan_complete", {"scenes": [MagicMock(scene_id=1, title="S1")]})
        dash.on_event("scene_submitted", {"scene_id": 1, "job_id": "42"})
        dash.scenes[1].started_at = time.time() - 12
        row = dash.scenes[1].render_row()
        assert "12s" in str(row[-1])


def test_quiet_flag_follows_activation():
    from kd1_anime.dashboard import _state, quiet

    assert quiet() is False
    dash = SceneDashboard()
    dash.live = MagicMock()  # 模拟已激活
    with _state.lock:
        _state.active = True
        _state.current = dash
    try:
        assert quiet() is True
        assert suppress_agent_logs() is True
    finally:
        with _state.lock:
            _state.active = False
            _state.current = None
