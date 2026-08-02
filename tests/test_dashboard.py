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

        # coded 完成
        dash.on_event("scene_coded", {"scene_id": 1, "file_path": "s1.py"})
        assert dash.scenes[1].state == "completed"

        # review pass
        dash.on_event("scene_review_pass", {"scene_id": 1})
        assert dash.scenes[1].stage == "审查"

        # submitted
        dash.on_event("scene_submitted", {"scene_id": 1, "job_id": "123"})
        assert dash.scenes[1].stage == "渲染"

        # rendered
        dash.on_event("scene_rendered", {"scene_id": 1})
        assert dash.scenes[1].state == "completed"

    def test_failed_and_give_up(self):
        dash = SceneDashboard()
        dash.live = MagicMock()
        dash.on_event("plan_complete", {"scenes": [MagicMock(scene_id=1, title="S1")]})
        dash.on_event("scene_failed", {"scene_id": 1, "reason": "LLM timeout"})
        assert dash.scenes[1].state == "failed"
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
