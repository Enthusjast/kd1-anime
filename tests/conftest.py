import pytest

from kd1_anime.config import Settings, settings


@pytest.fixture(autouse=True)
def isolate_global_settings(monkeypatch):
    """测试不得继承开发者当前目录 .env 或用户级集群配置。"""

    defaults = Settings(_env_file=None)
    for field_name in Settings.model_fields:
        monkeypatch.setattr(settings, field_name, getattr(defaults, field_name))


@pytest.fixture(autouse=True)
def isolate_run_endpoint_probes(monkeypatch):
    """Ordinary pipeline tests use deterministic agents and never contact endpoints."""
    monkeypatch.setattr(
        "kd1_anime.orchestrator.Orchestrator._probe_run_endpoints", lambda self, ctx: None
    )
