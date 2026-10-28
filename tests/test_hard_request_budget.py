import pytest

from kd1_anime.request_budget import RequestBudget, RequestBudgetExceeded, RequestReservation


def reservation(*, prompt=10, output=20, cost=0.5):
    return RequestReservation(
        stage="code",
        profile="main",
        model="test",
        estimated_prompt_tokens=prompt,
        reserved_output_tokens=output,
        reserved_cost_usd=cost,
        input_price_usd_per_million=0,
        output_price_usd_per_million=25_000,
    )


def test_hard_limits_reserve_full_envelope_before_dispatch_and_survive_roundtrip():
    budget = RequestBudget(run_limit=10, token_limit=60, cost_limit_usd=1)
    budget.reserve(reservation())
    restored = RequestBudget.model_validate_json(budget.model_dump_json())
    restored.reserve(reservation())
    assert len(restored.reservations) == 2
    with pytest.raises(RequestBudgetExceeded):
        restored.reserve(reservation())
    assert len(restored.reservations) == 2
    assert not restored.integrity_errors()


def test_unknown_price_or_unbounded_output_fails_closed_under_hard_cap():
    budget = RequestBudget(cost_limit_usd=1, token_limit=100)
    with pytest.raises(RequestBudgetExceeded):
        budget.reserve(reservation(cost=None))
    with pytest.raises(RequestBudgetExceeded):
        budget.reserve(reservation(output=None))
    assert not budget.reservations


def test_unknown_outcome_remains_reserved_and_cannot_be_spent_twice():
    budget = RequestBudget(token_limit=40, cost_limit_usd=0.75)
    budget.reserve(reservation())
    restored = RequestBudget.model_validate(budget.model_dump(mode="python"))
    with pytest.raises(RequestBudgetExceeded):
        restored.reserve(reservation())
    assert len(restored.reservations) == 1


def test_clarification_reserves_before_provider_and_records_reported_usage(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kd1_anime.agent_runtime.session import SessionStore
    from kd1_anime.agents.base import BaseAgent
    from kd1_anime.application import Application
    from kd1_anime.config import settings

    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(settings, "LLM_MODEL", "priced")
    monkeypatch.setattr(settings, "LLM_MODEL_PRICES_USD", {"priced": (1.0, 2.0)})
    monkeypatch.setattr(settings, "MAX_RUN_LLM_COST_USD", 1.0)
    app = Application()
    session_id = app.create_session("Explain")
    store = SessionStore(tmp_path / "sessions")
    calls = []

    def create(**kwargs):
        snapshot = store.load(session_id)
        assert len(snapshot.draft_budget.reservations) == 1
        assert snapshot.draft_budget.reservations[0].prompt_sha256
        calls.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3, total_tokens=7)
        )

    agent = BaseAgent()
    agent.request_context = app.session_request_context(session_id)
    agent._create_completion(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        messages=[{"role": "user", "content": "Explain"}],
        max_tokens=20,
    )
    snapshot = store.load(session_id)
    assert len(calls) == 1
    assert snapshot.draft_usage[0].usage_source == "provider_reported"
    assert snapshot.draft_usage[0].cost_status == "estimated" and snapshot.draft_usage[0].cost > 0


def test_hard_cost_limit_refuses_unpriced_provider_before_transport(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kd1_anime.agents.base import BaseAgent
    from kd1_anime.application import Application
    from kd1_anime.config import settings

    monkeypatch.setattr(settings, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(settings, "LLM_MODEL", "unpriced")
    monkeypatch.setattr(settings, "MAX_RUN_LLM_COST_USD", 1.0)
    app = Application()
    agent = BaseAgent()
    agent.request_context = app.session_request_context(app.create_session("Explain"))
    calls = []
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs))
        )
    )
    with pytest.raises(RequestBudgetExceeded):
        agent._create_completion(
            client, messages=[{"role": "user", "content": "Explain"}], max_tokens=20
        )
    assert not calls


def test_provider_usage_overrun_blocks_later_requests():
    from kd1_anime.agent_runtime.contracts import UsageReceipt

    budget = RequestBudget(token_limit=100, cost_limit_usd=1)
    first = reservation()
    budget.reserve(first)
    usage = UsageReceipt(
        request_id=first.request_id,
        usage_source="provider_reported",
        prompt_tokens=10,
        completion_tokens=21,
        total_tokens=31,
    )
    with pytest.raises(RequestBudgetExceeded):
        budget.check_reported_usage((usage,))
    assert len(budget.reservations) == 1


def test_hard_cap_rejects_underpriced_or_unpriced_reservation():
    budget = RequestBudget(cost_limit_usd=1)
    with pytest.raises(RequestBudgetExceeded):
        budget.reserve(reservation(cost=0.01))
    with pytest.raises(RequestBudgetExceeded):
        budget.reserve(reservation().model_copy(update={"input_price_usd_per_million": None}))
    assert not budget.reservations
