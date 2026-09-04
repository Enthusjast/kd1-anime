from kd1_anime.agents.failure_corpus import FailureCase, FailureCaseStore


def case(index: int, category: str = "python"):
    return FailureCase(
        category=category,
        fingerprint=f"fp-{index}",
        error_type="ValueError",
        message=f"failure {index} token=secret-key",
        original_code_sha256=f"old-{index}",
        fixed_code_sha256=f"new-{index}",
        verification="validated",
        patch_summary="replace bad call",
        source_run_id="20260904-120000-1234abcd",
        created_at=float(index + 1),
    )


def test_failure_case_store_is_bounded_and_searchable(tmp_path):
    store = FailureCaseStore(tmp_path / "cases.sqlite3", max_per_category=2)

    for index in range(3):
        assert store.record(case(index)) is True

    cases = store.search(category="python", limit=10)

    assert [item.fingerprint for item in cases] == ["fp-2", "fp-1"]
    assert "secret-key" not in store.context(category="python")
    assert "replace bad call" in store.context(category="python")


def test_failure_case_store_filters_by_fingerprint(tmp_path):
    store = FailureCaseStore(tmp_path / "cases.sqlite3")
    store.record(case(1, category="latex"))
    store.record(case(2, category="python"))

    matches = store.search(category="latex", fingerprint="fp-1")

    assert len(matches) == 1
    assert matches[0].category == "latex"
