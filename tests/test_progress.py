from kd1_anime.agents.progress import ProgressSnapshot, classify_progress


def test_progress_is_unknown_without_previous_snapshot():
    current = ProgressSnapshot.from_values("code", error_fingerprint="error")
    assert classify_progress(None, current) == "unknown"


def test_progress_detects_identical_code_and_error():
    previous = ProgressSnapshot.from_values("code", error_fingerprint="error")
    current = ProgressSnapshot.from_values("code", error_fingerprint="error")
    assert classify_progress(previous, current) == "unchanged"


def test_progress_treats_changed_error_as_improvement():
    previous = ProgressSnapshot.from_values("code", error_fingerprint="error-a")
    current = ProgressSnapshot.from_values("code", error_fingerprint="error-b")
    assert classify_progress(previous, current) == "improved"


def test_progress_marks_more_deterministic_issues_as_regression():
    previous = ProgressSnapshot.from_values("old", issue_count=1)
    current = ProgressSnapshot.from_values("new", issue_count=2)
    assert classify_progress(previous, current) == "regressed"
