from kd1_anime.agents.math_verifier import verify_expression_samples


def test_sampled_verification_is_reproducible_for_equivalent_non_polynomials():
    first = verify_expression_samples("x/(x+1)", "x/(x+1)", seed=7)
    second = verify_expression_samples("x/(x+1)", "x/(x+1)", seed=7)

    assert first.status == "sampled"
    assert first.to_dict() == second.to_dict()
    assert first.valid_samples > 0


def test_sampled_verification_reports_a_counterexample():
    result = verify_expression_samples("x/(x+1)", "x", seed=7)

    assert result.status == "counterexample"
    assert result.counterexample is not None
    assert result.difference is not None and result.difference > 0


def test_sampled_verification_does_not_execute_function_calls():
    result = verify_expression_samples("__import__('os')", "0")

    assert result.status == "unknown"


def test_sampled_verification_skips_undefined_samples():
    result = verify_expression_samples("1/(x-x)", "0", samples=4)

    assert result.status == "unknown"
    assert result.valid_samples == 0
