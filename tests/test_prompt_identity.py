import pytest

from kd1_anime.prompt_identity import (
    GenerationIdentity,
    capture_generation_identity,
    final_messages_sha256,
    verify_generation_identity,
)


def test_generation_identity_is_stable_but_final_message_hash_tracks_exact_inputs():
    identity = capture_generation_identity()
    first = [{"role": "system", "content": "Use exact math"}, {"role": "user", "content": "圆"}]
    reordered_keys = [
        {"content": "Use exact math", "role": "system"},
        {"content": "圆", "role": "user"},
    ]
    assert final_messages_sha256(first) == final_messages_sha256(reordered_keys)
    assert final_messages_sha256(first) != final_messages_sha256(
        [*first, {"role": "user", "content": "半径=2"}]
    )
    assert capture_generation_identity() == identity
    verify_generation_identity(identity)


@pytest.mark.parametrize(
    "field",
    [
        "prompt_pack_sha256",
        "response_schema_sha256",
        "context_builder_sha256",
        "generator_sha256",
        "validator_sha256",
        "policy_sha256",
    ],
)
def test_incompatible_generation_component_is_rejected(field):
    current = capture_generation_identity()
    previous = current.model_copy(update={field: "0" * 64})
    with pytest.raises(ValueError, match=field):
        verify_generation_identity(previous)


def test_missing_old_identity_is_never_filled_with_current_values():
    with pytest.raises(ValueError):
        verify_generation_identity(None)
    with pytest.raises(ValueError):
        GenerationIdentity.model_validate({})


def test_final_message_hash_does_not_accept_hidden_analysis_or_unknown_envelopes():
    with pytest.raises(ValueError):
        final_messages_sha256(
            [{"role": "assistant", "content": "visible", "reasoning_content": "private"}]
        )
    with pytest.raises(ValueError):
        final_messages_sha256([{"role": "unknown", "content": "text"}])
