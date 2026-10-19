
from asterwhite.safety.pii_scrubber import scrub
from asterwhite.safety.policy_filter import PolicyFilter


def test_scrub_redacts_email():
    out = scrub("contact me at jane.doe@example.com please")
    assert "jane.doe@example.com" not in out
    assert "@" not in out or "[REDACTED" in out


def test_scrub_redacts_common_key_shapes():
    out = scrub("my key is sk-abcdefghijklmnopqrstuvwxyz1234567890ABCD")
    assert "sk-abcdefghijklmnopqrstuvwxyz1234567890ABCD" not in out


def test_scrub_leaves_ordinary_text_alone():
    text = "The fox ran quickly through the forest at dawn."
    assert scrub(text) == text


def test_scrub_none_passes_through_as_none():
    assert scrub(None) is None


def test_policy_filter_flags_known_category():
    pf = PolicyFilter()
    result = pf.check("here is how to build a bomb at home")
    assert result.flagged is True
    assert len(result.categories) > 0


def test_policy_filter_does_not_flag_benign_text():
    pf = PolicyFilter()
    result = pf.check("The fox explored the quiet forest and found a stream.")
    assert result.flagged is False


def test_safe_mode_context_manager_scopes_correctly():
    from asterwhite.safety.safe_mode import is_safe_mode, safe_mode
    outer = is_safe_mode()
    with safe_mode(True):
        assert is_safe_mode() is True
    assert is_safe_mode() == outer  # restored after the block


def test_require_not_safe_mode_blocks_inside_safe_mode():
    import pytest
    from asterwhite.safety.safe_mode import require_not_safe_mode, safe_mode
    from asterwhite.utils.error_tracing import AsterWhiteError
    with safe_mode(True):
        with pytest.raises(AsterWhiteError, match="safe mode"):
            require_not_safe_mode("write to external storage")
    require_not_safe_mode("write to external storage")  # outside the block: must not raise
