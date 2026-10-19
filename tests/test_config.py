import pytest

from asterwhite.config import AsterWhiteConfig, AsterWhiteConfigError, resolve_secret, validate_config
from asterwhite.profiles import REQUIRED_PROFILES, list_profiles, load_profile, load_profile_raw


def test_tiny_config_validates():
    validate_config(AsterWhiteConfig.tiny_for_testing())


def test_rejects_indivisible_heads():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.n_heads = 3  # d_model=32 not divisible by 3
    with pytest.raises(AsterWhiteConfigError, match="n_heads"):
        validate_config(cfg)


def test_rejects_gqa_mismatch():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.n_heads = 4
    cfg.backbone.n_kv_heads = 3
    with pytest.raises(AsterWhiteConfigError, match="n_kv_heads"):
        validate_config(cfg)


def test_rejects_bad_amp_dtype():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.training.amp_dtype = "int4"
    with pytest.raises(AsterWhiteConfigError, match="amp_dtype"):
        validate_config(cfg)


def test_rejects_micro_batch_exceeding_batch():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.training.micro_batch_size = 100
    cfg.training.batch_size = 2
    with pytest.raises(AsterWhiteConfigError, match="micro_batch_size"):
        validate_config(cfg)


def test_rejects_grpo_group_size_of_one():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.grpo.group_size = 1
    with pytest.raises(AsterWhiteConfigError, match="group_size"):
        validate_config(cfg)


def test_to_dict_from_dict_round_trip_preserves_nested_types():
    cfg = AsterWhiteConfig.tiny_for_testing()
    restored = AsterWhiteConfig.from_dict(cfg.to_dict())
    assert restored == cfg
    assert isinstance(restored.backbone, type(cfg.backbone))
    assert isinstance(restored.inference, type(cfg.inference))


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(AsterWhiteConfigError, match="unknown config key"):
        AsterWhiteConfig.from_dict({"totally_made_up_field": 1})


def test_resolve_secret_reads_only_from_environment(monkeypatch):
    monkeypatch.delenv("ASTERWHITE_SECRET_MY_KEY", raising=False)
    assert resolve_secret("my_key") is None
    monkeypatch.setenv("ASTERWHITE_SECRET_MY_KEY", "value123")
    assert resolve_secret("my_key") == "value123"


@pytest.mark.parametrize("name", REQUIRED_PROFILES)
def test_every_required_profile_loads_and_validates(name):
    cfg = load_profile(name)
    validate_config(cfg)  # load_profile already validates; re-asserting is the point of this test
    raw = load_profile_raw(name)
    assert "device" in raw


def test_all_required_profiles_present_on_disk():
    on_disk = set(list_profiles())
    missing = set(REQUIRED_PROFILES) - on_disk
    assert not missing, f"missing profile files: {missing}"


def test_profile_cli_override_wins_over_file():
    cfg = load_profile("smoke_cpu", overrides={"training": {"max_steps": 999}})
    assert cfg.training.max_steps == 999


def test_profile_inheritance_from_base():
    cfg = load_profile("nvidia_single_gpu")
    base = load_profile("smoke_cpu")
    # nvidia_single_gpu doesn't override narrative.* -> inherited from base, not from smoke_cpu
    assert cfg.narrative.world_state_dim == AsterWhiteConfig().narrative.world_state_dim
    assert cfg.narrative.world_state_dim != base.narrative.world_state_dim


def test_unknown_profile_raises_clear_error():
    with pytest.raises(AsterWhiteConfigError, match="no such profile"):
        load_profile("does_not_exist")


def test_circular_extends_detected(tmp_path, monkeypatch):
    import asterwhite.profiles as profiles_mod
    monkeypatch.setattr(profiles_mod, "_CONFIGS_DIR", str(tmp_path))
    (tmp_path / "a.yaml").write_text("extends: b\n")
    (tmp_path / "b.yaml").write_text("extends: a\n")
    with pytest.raises(AsterWhiteConfigError, match="circular"):
        profiles_mod.load_profile_raw("a")
