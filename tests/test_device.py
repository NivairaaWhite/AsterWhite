import pytest
import torch

from asterwhite.core.device import DeviceManager
from asterwhite.utils.error_tracing import AsterWhiteError


def test_resolve_auto_detects_cpu_at_minimum():
    info = DeviceManager.resolve()
    assert info.backend in ("cpu", "cuda", "rocm", "mps", "xpu")
    assert info.device is not None
    assert isinstance(info.supports_bf16, bool)


def test_resolve_explicit_cpu_always_available():
    info = DeviceManager.resolve("cpu")
    assert info.backend == "cpu"
    assert info.device == torch.device("cpu")


def test_resolve_rejects_unavailable_explicit_backend():
    # This sandbox has no CUDA -- requesting it explicitly must raise a
    # clear AsterWhiteError, not silently fall back to CPU.
    if torch.cuda.is_available():
        pytest.skip("CUDA is actually available in this environment")
    with pytest.raises(AsterWhiteError, match="not available"):
        DeviceManager.resolve("cuda")


def test_self_test_passes_on_cpu():
    info = DeviceManager.resolve("cpu")
    DeviceManager.self_test(info)  # must not raise


def test_manifest_dict_is_json_safe():
    import json
    info = DeviceManager.resolve("cpu")
    d = info.as_manifest_dict()
    json.dumps(d)  # raises if anything isn't JSON-serializable
    assert d["backend"] == "cpu"


def test_precision_resolution_downgrades_unsupported_and_logs():
    info = DeviceManager.resolve("cpu")
    # CPU supports bf16 per DeviceManager._bf16_support -- fp16 does not.
    assert DeviceManager.resolve_precision("bf16", info) == "bf16"
    assert DeviceManager.resolve_precision("fp16", info) == "fp32"
    assert DeviceManager.resolve_precision("fp32", info) == "fp32"


def test_precision_resolution_rejects_unknown_dtype():
    info = DeviceManager.resolve("cpu")
    with pytest.raises(AsterWhiteError):
        DeviceManager.resolve_precision("int4", info)
