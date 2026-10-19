"""
AsterWhite :: core.device

Unified device manager. One entry point (`DeviceManager.resolve`) for every
subcommand (pretrain/align/generate/serve) instead of each one hand-rolling
`torch.cuda.is_available()`.

Backend priority (highest first): NVIDIA CUDA / AMD ROCm (same `torch.cuda`
API -- distinguished via `torch.version.hip`) -> Apple MPS -> Intel XPU ->
CPU. An explicit `--device` override always wins over auto-detection.

Honesty note: this sandbox is CPU-only, so the CUDA/ROCm/MPS/XPU branches
below are exercised by unit tests only through mocking/attribute-guards --
they cannot be hardware-validated here. The CPU path is real and is what
every test in this repo actually runs against.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ..utils.error_tracing import AsterWhiteError, get_logger

logger = get_logger("asterwhite.core.device")

_PRECISIONS = ("bf16", "fp16", "fp32")


@dataclass
class DeviceInfo:
    """Everything downstream code (Trainer, serving, run manifests) needs to
    record about the hardware a run actually used."""
    backend: str                       # "cuda" | "rocm" | "mps" | "xpu" | "cpu"
    device: torch.device
    device_name: str
    driver_version: str
    runtime_version: str
    total_memory_bytes: Optional[int]
    supports_bf16: bool
    supports_fp16: bool
    distributed_backend: str           # "nccl" | "gloo" | "mpi" | "none"
    features: Dict[str, bool] = field(default_factory=dict)
    precision_downgraded_from: Optional[str] = None  # set + logged if we had to change it

    def as_manifest_dict(self) -> Dict:
        """Plain-JSON-safe view for run manifests / checkpoints / model cards."""
        return {
            "backend": self.backend,
            "device": str(self.device),
            "device_name": self.device_name,
            "driver_version": self.driver_version,
            "runtime_version": self.runtime_version,
            "total_memory_bytes": self.total_memory_bytes,
            "supports_bf16": self.supports_bf16,
            "supports_fp16": self.supports_fp16,
            "distributed_backend": self.distributed_backend,
            "features": self.features,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "os": platform.platform(),
        }


class DeviceManager:
    """Detects, validates, and self-tests the device a run will use.
    Use `DeviceManager.resolve(...)` -- don't construct `DeviceInfo` by hand.
    """

    @staticmethod
    def _cuda_is_rocm() -> bool:
        return bool(getattr(torch.version, "hip", None))

    @staticmethod
    def _detect_order() -> List[str]:
        order = []
        if torch.cuda.is_available():
            order.append("rocm" if DeviceManager._cuda_is_rocm() else "cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            order.append("mps")
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and callable(getattr(xpu, "is_available", None)) and xpu.is_available():
            order.append("xpu")
        order.append("cpu")  # always last -- always available
        return order

    @staticmethod
    def _bf16_support(backend: str) -> bool:
        if backend in ("cuda", "rocm"):
            try:
                return torch.cuda.is_bf16_supported()
            except Exception:  # pragma: no cover -- defensive, older torch builds
                return False
        if backend == "cpu":
            return True  # torch.cpu.amp bf16 autocast is supported since torch 1.10
        if backend == "mps":
            return False  # MPS bf16 kernel coverage is inconsistent; prefer fp16/fp32
        if backend == "xpu":
            return True
        return False

    @staticmethod
    def _fp16_support(backend: str) -> bool:
        # fp16 *autocast* on plain CPU has poor kernel coverage -- allowed,
        # but callers should expect a downgrade (see resolve_precision).
        return backend != "cpu"

    @classmethod
    def resolve(cls, requested_device: Optional[str] = None) -> DeviceInfo:
        """requested_device: None (auto-detect) or one of
        "cuda"/"rocm"/"mps"/"xpu"/"cpu" (explicit override -- raises
        AsterWhiteError if that backend genuinely isn't available, rather
        than silently falling back)."""
        order = cls._detect_order()
        if requested_device is not None:
            requested_device = requested_device.lower()
            if requested_device not in order and requested_device != "cpu":
                raise AsterWhiteError(
                    f"--device {requested_device} was requested but is not available on this "
                    f"machine. Detected backends in priority order: {order}. Omit --device to "
                    f"auto-select, or pass one of {order}."
                )
            backend = requested_device
        else:
            backend = order[0]

        if backend in ("cuda", "rocm"):
            device = torch.device("cuda")  # ROCm reuses the CUDA device namespace
            idx = device.index or 0
            props = torch.cuda.get_device_properties(idx)
            device_name = props.name
            total_memory = props.total_memory
            driver_version = getattr(torch.version, "hip", None) if backend == "rocm" else str(torch.version.cuda)
            runtime_version = torch.__version__
            distributed_backend = "nccl" if backend == "cuda" else "gloo"
        elif backend == "mps":
            device = torch.device("mps")
            device_name = "Apple GPU (MPS)"
            total_memory = None  # MPS does not expose a stable total-memory query
            driver_version = platform.mac_ver()[0] or "unknown"
            runtime_version = torch.__version__
            distributed_backend = "gloo"  # NCCL/MPS distributed collectives aren't supported; single-device only
        elif backend == "xpu":
            device = torch.device("xpu")
            device_name = "Intel XPU"
            total_memory = None
            driver_version = "unknown"
            runtime_version = torch.__version__
            distributed_backend = "gloo"
        else:
            device = torch.device("cpu")
            device_name = platform.processor() or "CPU"
            total_memory = None
            driver_version = platform.platform()
            runtime_version = torch.__version__
            distributed_backend = "gloo"

        info = DeviceInfo(
            backend=backend, device=device, device_name=device_name,
            driver_version=str(driver_version), runtime_version=runtime_version,
            total_memory_bytes=total_memory,
            supports_bf16=cls._bf16_support(backend), supports_fp16=cls._fp16_support(backend),
            distributed_backend=distributed_backend,
            features={
                "flash_attention_dispatch": backend in ("cuda", "rocm"),
                "distributed_collectives": backend in ("cuda", "rocm", "cpu"),
            },
        )
        cls.self_test(info)
        logger.info(f"device resolved -> backend={info.backend} device={info.device} "
                    f"name={info.device_name!r} bf16={info.supports_bf16} fp16={info.supports_fp16}")
        return info

    @staticmethod
    def self_test(info: DeviceInfo) -> None:
        """Startup capability test: a real small op on the real device,
        before any training/serving begins. Raises AsterWhiteError with a
        clear message (not a bare stack trace) if the device claims
        availability but actually fails on first use -- a real, observed
        failure mode for partially-configured ROCm/MPS/XPU installs."""
        try:
            x = torch.randn(8, 8, device=info.device)
            (x @ x.T).sum().item()
        except Exception as exc:  # pragma: no cover -- only fires on broken installs
            raise AsterWhiteError(
                f"device {info.device} (backend={info.backend}) reported as available but failed "
                f"a basic matmul self-test: {exc}. Falling back is not automatic here because a "
                f"silent fallback would hide a real driver/runtime problem -- fix the {info.backend} "
                f"install or pass --device cpu explicitly.",
                cause=exc,
            ) from exc

    @staticmethod
    def resolve_precision(requested: str, info: DeviceInfo) -> str:
        """Validates the requested AMP dtype against the resolved device and
        downgrades with an explicit, logged reason -- never silently."""
        if requested not in _PRECISIONS:
            raise AsterWhiteError(f"amp_dtype must be one of {_PRECISIONS}, got {requested!r}")
        chosen = requested
        reason = None
        if requested == "bf16" and not info.supports_bf16:
            chosen = "fp16" if info.supports_fp16 else "fp32"
            reason = f"bf16 unsupported on backend={info.backend}"
        elif requested == "fp16" and not info.supports_fp16:
            chosen = "fp32"
            reason = f"fp16 unsupported on backend={info.backend}"
        if chosen != requested:
            logger.warning(f"precision downgraded: requested={requested} -> using={chosen} ({reason})")
        return chosen
