from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class HardwareSpec:
    peak_fp16_tensor_tflops_per_gpu: Optional[float] = None
    peak_fp32_tflops_per_gpu: Optional[float] = None
    memory_bandwidth_gbps_per_gpu: Optional[float] = None
    board_power_w_per_gpu: Optional[float] = None


@dataclass(frozen=True)
class HardwareFeatures:
    gpu_name: str
    gpu_count: int
    supported_frequency_mhz: List[int]
    min_frequency_mhz: Optional[int]
    max_frequency_mhz: Optional[int]
    power_limit_w_per_gpu: Optional[float]
    peak_fp16_tensor_tflops_per_gpu: Optional[float]
    peak_fp32_tflops_per_gpu: Optional[float]
    memory_bandwidth_gbps_per_gpu: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def total_peak_fp16_tensor_tflops(self) -> Optional[float]:
        if self.peak_fp16_tensor_tflops_per_gpu is None:
            return None
        return self.peak_fp16_tensor_tflops_per_gpu * self.gpu_count

    @property
    def total_memory_bandwidth_gbps(self) -> Optional[float]:
        if self.memory_bandwidth_gbps_per_gpu is None:
            return None
        return self.memory_bandwidth_gbps_per_gpu * self.gpu_count


KNOWN_GPU_SPECS: Dict[str, HardwareSpec] = {
    "tesla v100-sxm3-32gb": HardwareSpec(
        peak_fp16_tensor_tflops_per_gpu=125.0,
        peak_fp32_tflops_per_gpu=15.7,
        memory_bandwidth_gbps_per_gpu=900.0,
        board_power_w_per_gpu=300.0,
    ),
    "tesla v100-sxm2-32gb": HardwareSpec(
        peak_fp16_tensor_tflops_per_gpu=125.0,
        peak_fp32_tflops_per_gpu=15.7,
        memory_bandwidth_gbps_per_gpu=900.0,
        board_power_w_per_gpu=300.0,
    ),
    "tesla v100-sxm2-16gb": HardwareSpec(
        peak_fp16_tensor_tflops_per_gpu=125.0,
        peak_fp32_tflops_per_gpu=15.7,
        memory_bandwidth_gbps_per_gpu=900.0,
        board_power_w_per_gpu=300.0,
    ),
    "tesla v100-pcie-32gb": HardwareSpec(
        peak_fp16_tensor_tflops_per_gpu=112.0,
        peak_fp32_tflops_per_gpu=14.0,
        memory_bandwidth_gbps_per_gpu=900.0,
        board_power_w_per_gpu=250.0,
    ),
    "tesla v100-pcie-16gb": HardwareSpec(
        peak_fp16_tensor_tflops_per_gpu=112.0,
        peak_fp32_tflops_per_gpu=14.0,
        memory_bandwidth_gbps_per_gpu=900.0,
        board_power_w_per_gpu=250.0,
    ),
}


def normalize_gpu_name(name: Optional[str]) -> str:
    return (name or "unknown").strip().lower()


def _average_float(values: List[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def resolve_hardware_spec(
    gpu_name: Optional[str],
    peak_fp16_tflops_per_gpu: Optional[float] = None,
    memory_bandwidth_gbps_per_gpu: Optional[float] = None,
    power_limit_w_per_gpu: Optional[float] = None,
) -> HardwareSpec:
    spec = KNOWN_GPU_SPECS.get(normalize_gpu_name(gpu_name), HardwareSpec())
    return HardwareSpec(
        peak_fp16_tensor_tflops_per_gpu=(
            peak_fp16_tflops_per_gpu
            if peak_fp16_tflops_per_gpu is not None
            else spec.peak_fp16_tensor_tflops_per_gpu
        ),
        peak_fp32_tflops_per_gpu=spec.peak_fp32_tflops_per_gpu,
        memory_bandwidth_gbps_per_gpu=(
            memory_bandwidth_gbps_per_gpu
            if memory_bandwidth_gbps_per_gpu is not None
            else spec.memory_bandwidth_gbps_per_gpu
        ),
        board_power_w_per_gpu=(
            power_limit_w_per_gpu
            if power_limit_w_per_gpu is not None
            else spec.board_power_w_per_gpu
        ),
    )


def build_hardware_features(
    run_payload: Dict[str, Any],
    peak_fp16_tflops_per_gpu: Optional[float] = None,
    memory_bandwidth_gbps_per_gpu: Optional[float] = None,
    power_limit_w_per_gpu: Optional[float] = None,
) -> HardwareFeatures:
    nvml = run_payload.get("nvml") or {}
    gpus = nvml.get("gpus") or []
    first_gpu = gpus[0] if gpus else {}
    gpu_name = first_gpu.get("name") or "unknown"
    supported = sorted(
        {
            int(freq)
            for gpu in gpus
            for freq in gpu.get("supported_graphics_clocks_mhz", [])
            if freq is not None
        }
    )
    average_power_limit = _average_float([gpu.get("power_limit_w") for gpu in gpus])
    spec = resolve_hardware_spec(
        gpu_name,
        peak_fp16_tflops_per_gpu=peak_fp16_tflops_per_gpu,
        memory_bandwidth_gbps_per_gpu=memory_bandwidth_gbps_per_gpu,
        power_limit_w_per_gpu=power_limit_w_per_gpu or average_power_limit,
    )
    return HardwareFeatures(
        gpu_name=gpu_name,
        gpu_count=len(gpus) or int(nvml.get("gpu_count") or 0) or 1,
        supported_frequency_mhz=supported,
        min_frequency_mhz=min(supported) if supported else None,
        max_frequency_mhz=max(supported) if supported else None,
        power_limit_w_per_gpu=(
            power_limit_w_per_gpu
            if power_limit_w_per_gpu is not None
            else average_power_limit or spec.board_power_w_per_gpu
        ),
        peak_fp16_tensor_tflops_per_gpu=spec.peak_fp16_tensor_tflops_per_gpu,
        peak_fp32_tflops_per_gpu=spec.peak_fp32_tflops_per_gpu,
        memory_bandwidth_gbps_per_gpu=spec.memory_bandwidth_gbps_per_gpu,
    )
