from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable

from analysis.freq_model.features import DerivedModelFeatures


@dataclass(frozen=True)
class NetworkQualityObservation:
    transport_label: str
    effective_bandwidth_gbps: float
    small_message_jitter_cv: float
    large_message_jitter_cv: float
    source_path: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _sorted_results(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    results = list(payload.get('results') or [])
    if not results:
        raise ValueError('network benchmark payload must contain non-empty results')
    return sorted(results, key=lambda entry: float(entry.get('size_mb') or 0.0))


def summarize_network_benchmark(payload: Dict[str, Any], *, source_path: str | None = None) -> NetworkQualityObservation:
    results = _sorted_results(payload)
    large_messages = [entry for entry in results if float(entry.get('size_mb') or 0.0) >= 64.0]
    small_messages = [entry for entry in results if float(entry.get('size_mb') or 0.0) <= 16.0]
    representative_bandwidth_pool = large_messages or results
    representative_jitter_pool = small_messages or results

    effective_bandwidth_gbps = _mean(float(entry.get('busbw_gbps') or 0.0) for entry in representative_bandwidth_pool)
    small_message_jitter_cv = max(float(entry.get('cv') or 0.0) for entry in representative_jitter_pool)
    large_message_jitter_cv = max(float(entry.get('cv') or 0.0) for entry in representative_bandwidth_pool)

    transport_label = f"{payload.get('nccl_socket_ifname') or 'unknown'}|ib_disable={payload.get('nccl_ib_disable') or 'unknown'}"
    return NetworkQualityObservation(
        transport_label=transport_label,
        effective_bandwidth_gbps=effective_bandwidth_gbps,
        small_message_jitter_cv=small_message_jitter_cv,
        large_message_jitter_cv=large_message_jitter_cv,
        source_path=source_path,
    )


def extract_network_benchmark_curve(payload: Dict[str, Any]) -> Dict[str, Any]:
    results = _sorted_results(payload)
    return {
        'world_size': int(payload.get('world_size') or 0),
        'message_sizes_mb': tuple(float(entry.get('size_mb') or 0.0) for entry in results),
        'avg_times_ms': tuple(float(entry.get('avg_ms') or 0.0) for entry in results),
        'bus_bandwidth_gbps': tuple(float(entry.get('busbw_gbps') or 0.0) for entry in results),
    }


def load_network_quality_observation(path: str | Path) -> NetworkQualityObservation:
    payload_path = Path(path).expanduser().resolve()
    payload = json.loads(payload_path.read_text(encoding='utf-8'))
    return summarize_network_benchmark(payload, source_path=str(payload_path))


def apply_network_quality(
    features: DerivedModelFeatures,
    observation: NetworkQualityObservation,
) -> DerivedModelFeatures:
    return replace(
        features,
        network_transport_label=observation.transport_label,
        network_effective_bandwidth_gbps=observation.effective_bandwidth_gbps,
        network_jitter_cv=observation.small_message_jitter_cv,
        network_large_message_jitter_cv=observation.large_message_jitter_cv,
    )
