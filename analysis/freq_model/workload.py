from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ZEUS_LOG_PATTERN = re.compile(
    r"\[Zeus\] Steps (?P<step_start>\d+)-(?P<step_end>\d+): Energy=(?P<energy_wh>[0-9.]+) Wh \((?P<energy_j>[0-9.]+) J\), "
    r"Avg Power=(?P<avg_power_w>[0-9.]+) W, Time=(?P<time_s>[0-9.]+) s, Samples/Wh=(?P<samples_per_wh>[0-9.]+), Tokens/J=(?P<tokens_per_j>[0-9.]+)"
)


@dataclass(frozen=True)
class WorkloadFeatures:
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    seq_length: int
    micro_batch_size: int
    global_batch_size: int
    train_iters: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    data_parallel_size: int
    zero_stage: int
    precision_mode: str
    swiglu: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObservedMetrics:
    iteration: int
    frequency_mhz: Optional[int]
    num_steps: int
    time_s: float
    step_time_s: float
    avg_power_w: float
    energy_j: float
    energy_wh: float
    interval_tokens: Optional[float]
    interval_samples: Optional[float]
    throughput_tokens_per_s: Optional[float]
    throughput_samples_per_s: Optional[float]
    tokens_per_j: Optional[float]
    tokens_per_wh: Optional[float]
    samples_per_wh: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedRunSample:
    run_id: str
    run_dir: Path
    run_payload: Dict[str, Any]
    workload: WorkloadFeatures
    observed: ObservedMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "workload": self.workload.to_dict(),
            "observed": self.observed.to_dict(),
        }


@dataclass(frozen=True)
class ExperimentCollection:
    root_dir: Path
    samples: List[LoadedRunSample]
    skipped_runs: List[Dict[str, str]]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _parse_precision_mode(training_config: Dict[str, Any]) -> str:
    if training_config.get("bf16"):
        return "bf16"
    if training_config.get("fp16"):
        return "fp16"
    return "fp32"


def build_workload_features(run_payload: Dict[str, Any]) -> WorkloadFeatures:
    config = run_payload.get("config") or {}
    model = config.get("model") or {}
    training = config.get("training") or {}
    parallelism = config.get("parallelism") or {}
    topology = ((run_payload.get("topology") or {}).get("resolved")) or {}

    tp = int(parallelism.get("tensor_model_parallel_size") or topology.get("tp") or 1)
    pp = int(parallelism.get("pipeline_model_parallel_size") or topology.get("pp") or 1)
    world_size = int(topology.get("world_size") or run_payload.get("environment", {}).get("WORLD_SIZE") or tp * pp)
    dp = max(1, world_size // max(tp * pp, 1))

    return WorkloadFeatures(
        num_layers=int(model.get("num_layers") or 0),
        hidden_size=int(model.get("hidden_size") or 0),
        ffn_hidden_size=int(model.get("ffn_hidden_size") or 0),
        num_attention_heads=int(model.get("num_attention_heads") or 0),
        num_key_value_heads=int(model.get("num_key_value_heads") or 0),
        seq_length=int(model.get("seq_length") or 0),
        micro_batch_size=int(training.get("micro_batch_size") or 0),
        global_batch_size=int(training.get("global_batch_size") or 0),
        train_iters=int(training.get("train_iters") or 0),
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        data_parallel_size=dp,
        zero_stage=int(parallelism.get("zero_stage") or 0),
        precision_mode=_parse_precision_mode(training),
        swiglu=bool(model.get("swiglu")),
    )


def _build_observed_from_zeus_payload(
    iteration: int,
    frequency_mhz: Optional[int],
    zeus_payload: Dict[str, Any],
) -> Optional[ObservedMetrics]:
    if not zeus_payload:
        return None
    time_s = float(zeus_payload.get("time_s") or 0.0)
    num_steps = int(zeus_payload.get("num_steps") or max(1, (zeus_payload.get("step_end") or 0) - (zeus_payload.get("step_start") or 0) + 1))
    energy_j = float(zeus_payload.get("energy_j") or 0.0)
    energy_wh = float(zeus_payload.get("energy_wh") or 0.0)
    interval_tokens = zeus_payload.get("interval_tokens")
    interval_samples = zeus_payload.get("interval_samples")
    tokens_per_j = zeus_payload.get("interval_tokens_per_j")
    tokens_per_wh = zeus_payload.get("interval_tokens_per_wh")
    samples_per_wh = zeus_payload.get("interval_samples_per_wh")

    if interval_tokens is None and tokens_per_j is not None and energy_j:
        interval_tokens = float(tokens_per_j) * energy_j
    if interval_samples is None and samples_per_wh is not None and energy_wh:
        interval_samples = float(samples_per_wh) * energy_wh

    throughput_tokens = None
    throughput_samples = None
    if interval_tokens is not None and time_s > 0:
        throughput_tokens = float(interval_tokens) / time_s
    if interval_samples is not None and time_s > 0:
        throughput_samples = float(interval_samples) / time_s

    if tokens_per_j is None and interval_tokens is not None and energy_j > 0:
        tokens_per_j = float(interval_tokens) / energy_j
    if tokens_per_wh is None and interval_tokens is not None and energy_wh > 0:
        tokens_per_wh = float(interval_tokens) / energy_wh
    if samples_per_wh is None and interval_samples is not None and energy_wh > 0:
        samples_per_wh = float(interval_samples) / energy_wh

    if time_s <= 0 or energy_j <= 0:
        return None

    return ObservedMetrics(
        iteration=iteration,
        frequency_mhz=frequency_mhz,
        num_steps=num_steps,
        time_s=time_s,
        step_time_s=time_s / max(num_steps, 1),
        avg_power_w=float(zeus_payload.get("avg_power_w") or 0.0),
        energy_j=energy_j,
        energy_wh=energy_wh,
        interval_tokens=float(interval_tokens) if interval_tokens is not None else None,
        interval_samples=float(interval_samples) if interval_samples is not None else None,
        throughput_tokens_per_s=throughput_tokens,
        throughput_samples_per_s=throughput_samples,
        tokens_per_j=float(tokens_per_j) if tokens_per_j is not None else None,
        tokens_per_wh=float(tokens_per_wh) if tokens_per_wh is not None else None,
        samples_per_wh=float(samples_per_wh) if samples_per_wh is not None else None,
    )


def _extract_latest_interval_observed(
    run_payload: Dict[str, Any],
    events: Iterable[Dict[str, Any]],
    include_baseline: bool,
) -> Optional[ObservedMetrics]:
    freq_policy = run_payload.get("freq_policy") or {}
    mode = freq_policy.get("mode")
    static_freq = freq_policy.get("static_clock_mhz")
    frequency_mhz = int(static_freq) if static_freq not in (None, "") else None
    if mode != "static" and not include_baseline:
        return None

    latest: Optional[ObservedMetrics] = None
    for entry in events:
        if entry.get("event_type") != "interval":
            continue
        payload = entry.get("payload") or {}
        interval_metrics = payload.get("interval_metrics") or {}
        zeus_payload = interval_metrics.get("zeus") or {}
        observed = _build_observed_from_zeus_payload(
            iteration=int(payload.get("iteration") or 0),
            frequency_mhz=frequency_mhz,
            zeus_payload=zeus_payload,
        )
        if observed is None:
            continue
        if latest is None or observed.iteration >= latest.iteration:
            latest = observed
    return latest


def _find_log_files(run_dir: Path) -> List[Path]:
    log_dir = run_dir / "logs"
    if not log_dir.exists():
        return []
    return sorted(path for path in log_dir.iterdir() if path.is_file() and path.suffix == ".log")


def _extract_from_logs(run_payload: Dict[str, Any], run_dir: Path, include_baseline: bool) -> Optional[ObservedMetrics]:
    freq_policy = run_payload.get("freq_policy") or {}
    mode = freq_policy.get("mode")
    static_freq = freq_policy.get("static_clock_mhz")
    frequency_mhz = int(static_freq) if static_freq not in (None, "") else None
    if mode != "static" and not include_baseline:
        return None

    latest_match = None
    for log_file in _find_log_files(run_dir):
        for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
            match = ZEUS_LOG_PATTERN.search(line)
            if match:
                latest_match = match
    if latest_match is None:
        return None
    payload = {key: float(value) for key, value in latest_match.groupdict().items() if key not in {"step_start", "step_end"}}
    step_start = int(latest_match.group("step_start"))
    step_end = int(latest_match.group("step_end"))
    payload.update(
        {
            "step_start": step_start,
            "step_end": step_end,
            "num_steps": max(1, step_end - step_start + 1),
            "interval_tokens_per_j": payload["tokens_per_j"],
            "interval_samples_per_wh": payload["samples_per_wh"],
            "interval_tokens": payload["tokens_per_j"] * payload["energy_j"],
            "interval_samples": payload["samples_per_wh"] * payload["energy_wh"],
            "interval_tokens_per_wh": (
                (payload["tokens_per_j"] * payload["energy_j"]) / payload["energy_wh"]
                if payload["energy_wh"] > 0
                else None
            ),
        }
    )
    return _build_observed_from_zeus_payload(step_end, frequency_mhz, payload)


def load_experiment_samples(experiment_root: str, include_baseline: bool = False) -> ExperimentCollection:
    root_dir = Path(experiment_root).expanduser().resolve()
    samples: List[LoadedRunSample] = []
    skipped_runs: List[Dict[str, str]] = []

    if not root_dir.exists():
        return ExperimentCollection(root_dir=root_dir, samples=samples, skipped_runs=[{"run_dir": str(root_dir), "reason": "experiment root does not exist"}])

    for run_json_path in sorted(root_dir.glob("*/run.json")):
        run_dir = run_json_path.parent
        run_payload = _load_json(run_json_path)
        events = _load_jsonl(run_dir / "events.jsonl")
        observed = _extract_latest_interval_observed(run_payload, events, include_baseline=include_baseline)
        if observed is None:
            observed = _extract_from_logs(run_payload, run_dir, include_baseline=include_baseline)
        if observed is None:
            skipped_runs.append({"run_dir": str(run_dir), "reason": "no usable Zeus interval metrics"})
            continue
        workload = build_workload_features(run_payload)
        samples.append(
            LoadedRunSample(
                run_id=str(run_payload.get("run_id") or run_dir.name),
                run_dir=run_dir,
                run_payload=run_payload,
                workload=workload,
                observed=observed,
            )
        )

    return ExperimentCollection(root_dir=root_dir, samples=samples, skipped_runs=skipped_runs)
