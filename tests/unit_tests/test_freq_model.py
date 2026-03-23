import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.freq_model.calibrate import calibrate_frequency_model
from analysis.freq_model.cross_node import fit_cross_node_penalty_model
from analysis.freq_model.features import DerivedModelFeatures, derive_model_features
from analysis.freq_model.hardware import HardwareFeatures, build_hardware_features
from analysis.freq_model.model import CalibrationParams, predict_power_w, predict_throughput_tokens_per_s
from analysis.freq_model.recommend import build_prediction_bundle
from analysis.freq_model.workload import LoadedRunSample, ObservedMetrics, WorkloadFeatures, load_experiment_samples
from scripts.predict_freq_sweet_spot import _validate_baseline_compatibility, _validate_workload_consistency


SUPPORTED_CLOCKS = [1000, 1200, 1400]
TOKENS_PER_STEP = 16 * 2048


def _write_run(
    root: Path,
    run_id: str,
    *,
    throughput_tokens_per_s: float,
    avg_power_w: float,
    steps: int = 20,
    frequency_mhz: int | None = None,
    mode: str = "static",
) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    interval_tokens = float(steps * TOKENS_PER_STEP)
    time_s = interval_tokens / throughput_tokens_per_s
    energy_j = avg_power_w * time_s
    energy_wh = energy_j / 3600.0
    interval_samples = 10.0 * steps
    freq_policy = {"mode": mode}
    if frequency_mhz is not None:
        freq_policy["static_clock_mhz"] = str(frequency_mhz)

    run_payload = {
        "run_id": run_id,
        "freq_policy": freq_policy,
        "config": {
            "model": {
                "num_layers": 28,
                "hidden_size": 3584,
                "ffn_hidden_size": 18944,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "seq_length": 2048,
                "swiglu": True,
            },
            "training": {
                "micro_batch_size": 1,
                "global_batch_size": 16,
                "train_iters": 20,
                "bf16": True,
                "fp16": False,
            },
            "parallelism": {
                "tensor_model_parallel_size": 4,
                "pipeline_model_parallel_size": 1,
                "zero_stage": 1,
            },
        },
        "topology": {
            "resolved": {
                "world_size": 16,
                "tp": 4,
                "pp": 1,
            }
        },
        "nvml": {
            "gpu_count": 16,
            "gpus": [
                {
                    "name": "Tesla V100-SXM3-32GB",
                    "power_limit_w": 300.0,
                    "supported_graphics_clocks_mhz": SUPPORTED_CLOCKS,
                }
            ]
            * 16,
        },
    }
    events = [
        {
            "event_type": "interval",
            "payload": {
                "iteration": steps,
                "interval_metrics": {
                    "zeus": {
                        "step_start": 1,
                        "step_end": steps,
                        "num_steps": steps,
                        "time_s": time_s,
                        "avg_power_w": avg_power_w,
                        "energy_j": energy_j,
                        "energy_wh": energy_wh,
                        "interval_tokens": interval_tokens,
                        "interval_samples": interval_samples,
                        "interval_tokens_per_j": interval_tokens / energy_j,
                        "interval_tokens_per_wh": interval_tokens / energy_wh,
                        "interval_samples_per_wh": interval_samples / energy_wh,
                    }
                },
            },
        }
    ]
    (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def _load_single_baseline_sample(root: Path):
    collection = load_experiment_samples(str(root), include_baseline=True)
    baseline_samples = [sample for sample in collection.samples if sample.observed.frequency_mhz is None]
    assert len(baseline_samples) == 1
    return baseline_samples[0]


def _write_zeus_log(run_dir: Path, *, steps: int, time_s: float, avg_power_w: float, energy_j: float, energy_wh: float) -> None:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    samples_per_wh = (10.0 * steps) / energy_wh
    tokens_per_j = (steps * TOKENS_PER_STEP) / energy_j
    line = (
        f"[Zeus] Steps 1-{steps}: Energy={energy_wh:.6f} Wh ({energy_j:.6f} J), "
        f"Avg Power={avg_power_w:.6f} W, Time={time_s:.6f} s, "
        f"Samples/Wh={samples_per_wh:.6f}, Tokens/J={tokens_per_j:.6f}"
    )
    (log_dir / "train.log").write_text(line + "\n", encoding="utf-8")


def test_prediction_bundle_reports_baseline_relative_tradeoff(tmp_path: Path):
    hardware = HardwareFeatures(
        gpu_name="synthetic",
        gpu_count=1,
        supported_frequency_mhz=SUPPORTED_CLOCKS,
        min_frequency_mhz=SUPPORTED_CLOCKS[0],
        max_frequency_mhz=SUPPORTED_CLOCKS[-1],
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=100.0,
        peak_fp32_tflops_per_gpu=50.0,
        memory_bandwidth_gbps_per_gpu=900.0,
    )
    features = DerivedModelFeatures(
        tokens_per_step=1000.0,
        samples_per_step=1.0,
        approx_model_params=1.0,
        approx_flops_per_token=1.0,
        approx_flops_per_step=1000.0,
        approx_memory_bytes_per_step=1.0,
        approx_communication_bytes_per_step=0.0,
        arithmetic_intensity_flops_per_byte=1.0,
        communication_share=0.0,
        pipeline_parallel_efficiency=1.0,
        pipeline_exposed_fraction=0.0,
        dp_overlapable_fraction=0.0,
        tp_sync_fraction=0.0,
        hardware_balance_flops_per_byte=1.0,
        compute_weight=1.0,
        memory_weight=0.0,
        communication_weight=0.0,
    )
    params = CalibrationParams(
        compute_limit_at_max_tokens_per_s=1400.0,
        memory_limit_tokens_per_s=1_000_000.0,
        communication_limit_tokens_per_s=1_000_000.0,
        communication_penalty=0.0,
        static_power_w=50.0,
        dynamic_power_w=200.0,
        dynamic_power_exponent=1.5,
    )

    baseline_workload = WorkloadFeatures(
        num_layers=0,
        hidden_size=0,
        ffn_hidden_size=0,
        num_attention_heads=0,
        num_key_value_heads=0,
        seq_length=1000,
        micro_batch_size=1,
        global_batch_size=1,
        train_iters=100,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        data_parallel_size=1,
        zero_stage=0,
        precision_mode="fp16",
        swiglu=False,
    )
    baseline_observed = ObservedMetrics(
        iteration=100,
        frequency_mhz=None,
        num_steps=100,
        time_s=100000.0 / 1150.0,
        step_time_s=(100000.0 / 1150.0) / 100.0,
        avg_power_w=220.0,
        energy_j=220.0 * (100000.0 / 1150.0),
        energy_wh=(220.0 * (100000.0 / 1150.0)) / 3600.0,
        interval_tokens=100000.0,
        interval_samples=100.0,
        throughput_tokens_per_s=1150.0,
        throughput_samples_per_s=1.0,
        tokens_per_j=1150.0 / 220.0,
        tokens_per_wh=(1150.0 / 220.0) * 3600.0,
        samples_per_wh=3600.0 / 220.0,
    )
    baseline_sample = LoadedRunSample(
        run_id="baseline_001",
        run_dir=tmp_path / "baseline_bundle" / "baseline_001",
        run_payload={},
        workload=baseline_workload,
        observed=baseline_observed,
    )

    bundle = build_prediction_bundle(
        hardware,
        features,
        params,
        comparison_steps=100,
        neighborhood=0,
        baseline_sample=baseline_sample,
    )

    assert bundle["objective"]["mode"] == "baseline_relative_energy_then_time_tradeoff"
    assert bundle["pareto_frontier_frequencies_mhz"]
    assert bundle["supported_sweet_spot"]["runtime_ratio_vs_baseline"] > bundle["supported_balanced_sweet_spot"]["runtime_ratio_vs_baseline"]
    assert bundle["supported_sweet_spot"]["energy_ratio_vs_baseline"] < 1.0
    assert bundle["supported_balanced_sweet_spot"]["runtime_ratio_vs_baseline"] < bundle["supported_sweet_spot"]["runtime_ratio_vs_baseline"]
    assert bundle["supported_balanced_sweet_spot"]["energy_ratio_vs_baseline"] > bundle["supported_sweet_spot"]["energy_ratio_vs_baseline"]
    assert bundle["recommended_frequencies_mhz"] == [bundle["supported_sweet_spot"]["frequency_mhz"]]
    assert bundle["baseline_reference"]["run_id"] == "baseline_001"



def test_load_calibrate_and_overlay_modes(tmp_path: Path):
    experiment_root = tmp_path / "experiments"
    baseline_root = tmp_path / "baseline"
    _write_run(experiment_root, "run_1000", frequency_mhz=1000, throughput_tokens_per_s=1000.0, avg_power_w=171.0)
    _write_run(experiment_root, "run_1200", frequency_mhz=1200, throughput_tokens_per_s=1200.0, avg_power_w=209.0)
    _write_run(experiment_root, "run_1400", frequency_mhz=1400, throughput_tokens_per_s=1400.0, avg_power_w=250.0)
    _write_run(
        baseline_root,
        "baseline_001",
        throughput_tokens_per_s=1150.0,
        avg_power_w=220.0,
        frequency_mhz=None,
        mode="baseline",
    )

    collection = load_experiment_samples(str(experiment_root))
    baseline_sample = _load_single_baseline_sample(baseline_root)
    hardware = build_hardware_features(collection.samples[0].run_payload)
    derived = [derive_model_features(hardware, sample.workload) for sample in collection.samples]
    calibration = calibrate_frequency_model(collection.samples, hardware, derived, baseline_sample=baseline_sample)
    cross_node_fit = fit_cross_node_penalty_model(hardware)

    analytic_bundle = build_prediction_bundle(
        hardware,
        derived[0],
        calibration.params,
        metric="tokens_per_j",
        neighborhood=1,
        comparison_steps=20,
        baseline_sample=baseline_sample,
        observed_samples=collection.samples,
        use_observed_overlay=False,
    )
    overlay_bundle = build_prediction_bundle(
        hardware,
        derived[0],
        calibration.params,
        metric="tokens_per_j",
        neighborhood=1,
        comparison_steps=20,
        baseline_sample=baseline_sample,
        observed_samples=collection.samples,
        use_observed_overlay=True,
    )

    assert calibration.throughput_mape >= 0.0
    assert calibration.power_mape >= 0.0
    assert calibration.runtime_ratio_mape >= 0.0
    assert calibration.energy_ratio_mape >= 0.0
    assert calibration.total_time_mape >= 0.0
    assert calibration.params.cross_node_beta_pp_edge_s == pytest.approx(cross_node_fit.beta_pp_edge_s)
    assert calibration.params.cross_node_power_base_drop == pytest.approx(cross_node_fit.power_base_drop)
    assert calibration.total_energy_mape >= 0.0
    assert calibration.objective >= 0.0

    assert analytic_bundle["baseline_reference"]["run_id"] == "baseline_001"
    assert analytic_bundle["overlay"]["enabled"] is False
    assert analytic_bundle["overlay"]["uses_observed_interpolation"] is False
    assert overlay_bundle["overlay"]["enabled"] is True
    assert overlay_bundle["overlay"]["uses_observed_interpolation"] is True
    assert overlay_bundle["supported_sweet_spot"]["frequency_mhz"] in overlay_bundle["recommended_frequencies_mhz"]
    assert overlay_bundle["pareto_frontier_frequencies_mhz"] == [1400, 1200, 1000]
    assert all("runtime_ratio_vs_baseline" in point for point in overlay_bundle["supported_predictions"])
    assert all("energy_ratio_vs_baseline" in point for point in overlay_bundle["supported_predictions"])


def test_cli_writes_baseline_relative_outputs(tmp_path: Path):
    experiment_root = tmp_path / "experiments"
    baseline_root = tmp_path / "baseline"
    output_dir = tmp_path / "predictions"
    _write_run(experiment_root, "run_1000", frequency_mhz=1000, throughput_tokens_per_s=1000.0, avg_power_w=171.0)
    _write_run(experiment_root, "run_1200", frequency_mhz=1200, throughput_tokens_per_s=1200.0, avg_power_w=209.0)
    _write_run(experiment_root, "run_1400", frequency_mhz=1400, throughput_tokens_per_s=1400.0, avg_power_w=250.0)
    _write_run(
        baseline_root,
        "baseline_001",
        throughput_tokens_per_s=1100.0,
        avg_power_w=225.0,
        frequency_mhz=None,
        mode="baseline",
    )
    _write_run(
        baseline_root,
        "baseline_002",
        throughput_tokens_per_s=1150.0,
        avg_power_w=220.0,
        frequency_mhz=None,
        mode="baseline",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/predict_freq_sweet_spot.py",
            "--experiment-root",
            str(experiment_root),
            "--baseline-root",
            str(baseline_root),
            "--output-dir",
            str(output_dir),
            "--use-observed-overlay",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Recommended frequencies" in result.stdout
    assert "Baseline reference: baseline_002" in result.stdout
    prediction_json = output_dir / "prediction.json"
    prediction_report = output_dir / "prediction_report.md"
    assert prediction_json.exists()
    assert prediction_report.exists()

    payload = json.loads(prediction_json.read_text(encoding="utf-8"))
    assert payload["prediction"]["recommended_frequencies_mhz"]
    assert payload["prediction"]["objective"]["mode"] == "baseline_relative_energy_then_time_tradeoff"
    assert "max_slowdown_ratio" not in payload["prediction"]["objective"]
    assert payload["prediction"]["overlay"]["uses_observed_interpolation"] is True
    assert payload["baseline_sample"]["run_id"] == "baseline_002"
    assert payload["prediction"]["baseline_reference"]["run_id"] == "baseline_002"
    assert payload["prediction"]["supported_sweet_spot"]["frequency_mhz"] in payload["prediction"]["recommended_frequencies_mhz"]
    assert "supported_balanced_sweet_spot" in payload["prediction"]
    assert payload["prediction"]["pareto_frontier_frequencies_mhz"] == [1400, 1200, 1000]
    assert "runtime_ratio_mape" in payload["calibration"]
    assert "energy_ratio_mape" in payload["calibration"]
    assert "total_time_mape" in payload["calibration"]
    assert "total_energy_mape" in payload["calibration"]
    assert "efficiency_mape" not in payload["calibration"]
    assert "guardrail_penalty" not in payload["calibration"]
    assert payload["prediction_accuracy"]["source"] == "analytic_supported_predictions_without_overlay"
    assert payload["prediction_accuracy"]["point_count"] == 3
    assert len(payload["prediction_accuracy"]["points"]) == 3
    assert all("runtime_ratio_ape" in point for point in payload["prediction_accuracy"]["points"])
    assert all("energy_ratio_ape" in point for point in payload["prediction_accuracy"]["points"])
    report_text = prediction_report.read_text(encoding="utf-8")
    assert "## Pareto Frontier (Primary View)" in report_text
    assert "## Prediction Accuracy" in report_text
    assert "## Observed vs Predicted Ratios" in report_text
    assert payload["hardware"]["gpu_name"] == "Tesla V100-SXM3-32GB"


def test_pipeline_parallel_proxy_inflates_communication_pressure_for_deeper_pp():
    hardware = HardwareFeatures(
        gpu_name="synthetic",
        gpu_count=16,
        supported_frequency_mhz=SUPPORTED_CLOCKS,
        min_frequency_mhz=SUPPORTED_CLOCKS[0],
        max_frequency_mhz=SUPPORTED_CLOCKS[-1],
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=100.0,
        peak_fp32_tflops_per_gpu=50.0,
        memory_bandwidth_gbps_per_gpu=900.0,
    )

    tp4_dp4 = WorkloadFeatures(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=16,
        train_iters=20,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        data_parallel_size=4,
        zero_stage=1,
        precision_mode="bf16",
        swiglu=True,
    )
    tp2_pp4_dp2 = WorkloadFeatures(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=16,
        train_iters=20,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        data_parallel_size=2,
        zero_stage=1,
        precision_mode="bf16",
        swiglu=True,
    )

    ref_features = derive_model_features(hardware, tp4_dp4)
    transfer_features = derive_model_features(hardware, tp2_pp4_dp2)

    assert transfer_features.approx_communication_bytes_per_step > ref_features.approx_communication_bytes_per_step
    assert transfer_features.communication_share > ref_features.communication_share


def test_pipeline_bubble_fraction_raises_pp_communication_when_microbatches_are_scarce():
    hardware = HardwareFeatures(
        gpu_name="synthetic",
        gpu_count=16,
        supported_frequency_mhz=SUPPORTED_CLOCKS,
        min_frequency_mhz=SUPPORTED_CLOCKS[0],
        max_frequency_mhz=SUPPORTED_CLOCKS[-1],
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=100.0,
        peak_fp32_tflops_per_gpu=50.0,
        memory_bandwidth_gbps_per_gpu=900.0,
    )

    more_microbatches = WorkloadFeatures(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=16,
        train_iters=20,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        data_parallel_size=2,
        zero_stage=1,
        precision_mode="bf16",
        swiglu=True,
    )
    fewer_microbatches = WorkloadFeatures(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        seq_length=2048,
        micro_batch_size=2,
        global_batch_size=16,
        train_iters=20,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        data_parallel_size=2,
        zero_stage=1,
        precision_mode="bf16",
        swiglu=True,
    )

    more_features = derive_model_features(hardware, more_microbatches)
    fewer_features = derive_model_features(hardware, fewer_microbatches)

    assert fewer_features.pipeline_parallel_efficiency < more_features.pipeline_parallel_efficiency
    assert fewer_features.approx_communication_bytes_per_step > more_features.approx_communication_bytes_per_step
    assert fewer_features.communication_share > more_features.communication_share


def test_pipeline_parallel_efficiency_penalizes_deeper_pp():
    hardware = HardwareFeatures(
        gpu_name="synthetic",
        gpu_count=16,
        supported_frequency_mhz=SUPPORTED_CLOCKS,
        min_frequency_mhz=SUPPORTED_CLOCKS[0],
        max_frequency_mhz=SUPPORTED_CLOCKS[-1],
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=100.0,
        peak_fp32_tflops_per_gpu=50.0,
        memory_bandwidth_gbps_per_gpu=900.0,
    )

    pp1 = WorkloadFeatures(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=16,
        train_iters=20,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        data_parallel_size=4,
        zero_stage=1,
        precision_mode="bf16",
        swiglu=True,
    )
    pp4 = WorkloadFeatures(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=16,
        train_iters=20,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        data_parallel_size=2,
        zero_stage=1,
        precision_mode="bf16",
        swiglu=True,
    )

    pp1_features = derive_model_features(hardware, pp1)
    pp4_features = derive_model_features(hardware, pp4)

    assert pp1_features.pipeline_parallel_efficiency == 1.0
    assert pp4_features.pipeline_parallel_efficiency < 1.0
    assert pp1_features.pipeline_exposed_fraction == 0.0
    assert pp4_features.pipeline_exposed_fraction > pp1_features.pipeline_exposed_fraction


def test_validate_workload_consistency_rejects_mixed_micro_batch_sizes(tmp_path: Path):
    experiment_root = tmp_path / "microbatch_mismatch"
    _write_run(experiment_root, "run_1000", frequency_mhz=1000, throughput_tokens_per_s=1000.0, avg_power_w=171.0)
    _write_run(experiment_root, "run_1200", frequency_mhz=1200, throughput_tokens_per_s=1200.0, avg_power_w=209.0)

    run_1200_path = experiment_root / "run_1200" / "run.json"
    payload = json.loads(run_1200_path.read_text(encoding="utf-8"))
    payload["config"]["training"]["micro_batch_size"] = 2
    run_1200_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    collection = load_experiment_samples(str(experiment_root), include_baseline=False)

    with pytest.raises(SystemExit, match="equivalent workloads"):
        _validate_workload_consistency(collection.samples)


def test_validate_workload_consistency_rejects_mixed_attention_shape_metadata(tmp_path: Path):
    experiment_root = tmp_path / "attention_shape_mismatch"
    _write_run(experiment_root, "run_1000", frequency_mhz=1000, throughput_tokens_per_s=1000.0, avg_power_w=171.0)
    _write_run(experiment_root, "run_1200", frequency_mhz=1200, throughput_tokens_per_s=1200.0, avg_power_w=209.0)

    run_1200_path = experiment_root / "run_1200" / "run.json"
    payload = json.loads(run_1200_path.read_text(encoding="utf-8"))
    payload["config"]["model"]["num_attention_heads"] = 32
    run_1200_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    collection = load_experiment_samples(str(experiment_root), include_baseline=False)

    with pytest.raises(SystemExit, match="equivalent workloads"):
        _validate_workload_consistency(collection.samples)


def test_validate_baseline_compatibility_rejects_mismatched_baseline_workload(tmp_path: Path):
    experiment_root = tmp_path / "static_bundle"
    baseline_root = tmp_path / "baseline_bundle"
    _write_run(experiment_root, "run_1000", frequency_mhz=1000, throughput_tokens_per_s=1000.0, avg_power_w=171.0)
    _write_run(baseline_root, "baseline_001", throughput_tokens_per_s=1100.0, avg_power_w=220.0, mode="baseline")

    baseline_run_path = baseline_root / "baseline_001" / "run.json"
    payload = json.loads(baseline_run_path.read_text(encoding="utf-8"))
    payload["config"]["training"]["global_batch_size"] = 32
    baseline_run_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    collection = load_experiment_samples(str(experiment_root), include_baseline=False)
    baseline_collection = load_experiment_samples(str(baseline_root), include_baseline=True)
    baseline_sample = [sample for sample in baseline_collection.samples if sample.observed.frequency_mhz is None][0]

    with pytest.raises(SystemExit, match="baseline workload must match"):
        _validate_baseline_compatibility(collection.samples, baseline_sample)


def test_pipeline_exposed_fraction_increases_when_microbatches_are_scarce():
    hardware = HardwareFeatures(
        gpu_name="synthetic",
        gpu_count=16,
        supported_frequency_mhz=SUPPORTED_CLOCKS,
        min_frequency_mhz=SUPPORTED_CLOCKS[0],
        max_frequency_mhz=SUPPORTED_CLOCKS[-1],
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=100.0,
        peak_fp32_tflops_per_gpu=50.0,
        memory_bandwidth_gbps_per_gpu=900.0,
    )

    more_microbatches = WorkloadFeatures(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=16,
        train_iters=20,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        data_parallel_size=2,
        zero_stage=1,
        precision_mode="bf16",
        swiglu=True,
    )
    fewer_microbatches = WorkloadFeatures(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        seq_length=2048,
        micro_batch_size=2,
        global_batch_size=16,
        train_iters=20,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        data_parallel_size=2,
        zero_stage=1,
        precision_mode="bf16",
        swiglu=True,
    )

    more_features = derive_model_features(hardware, more_microbatches)
    fewer_features = derive_model_features(hardware, fewer_microbatches)

    assert fewer_features.pipeline_exposed_fraction > more_features.pipeline_exposed_fraction


def test_pipeline_parallel_efficiency_matches_microbatch_bubble_formula():
    hardware = HardwareFeatures(
        gpu_name="synthetic",
        gpu_count=16,
        supported_frequency_mhz=SUPPORTED_CLOCKS,
        min_frequency_mhz=SUPPORTED_CLOCKS[0],
        max_frequency_mhz=SUPPORTED_CLOCKS[-1],
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=100.0,
        peak_fp32_tflops_per_gpu=50.0,
        memory_bandwidth_gbps_per_gpu=900.0,
    )

    workload = WorkloadFeatures(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=16,
        train_iters=20,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        data_parallel_size=2,
        zero_stage=1,
        precision_mode="bf16",
        swiglu=True,
    )

    features = derive_model_features(hardware, workload)

    expected_microbatches = 16 / (1 * 2)
    expected_efficiency = expected_microbatches / (expected_microbatches + (4 - 1))
    assert features.pipeline_parallel_efficiency == pytest.approx(expected_efficiency)


def test_low_freq_correction_is_weaker_without_pipeline_exposure():
    hardware = HardwareFeatures(
        gpu_name="synthetic",
        gpu_count=16,
        supported_frequency_mhz=SUPPORTED_CLOCKS,
        min_frequency_mhz=SUPPORTED_CLOCKS[0],
        max_frequency_mhz=SUPPORTED_CLOCKS[-1],
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=100.0,
        peak_fp32_tflops_per_gpu=50.0,
        memory_bandwidth_gbps_per_gpu=900.0,
    )
    no_bubble = DerivedModelFeatures(
        tokens_per_step=1000.0,
        samples_per_step=1.0,
        approx_model_params=1.0,
        approx_flops_per_token=1.0,
        approx_flops_per_step=1000.0,
        approx_memory_bytes_per_step=1.0,
        approx_communication_bytes_per_step=0.0,
        arithmetic_intensity_flops_per_byte=1.0,
        communication_share=0.20,
        pipeline_parallel_efficiency=1.0,
        pipeline_exposed_fraction=0.0,
        dp_overlapable_fraction=0.18,
        tp_sync_fraction=0.45,
        hardware_balance_flops_per_byte=1.0,
        compute_weight=1.0,
        memory_weight=0.0,
        communication_weight=0.0,
    )
    bubbled = DerivedModelFeatures(
        tokens_per_step=1000.0,
        samples_per_step=1.0,
        approx_model_params=1.0,
        approx_flops_per_token=1.0,
        approx_flops_per_step=1000.0,
        approx_memory_bytes_per_step=1.0,
        approx_communication_bytes_per_step=0.0,
        arithmetic_intensity_flops_per_byte=1.0,
        communication_share=0.20,
        pipeline_parallel_efficiency=1.0,
        pipeline_exposed_fraction=0.45,
        dp_overlapable_fraction=0.18,
        tp_sync_fraction=0.45,
        hardware_balance_flops_per_byte=1.0,
        compute_weight=1.0,
        memory_weight=0.0,
        communication_weight=0.0,
    )
    params = CalibrationParams(
        compute_limit_at_max_tokens_per_s=1400.0,
        memory_limit_tokens_per_s=1_000_000.0,
        communication_limit_tokens_per_s=1_000_000.0,
        communication_penalty=0.0,
        static_power_w=50.0,
        dynamic_power_w=200.0,
        dynamic_power_exponent=1.5,
        throughput_saturation_ratio=0.9,
        correction_split_ratio=0.72,
        correction_transition_width=0.05,
        throughput_low_freq_correction=-0.10,
        correction_topology_weight=1.0,
        correction_communication_weight=0.5,
    )

    low_freq = hardware.min_frequency_mhz

    assert predict_throughput_tokens_per_s(low_freq, hardware, no_bubble, params) > predict_throughput_tokens_per_s(low_freq, hardware, bubbled, params)


def test_low_tp_topologies_have_flatter_frequency_response_and_retain_more_power():
    hardware = HardwareFeatures(
        gpu_name="synthetic",
        gpu_count=16,
        supported_frequency_mhz=SUPPORTED_CLOCKS,
        min_frequency_mhz=SUPPORTED_CLOCKS[0],
        max_frequency_mhz=SUPPORTED_CLOCKS[-1],
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=100.0,
        peak_fp32_tflops_per_gpu=50.0,
        memory_bandwidth_gbps_per_gpu=900.0,
    )
    tp_sync_heavy = DerivedModelFeatures(
        tokens_per_step=1000.0,
        samples_per_step=1.0,
        approx_model_params=1.0,
        approx_flops_per_token=1.0,
        approx_flops_per_step=1000.0,
        approx_memory_bytes_per_step=1.0,
        approx_communication_bytes_per_step=0.0,
        arithmetic_intensity_flops_per_byte=1.0,
        communication_share=0.20,
        pipeline_parallel_efficiency=1.0,
        pipeline_exposed_fraction=0.30,
        dp_overlapable_fraction=0.15,
        tp_sync_fraction=0.45,
        hardware_balance_flops_per_byte=1.0,
        compute_weight=1.0,
        memory_weight=0.0,
        communication_weight=0.0,
    )
    low_tp = DerivedModelFeatures(
        tokens_per_step=1000.0,
        samples_per_step=1.0,
        approx_model_params=1.0,
        approx_flops_per_token=1.0,
        approx_flops_per_step=1000.0,
        approx_memory_bytes_per_step=1.0,
        approx_communication_bytes_per_step=0.0,
        arithmetic_intensity_flops_per_byte=1.0,
        communication_share=0.20,
        pipeline_parallel_efficiency=1.0,
        pipeline_exposed_fraction=0.30,
        dp_overlapable_fraction=0.15,
        tp_sync_fraction=0.0,
        hardware_balance_flops_per_byte=1.0,
        compute_weight=1.0,
        memory_weight=0.0,
        communication_weight=0.0,
    )
    params = CalibrationParams(
        compute_limit_at_max_tokens_per_s=1400.0,
        memory_limit_tokens_per_s=1_000_000.0,
        communication_limit_tokens_per_s=1_000_000.0,
        communication_penalty=0.0,
        static_power_w=50.0,
        dynamic_power_w=200.0,
        dynamic_power_exponent=1.5,
        throughput_saturation_ratio=0.9,
    )

    low_freq = hardware.min_frequency_mhz
    high_freq = hardware.max_frequency_mhz

    assert predict_throughput_tokens_per_s(low_freq, hardware, low_tp, params) > predict_throughput_tokens_per_s(low_freq, hardware, tp_sync_heavy, params)
    assert predict_throughput_tokens_per_s(high_freq, hardware, low_tp, params) == pytest.approx(
        predict_throughput_tokens_per_s(high_freq, hardware, tp_sync_heavy, params),
        rel=1e-6,
    )
    assert predict_power_w(low_freq, hardware, low_tp, params) > predict_power_w(low_freq, hardware, tp_sync_heavy, params)
    assert predict_power_w(high_freq, hardware, low_tp, params) == pytest.approx(
        predict_power_w(high_freq, hardware, tp_sync_heavy, params),
        rel=1e-6,
    )



def test_low_frequency_extrapolation_regularization_only_affects_unobserved_tail():
    hardware = HardwareFeatures(
        gpu_name="synthetic",
        gpu_count=16,
        supported_frequency_mhz=SUPPORTED_CLOCKS,
        min_frequency_mhz=SUPPORTED_CLOCKS[0],
        max_frequency_mhz=SUPPORTED_CLOCKS[-1],
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=100.0,
        peak_fp32_tflops_per_gpu=50.0,
        memory_bandwidth_gbps_per_gpu=900.0,
    )
    features = DerivedModelFeatures(
        tokens_per_step=1000.0,
        samples_per_step=1.0,
        approx_model_params=1.0,
        approx_flops_per_token=1.0,
        approx_flops_per_step=1000.0,
        approx_memory_bytes_per_step=1.0,
        approx_communication_bytes_per_step=0.0,
        arithmetic_intensity_flops_per_byte=1.0,
        communication_share=0.20,
        pipeline_parallel_efficiency=1.0,
        pipeline_exposed_fraction=0.25,
        dp_overlapable_fraction=0.15,
        tp_sync_fraction=0.0,
        hardware_balance_flops_per_byte=1.0,
        compute_weight=1.0,
        memory_weight=0.0,
        communication_weight=0.0,
    )
    baseline = CalibrationParams(
        compute_limit_at_max_tokens_per_s=1400.0,
        memory_limit_tokens_per_s=1_000_000.0,
        communication_limit_tokens_per_s=1_000_000.0,
        communication_penalty=0.0,
        static_power_w=50.0,
        dynamic_power_w=200.0,
        dynamic_power_exponent=1.5,
        throughput_saturation_ratio=0.9,
    )
    regularized = CalibrationParams(
        compute_limit_at_max_tokens_per_s=1400.0,
        memory_limit_tokens_per_s=1_000_000.0,
        communication_limit_tokens_per_s=1_000_000.0,
        communication_penalty=0.0,
        static_power_w=50.0,
        dynamic_power_w=200.0,
        dynamic_power_exponent=1.5,
        throughput_saturation_ratio=0.9,
        reference_min_frequency_ratio=0.70,
        reference_max_frequency_ratio=0.90,
    )

    assert predict_throughput_tokens_per_s(1100, hardware, features, regularized) == pytest.approx(
        predict_throughput_tokens_per_s(1100, hardware, features, baseline),
        rel=1e-6,
    )
    assert predict_throughput_tokens_per_s(900, hardware, features, regularized) < predict_throughput_tokens_per_s(900, hardware, features, baseline)



def test_low_band_transfer_regularization_requires_reference_topology_features():
    hardware = HardwareFeatures(
        gpu_name="synthetic",
        gpu_count=16,
        supported_frequency_mhz=[900, 1000, 1100, 1200, 1300, 1400],
        min_frequency_mhz=900,
        max_frequency_mhz=1400,
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=100.0,
        peak_fp32_tflops_per_gpu=50.0,
        memory_bandwidth_gbps_per_gpu=900.0,
    )
    target = DerivedModelFeatures(
        tokens_per_step=1000.0,
        samples_per_step=1.0,
        approx_model_params=1.0,
        approx_flops_per_token=1.0,
        approx_flops_per_step=1000.0,
        approx_memory_bytes_per_step=1.0,
        approx_communication_bytes_per_step=0.0,
        arithmetic_intensity_flops_per_byte=1.0,
        communication_share=0.20,
        pipeline_parallel_efficiency=1.0,
        pipeline_exposed_fraction=0.30,
        dp_overlapable_fraction=0.20,
        tp_sync_fraction=0.0,
        hardware_balance_flops_per_byte=1.0,
        compute_weight=1.0,
        memory_weight=0.0,
        communication_weight=0.0,
    )
    without_reference = CalibrationParams(
        compute_limit_at_max_tokens_per_s=1400.0,
        memory_limit_tokens_per_s=1_000_000.0,
        communication_limit_tokens_per_s=1_000_000.0,
        communication_penalty=0.0,
        static_power_w=50.0,
        dynamic_power_w=200.0,
        dynamic_power_exponent=1.5,
        throughput_saturation_ratio=0.9,
        reference_observed_frequency_ratios=(900 / 1400, 930 / 1400, 1185 / 1400, 1260 / 1400),
    )
    with_reference = CalibrationParams(
        compute_limit_at_max_tokens_per_s=1400.0,
        memory_limit_tokens_per_s=1_000_000.0,
        communication_limit_tokens_per_s=1_000_000.0,
        communication_penalty=0.0,
        static_power_w=50.0,
        dynamic_power_w=200.0,
        dynamic_power_exponent=1.5,
        throughput_saturation_ratio=0.9,
        reference_observed_frequency_ratios=(900 / 1400, 930 / 1400, 1185 / 1400, 1260 / 1400),
        reference_pipeline_exposed_fraction=0.05,
        reference_dp_overlapable_fraction=0.0,
        reference_tp_sync_fraction=0.45,
        reference_topology_features_present=True,
    )

    low_freq = 900
    assert predict_throughput_tokens_per_s(low_freq, hardware, target, without_reference) == pytest.approx(
        predict_throughput_tokens_per_s(low_freq, hardware, target, CalibrationParams(
            compute_limit_at_max_tokens_per_s=1400.0,
            memory_limit_tokens_per_s=1_000_000.0,
            communication_limit_tokens_per_s=1_000_000.0,
            communication_penalty=0.0,
            static_power_w=50.0,
            dynamic_power_w=200.0,
            dynamic_power_exponent=1.5,
            throughput_saturation_ratio=0.9,
        )),
        rel=1e-6,
    )
    assert predict_throughput_tokens_per_s(low_freq, hardware, target, with_reference) < predict_throughput_tokens_per_s(
        low_freq, hardware, target, without_reference
    )


def test_large_observation_gap_regularization_makes_mid_gap_more_conservative():
    hardware = HardwareFeatures(
        gpu_name="synthetic",
        gpu_count=16,
        supported_frequency_mhz=[900, 1000, 1100, 1200, 1300, 1400],
        min_frequency_mhz=900,
        max_frequency_mhz=1400,
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=100.0,
        peak_fp32_tflops_per_gpu=50.0,
        memory_bandwidth_gbps_per_gpu=900.0,
    )
    features = DerivedModelFeatures(
        tokens_per_step=1000.0,
        samples_per_step=1.0,
        approx_model_params=1.0,
        approx_flops_per_token=1.0,
        approx_flops_per_step=1000.0,
        approx_memory_bytes_per_step=1.0,
        approx_communication_bytes_per_step=0.0,
        arithmetic_intensity_flops_per_byte=1.0,
        communication_share=0.20,
        pipeline_parallel_efficiency=1.0,
        pipeline_exposed_fraction=0.25,
        dp_overlapable_fraction=0.15,
        tp_sync_fraction=0.0,
        hardware_balance_flops_per_byte=1.0,
        compute_weight=1.0,
        memory_weight=0.0,
        communication_weight=0.0,
    )
    baseline = CalibrationParams(
        compute_limit_at_max_tokens_per_s=1400.0,
        memory_limit_tokens_per_s=1_000_000.0,
        communication_limit_tokens_per_s=1_000_000.0,
        communication_penalty=0.0,
        static_power_w=50.0,
        dynamic_power_w=200.0,
        dynamic_power_exponent=1.5,
        throughput_saturation_ratio=0.9,
    )
    regularized = CalibrationParams(
        compute_limit_at_max_tokens_per_s=1400.0,
        memory_limit_tokens_per_s=1_000_000.0,
        communication_limit_tokens_per_s=1_000_000.0,
        communication_penalty=0.0,
        static_power_w=50.0,
        dynamic_power_w=200.0,
        dynamic_power_exponent=1.5,
        throughput_saturation_ratio=0.9,
        reference_observed_frequency_ratios=(900/1400, 930/1400, 1185/1400, 1260/1400),
    )

    assert predict_throughput_tokens_per_s(1000, hardware, features, regularized) < predict_throughput_tokens_per_s(1000, hardware, features, baseline)
    assert predict_throughput_tokens_per_s(900, hardware, features, regularized) == pytest.approx(
        predict_throughput_tokens_per_s(900, hardware, features, baseline),
        rel=1e-6,
    )
    assert predict_throughput_tokens_per_s(1200, hardware, features, regularized) == pytest.approx(
        predict_throughput_tokens_per_s(1200, hardware, features, baseline),
        rel=0.03,
    )


def test_two_band_correction_can_lower_low_freq_throughput_and_raise_high_freq_power():
    hardware = HardwareFeatures(
        gpu_name="synthetic",
        gpu_count=16,
        supported_frequency_mhz=SUPPORTED_CLOCKS,
        min_frequency_mhz=SUPPORTED_CLOCKS[0],
        max_frequency_mhz=SUPPORTED_CLOCKS[-1],
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=100.0,
        peak_fp32_tflops_per_gpu=50.0,
        memory_bandwidth_gbps_per_gpu=900.0,
    )
    features = DerivedModelFeatures(
        tokens_per_step=1000.0,
        samples_per_step=1.0,
        approx_model_params=1.0,
        approx_flops_per_token=1.0,
        approx_flops_per_step=1000.0,
        approx_memory_bytes_per_step=1.0,
        approx_communication_bytes_per_step=0.0,
        arithmetic_intensity_flops_per_byte=1.0,
        communication_share=0.20,
        pipeline_parallel_efficiency=0.70,
        pipeline_exposed_fraction=0.45,
        dp_overlapable_fraction=0.10,
        tp_sync_fraction=0.35,
        hardware_balance_flops_per_byte=1.0,
        compute_weight=1.0,
        memory_weight=0.0,
        communication_weight=0.0,
    )
    baseline = CalibrationParams(
        compute_limit_at_max_tokens_per_s=1400.0,
        memory_limit_tokens_per_s=1_000_000.0,
        communication_limit_tokens_per_s=1_000_000.0,
        communication_penalty=0.0,
        static_power_w=50.0,
        dynamic_power_w=200.0,
        dynamic_power_exponent=1.5,
        throughput_saturation_ratio=0.9,
    )
    corrected = CalibrationParams(
        compute_limit_at_max_tokens_per_s=1400.0,
        memory_limit_tokens_per_s=1_000_000.0,
        communication_limit_tokens_per_s=1_000_000.0,
        communication_penalty=0.0,
        static_power_w=50.0,
        dynamic_power_w=200.0,
        dynamic_power_exponent=1.5,
        throughput_saturation_ratio=0.9,
        correction_split_ratio=0.72,
        correction_transition_width=0.05,
        throughput_low_freq_correction=-0.10,
        throughput_high_freq_correction=0.0,
        power_low_freq_correction=0.0,
        power_high_freq_correction=0.06,
        correction_topology_weight=1.0,
        correction_communication_weight=0.5,
    )

    low_freq = hardware.min_frequency_mhz
    high_freq = hardware.max_frequency_mhz

    assert predict_throughput_tokens_per_s(low_freq, hardware, features, corrected) < predict_throughput_tokens_per_s(low_freq, hardware, features, baseline)
    assert predict_throughput_tokens_per_s(high_freq, hardware, features, corrected) == pytest.approx(
        predict_throughput_tokens_per_s(high_freq, hardware, features, baseline),
        rel=0.03,
    )
    assert build_prediction_bundle(hardware, features, corrected, comparison_steps=1)["supported_predictions"][-1]["power_w"] > build_prediction_bundle(hardware, features, baseline, comparison_steps=1)["supported_predictions"][-1]["power_w"]


def test_throughput_saturation_ratio_saturates_compute_limit_before_max_clock():
    hardware = HardwareFeatures(
        gpu_name="synthetic",
        gpu_count=1,
        supported_frequency_mhz=SUPPORTED_CLOCKS,
        min_frequency_mhz=SUPPORTED_CLOCKS[0],
        max_frequency_mhz=SUPPORTED_CLOCKS[-1],
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=100.0,
        peak_fp32_tflops_per_gpu=50.0,
        memory_bandwidth_gbps_per_gpu=900.0,
    )
    features = DerivedModelFeatures(
        tokens_per_step=1000.0,
        samples_per_step=1.0,
        approx_model_params=1.0,
        approx_flops_per_token=1.0,
        approx_flops_per_step=1000.0,
        approx_memory_bytes_per_step=1.0,
        approx_communication_bytes_per_step=0.0,
        arithmetic_intensity_flops_per_byte=1.0,
        communication_share=0.0,
        pipeline_parallel_efficiency=1.0,
        pipeline_exposed_fraction=0.0,
        dp_overlapable_fraction=0.0,
        tp_sync_fraction=0.0,
        hardware_balance_flops_per_byte=1.0,
        compute_weight=1.0,
        memory_weight=0.0,
        communication_weight=0.0,
    )
    params = CalibrationParams(
        compute_limit_at_max_tokens_per_s=1400.0,
        memory_limit_tokens_per_s=1_000_000.0,
        communication_limit_tokens_per_s=1_000_000.0,
        communication_penalty=0.0,
        static_power_w=50.0,
        dynamic_power_w=200.0,
        dynamic_power_exponent=1.5,
        throughput_saturation_ratio=0.8,
    )

    early_saturation_freq = int(hardware.max_frequency_mhz * params.throughput_saturation_ratio)
    throughput_at_early_saturation = predict_throughput_tokens_per_s(early_saturation_freq, hardware, features, params)
    throughput_at_max = predict_throughput_tokens_per_s(hardware.max_frequency_mhz, hardware, features, params)
    throughput_below_saturation = predict_throughput_tokens_per_s(1000, hardware, features, params)

    assert throughput_at_early_saturation == pytest.approx(params.compute_limit_at_max_tokens_per_s)
    assert throughput_at_max == pytest.approx(throughput_at_early_saturation)
    assert throughput_below_saturation < throughput_at_early_saturation


def test_load_experiment_samples_falls_back_to_zeus_logs(tmp_path: Path):
    experiment_root = tmp_path / "experiments"
    run_dir = experiment_root / "run_1200"
    _write_run(experiment_root, "run_1200", frequency_mhz=1200, throughput_tokens_per_s=1200.0, avg_power_w=209.0)
    (run_dir / "events.jsonl").unlink()

    steps = 20
    interval_tokens = float(steps * TOKENS_PER_STEP)
    time_s = interval_tokens / 1200.0
    energy_j = 209.0 * time_s
    energy_wh = energy_j / 3600.0
    _write_zeus_log(run_dir, steps=steps, time_s=time_s, avg_power_w=209.0, energy_j=energy_j, energy_wh=energy_wh)

    collection = load_experiment_samples(str(experiment_root))

    assert collection.skipped_runs == []
    assert len(collection.samples) == 1
    observed = collection.samples[0].observed
    assert observed.frequency_mhz == 1200
    assert observed.num_steps == steps
    assert observed.throughput_tokens_per_s == pytest.approx(1200.0, rel=1e-5)
    assert observed.avg_power_w == pytest.approx(209.0, rel=1e-5)



def test_cross_node_penalty_fit_returns_positive_coefficients():
    hardware = HardwareFeatures(
        gpu_name="Tesla V100-SXM3-32GB",
        gpu_count=16,
        supported_frequency_mhz=SUPPORTED_CLOCKS,
        min_frequency_mhz=SUPPORTED_CLOCKS[0],
        max_frequency_mhz=SUPPORTED_CLOCKS[-1],
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=125.0,
        peak_fp32_tflops_per_gpu=15.7,
        memory_bandwidth_gbps_per_gpu=900.0,
    )

    fit = fit_cross_node_penalty_model(hardware)

    assert fit.alpha_pp_s_per_byte >= 0.0
    assert fit.alpha_dp_s_per_byte >= 0.0
    assert fit.alpha_tp_s_per_byte >= 0.0
    assert fit.alpha_dp_s_per_byte > 0.0, "DP cross-node bytes should drive overhead"
    assert fit.reference_cross_node_dp_bytes > 0.0
    assert fit.beta_pp_wait_s >= 0.0
    assert fit.beta_pp_edge_s >= 0.0
    assert fit.power_base_drop > 0.0
    assert 0.74 <= fit.power_low_freq_reference_ratio <= 0.82
    assert len(fit.points) >= 11


def test_cross_node_penalty_slows_and_depowers_multinode_prediction():
    hardware = HardwareFeatures(
        gpu_name="Tesla V100-SXM3-32GB",
        gpu_count=16,
        supported_frequency_mhz=SUPPORTED_CLOCKS,
        min_frequency_mhz=SUPPORTED_CLOCKS[0],
        max_frequency_mhz=SUPPORTED_CLOCKS[-1],
        power_limit_w_per_gpu=300.0,
        peak_fp16_tensor_tflops_per_gpu=125.0,
        peak_fp32_tflops_per_gpu=15.7,
        memory_bandwidth_gbps_per_gpu=900.0,
    )
    params = CalibrationParams(
        compute_limit_at_max_tokens_per_s=50000.0,
        memory_limit_tokens_per_s=45000.0,
        communication_limit_tokens_per_s=40000.0,
        communication_penalty=0.35,
        static_power_w=120.0,
        dynamic_power_w=180.0,
        dynamic_power_exponent=1.6,
        cross_node_alpha_pp_s_per_byte=7.5e-09,
        cross_node_alpha_dp_s_per_byte=1.0e-10,
        cross_node_alpha_tp_s_per_byte=8.8e-11,
    )
    single_node = WorkloadFeatures(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=8,
        train_iters=20,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        data_parallel_size=1,
        zero_stage=1,
        precision_mode="bf16",
        swiglu=True,
        node_count=1,
        gpus_per_node=8,
    )
    dual_node = WorkloadFeatures(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=16,
        train_iters=20,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        data_parallel_size=2,
        zero_stage=1,
        precision_mode="bf16",
        swiglu=True,
        node_count=2,
        gpus_per_node=8,
    )

    single_features = derive_model_features(hardware, single_node)
    dual_features = derive_model_features(hardware, dual_node)

    single_throughput = predict_throughput_tokens_per_s(1200, hardware, single_features, params)
    dual_throughput = predict_throughput_tokens_per_s(1200, hardware, dual_features, params)
    single_power = predict_power_w(1200, hardware, single_features, params)
    dual_power = predict_power_w(1200, hardware, dual_features, params)

    assert dual_features.cross_node_pp_bytes == pytest.approx(0.0)
    assert dual_features.cross_node_dp_bytes > 0.0
    assert dual_features.cross_node_tp_bytes == pytest.approx(0.0)
    assert dual_features.pp_cross_node_edge_fraction == pytest.approx(0.0)
    assert dual_features.dp_cross_node_group_fraction == pytest.approx(1.0)
    assert dual_features.tp_cross_node_group_fraction == pytest.approx(0.0)
    assert dual_throughput < single_throughput
    assert dual_power < single_power
