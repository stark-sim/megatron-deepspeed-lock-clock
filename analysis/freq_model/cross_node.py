from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from analysis.freq_model.features import derive_model_features
from analysis.freq_model.hardware import HardwareFeatures
from analysis.freq_model.workload import WorkloadFeatures


@dataclass(frozen=True)
class CrossNodeCalibrationPoint:
    name: str
    workload: WorkloadFeatures
    single_node_time_s: float
    dual_node_time_s: float
    frequency_mhz: float = 1200.0  # Default frequency for backward compatibility
    weight: float = 1.0

    @property
    def overhead_time_s(self) -> float:
        return max(self.dual_node_time_s - self.single_node_time_s, 0.0)

    @property
    def overhead_time_per_step_s(self) -> float:
        return self.overhead_time_s / max(self.workload.train_iters, 1)


@dataclass(frozen=True)
class CrossNodePowerCalibrationPoint:
    name: str
    workload: WorkloadFeatures
    frequency_mhz: float
    single_node_power_w: float
    dual_node_cluster_power_w: float
    weight: float = 1.0

    @property
    def observed_power_ratio(self) -> float:
        return self.dual_node_cluster_power_w / max(self.single_node_power_w, 1e-9)


@dataclass(frozen=True)
class CrossNodeFitResult:
    alpha_pp_s_per_byte: float
    alpha_dp_s_per_byte: float
    alpha_tp_s_per_byte: float
    reference_cross_node_pp_bytes: float = 0.0
    reference_pp_cross_node_wait_pressure: float = 0.0
    reference_cross_node_dp_bytes: float = 0.0
    beta_pp_wait_s: float = 0.0
    beta_pp_edge_s: float = 0.0
    power_base_drop: float = 0.0
    power_low_freq_reference_ratio: float = 0.782
    power_low_freq_gamma: float = 0.0
    points: List[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _v100_dual_node_calibration_points() -> List[CrossNodeCalibrationPoint]:
    common = dict(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        seq_length=2048,
        micro_batch_size=1,
        train_iters=20,
        zero_stage=1,
        precision_mode='bf16',
        swiglu=True,
        node_count=2,
        gpus_per_node=8,
    )
    legacy_weight = 0.05
    return [
        CrossNodeCalibrationPoint(
            name='tp2_pp4_dp2_static1185',
            workload=WorkloadFeatures(global_batch_size=16, tensor_model_parallel_size=2, pipeline_model_parallel_size=4, data_parallel_size=2, **common),
            single_node_time_s=449.70630288124084,
            dual_node_time_s=456.8,
            frequency_mhz=1185.0,
        ),
        CrossNodeCalibrationPoint(
            name='tp2_pp4_dp2_static1200',
            workload=WorkloadFeatures(global_batch_size=16, tensor_model_parallel_size=2, pipeline_model_parallel_size=4, data_parallel_size=2, **common),
            single_node_time_s=444.87426137924194,
            dual_node_time_s=455.7,
            frequency_mhz=1200.0,
        ),
        CrossNodeCalibrationPoint(
            name='tp2_pp4_dp2_static1215',
            workload=WorkloadFeatures(global_batch_size=16, tensor_model_parallel_size=2, pipeline_model_parallel_size=4, data_parallel_size=2, **common),
            single_node_time_s=440.6333487033844,
            dual_node_time_s=447.5,
            frequency_mhz=1215.0,
        ),
        # Re-measured 2026-03-23 with consistent network conditions.
        # single_node_time_s is ESTIMATED (never ran TP2PP2DP4 on single 16-GPU node).
        # Estimates derived from TP2PP4DP2 scaling; actual value may differ by ±30s.
        # Weight reduced to 0.3 to limit influence of unreliable overhead calculation.
        # TODO: run actual single-node TP2PP2DP4 on v100x16-1 to replace estimates.
        CrossNodeCalibrationPoint(
            name='tp2_pp2_dp4_static1072',
            workload=WorkloadFeatures(global_batch_size=16, tensor_model_parallel_size=2, pipeline_model_parallel_size=2, data_parallel_size=4, **common),
            single_node_time_s=421.4,  # estimated
            dual_node_time_s=444.4,
            frequency_mhz=1072.0,
            weight=0.3,
        ),
        CrossNodeCalibrationPoint(
            name='tp2_pp2_dp4_static1080',
            workload=WorkloadFeatures(global_batch_size=16, tensor_model_parallel_size=2, pipeline_model_parallel_size=2, data_parallel_size=4, **common),
            single_node_time_s=418.5,  # estimated
            dual_node_time_s=462.7,
            frequency_mhz=1080.0,
            weight=0.3,
        ),
        CrossNodeCalibrationPoint(
            name='tp2_pp2_dp4_static1087',
            workload=WorkloadFeatures(global_batch_size=16, tensor_model_parallel_size=2, pipeline_model_parallel_size=2, data_parallel_size=4, **common),
            single_node_time_s=416.0,  # estimated
            dual_node_time_s=464.3,
            frequency_mhz=1087.0,
            weight=0.3,
        ),
        # Real TP=1,PP=4,DP=4 dual-node static frequency measurements (2026-03-22)
        # Single-node: 16 GPUs on sd@v100x16-1; dual-node: 8+8 GPUs on v100x16-{1,2}
        CrossNodeCalibrationPoint(
            name='tp1_pp4_dp4_static1185',
            workload=WorkloadFeatures(global_batch_size=16, tensor_model_parallel_size=1, pipeline_model_parallel_size=4, data_parallel_size=4, **common),
            single_node_time_s=545.5,
            dual_node_time_s=554.1,
            frequency_mhz=1185.0,
        ),
        CrossNodeCalibrationPoint(
            name='tp1_pp4_dp4_static1192',
            workload=WorkloadFeatures(global_batch_size=16, tensor_model_parallel_size=1, pipeline_model_parallel_size=4, data_parallel_size=4, **common),
            single_node_time_s=545.0,
            dual_node_time_s=554.4,
            frequency_mhz=1192.0,
        ),
        CrossNodeCalibrationPoint(
            name='tp1_pp4_dp4_static1200',
            workload=WorkloadFeatures(global_batch_size=16, tensor_model_parallel_size=1, pipeline_model_parallel_size=4, data_parallel_size=4, **common),
            single_node_time_s=539.7,
            dual_node_time_s=552.8,
            frequency_mhz=1200.0,
        ),
    ]


def _v100_dual_node_power_calibration_points() -> List[CrossNodePowerCalibrationPoint]:
    common = dict(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=16,
        train_iters=20,
        zero_stage=1,
        precision_mode='bf16',
        swiglu=True,
        node_count=2,
        gpus_per_node=8,
    )
    return [
        CrossNodePowerCalibrationPoint(
            name='tp2_pp4_dp2_static1185_power',
            workload=WorkloadFeatures(tensor_model_parallel_size=2, pipeline_model_parallel_size=4, data_parallel_size=2, **common),
            frequency_mhz=1185.0,
            single_node_power_w=2422.202410820474,
            dual_node_cluster_power_w=2334.2,
        ),
        CrossNodePowerCalibrationPoint(
            name='tp2_pp4_dp2_static1200_power',
            workload=WorkloadFeatures(tensor_model_parallel_size=2, pipeline_model_parallel_size=4, data_parallel_size=2, **common),
            frequency_mhz=1200.0,
            single_node_power_w=2468.240139574817,
            dual_node_cluster_power_w=2363.6,
        ),
        CrossNodePowerCalibrationPoint(
            name='tp2_pp4_dp2_static1215_power',
            workload=WorkloadFeatures(tensor_model_parallel_size=2, pipeline_model_parallel_size=4, data_parallel_size=2, **common),
            frequency_mhz=1215.0,
            single_node_power_w=2522.288052573245,
            dual_node_cluster_power_w=2432.0,
        ),
        CrossNodePowerCalibrationPoint(
            name='tp2_pp2_dp4_static1072_power',
            workload=WorkloadFeatures(tensor_model_parallel_size=2, pipeline_model_parallel_size=2, data_parallel_size=4, **common),
            frequency_mhz=1072.0,
            single_node_power_w=2349.845119661315,
            dual_node_cluster_power_w=2123.3145877716224,
        ),
        CrossNodePowerCalibrationPoint(
            name='tp2_pp2_dp4_static1080_power',
            workload=WorkloadFeatures(tensor_model_parallel_size=2, pipeline_model_parallel_size=2, data_parallel_size=4, **common),
            frequency_mhz=1080.0,
            single_node_power_w=2362.4747064926173,
            dual_node_cluster_power_w=2198.4718698275115,
        ),
        CrossNodePowerCalibrationPoint(
            name='tp2_pp2_dp4_static1087_power',
            workload=WorkloadFeatures(tensor_model_parallel_size=2, pipeline_model_parallel_size=2, data_parallel_size=4, **common),
            frequency_mhz=1087.0,
            single_node_power_w=2386.640305356655,
            dual_node_cluster_power_w=2168.0343278453625,
        ),
        # Real TP=1,PP=4,DP=4 power calibration (2026-03-22)
        # dual_node_cluster_power_w = rank-0 node Zeus power × 2
        CrossNodePowerCalibrationPoint(
            name='tp1_pp4_dp4_static1185_power',
            workload=WorkloadFeatures(tensor_model_parallel_size=1, pipeline_model_parallel_size=4, data_parallel_size=4, **common),
            frequency_mhz=1185.0,
            single_node_power_w=2171.8,
            dual_node_cluster_power_w=1053.0 * 2,
        ),
        CrossNodePowerCalibrationPoint(
            name='tp1_pp4_dp4_static1192_power',
            workload=WorkloadFeatures(tensor_model_parallel_size=1, pipeline_model_parallel_size=4, data_parallel_size=4, **common),
            frequency_mhz=1192.0,
            single_node_power_w=2194.4,
            dual_node_cluster_power_w=1055.0 * 2,
        ),
        CrossNodePowerCalibrationPoint(
            name='tp1_pp4_dp4_static1200_power',
            workload=WorkloadFeatures(tensor_model_parallel_size=1, pipeline_model_parallel_size=4, data_parallel_size=4, **common),
            frequency_mhz=1200.0,
            single_node_power_w=2216.9,
            dual_node_cluster_power_w=1055.7 * 2,
        ),
    ]


def _proxy_terms(
    hardware: HardwareFeatures,
    workload: WorkloadFeatures,
) -> tuple[float, float, float, Dict[str, Any]]:
    """Extract cross-node relevant features for holistic multi-node modeling.

    Instead of treating cross-node as 'single-node + additive penalty',
    we model multi-node performance holistically by considering:
    1. Cross-node communication volume (bytes)
    2. Topology-structure interactions (how PP/DP/TP map to nodes)
    3. Backpressure effects (how cross-node waits affect local performance)
    """
    features = derive_model_features(hardware, workload)

    # Raw cross-node bytes for PP/DP/TP
    c_pp = features.cross_node_pp_bytes
    c_dp = features.cross_node_dp_bytes
    c_tp = features.cross_node_tp_bytes

    # Topology structure features
    pp_size = workload.pipeline_model_parallel_size
    dp_size = workload.data_parallel_size
    tp_size = workload.tensor_model_parallel_size
    node_count = max(workload.node_count, 1)

    # Interaction: PP depth affects how much DP allreduce can be hidden
    # Deeper PP = more pipeline stages = more opportunities for overlap
    # But also = more potential for cross-node PP edges (if PP > GPUs-per-node)
    pp_dp_interaction = features.pipeline_bubble_fraction * features.dp_cross_node_group_fraction

    # Interaction: TP width affects effective DP group size
    # TP shards parameters, so larger TP = smaller per-rank allreduce
    tp_dp_interaction = (tp_size / max(pp_size * dp_size, 1)) * features.dp_cross_node_group_fraction

    # Backpressure: How much cross-node communication pressure per local GPU?
    # More cross-node groups per node = more NIC pressure = more backpressure
    total_dp_groups = pp_size * tp_size
    cross_node_dp_groups = total_dp_groups * features.dp_cross_node_group_fraction
    cross_node_pressure_per_node = cross_node_dp_groups / max(node_count, 1)

    # Pipeline serialization: With cross-node PP edges, the bubble becomes
    # latency-dominated rather than compute-dominated
    pp_cross_node_serialization = features.pp_cross_node_edge_fraction * pp_size

    return c_pp, c_dp, c_tp, {
        'pipeline_exposed_fraction': features.pipeline_exposed_fraction,
        'dp_overlapable_fraction': features.dp_overlapable_fraction,
        'tp_sync_fraction': features.tp_sync_fraction,
        'pipeline_bubble_fraction': features.pipeline_bubble_fraction,
        'replicas_per_node': features.replicas_per_node,
        'pp_cross_node_wait_pressure': features.pp_cross_node_wait_pressure,
        'pp_cross_node_edge_fraction': features.pp_cross_node_edge_fraction,
        'dp_cross_node_group_fraction': features.dp_cross_node_group_fraction,
        'tp_cross_node_group_fraction': features.tp_cross_node_group_fraction,
        'cross_node_pp_bytes': features.cross_node_pp_bytes,
        'cross_node_dp_bytes': features.cross_node_dp_bytes,
        'cross_node_tp_bytes': features.cross_node_tp_bytes,
        # New holistic interaction features
        'pp_dp_interaction': pp_dp_interaction,
        'tp_dp_interaction': tp_dp_interaction,
        'cross_node_pressure_per_node': cross_node_pressure_per_node,
        'pp_cross_node_serialization': pp_cross_node_serialization,
        'total_dp_groups': total_dp_groups,
        'cross_node_dp_groups': cross_node_dp_groups,
    }


def _least_squares(values: List[List[float]], targets: List[float], columns: tuple[int, ...]) -> List[float]:
    if not columns:
        return [0.0 for _ in range(len(values[0]) if values else 0)]

    gram = [[0.0 for _ in columns] for _ in columns]
    rhs = [0.0 for _ in columns]
    for row, target in zip(values, targets):
        active = [row[index] for index in columns]
        for i, left in enumerate(active):
            rhs[i] += left * target
            for j, right in enumerate(active):
                gram[i][j] += left * right

    size = len(columns)
    augmented = [gram_row[:] + [rhs_value] for gram_row, rhs_value in zip(gram, rhs)]
    for pivot in range(size):
        best = max(range(pivot, size), key=lambda index: abs(augmented[index][pivot]))
        if abs(augmented[best][pivot]) < 1e-30:
            return [0.0 for _ in range(len(values[0]) if values else 0)]
        augmented[pivot], augmented[best] = augmented[best], augmented[pivot]
        pivot_value = augmented[pivot][pivot]
        for column in range(pivot, size + 1):
            augmented[pivot][column] /= pivot_value
        for row_index in range(size):
            if row_index == pivot:
                continue
            factor = augmented[row_index][pivot]
            for column in range(pivot, size + 1):
                augmented[row_index][column] -= factor * augmented[pivot][column]

    solution = [0.0 for _ in range(len(values[0]) if values else 0)]
    for position, column in enumerate(columns):
        solution[column] = max(augmented[position][size], 0.0)
    return solution


def _solve_non_negative_least_squares(values: List[List[float]], targets: List[float]) -> tuple[float, ...]:
    if not values:
        return ()
    feature_count = len(values[0])
    best_error = None
    best_solution = tuple(0.0 for _ in range(feature_count))
    for active_size in range(1, feature_count + 1):
        for active_columns in itertools.combinations(range(feature_count), active_size):
            solution = _least_squares(values, targets, active_columns)
            if any(coefficient < -1e-12 for coefficient in solution):
                continue
            error = 0.0
            for row, target in zip(values, targets):
                prediction = sum(coefficient * value for coefficient, value in zip(solution, row))
                error += (prediction - target) ** 2
            if best_error is None or error < best_error:
                best_error = error
                best_solution = tuple(solution)
    return best_solution


def _fit_power_drop_model(
    hardware: HardwareFeatures,
    reference_cross_node_dp_bytes: float,
) -> tuple[float, float, float, List[Dict[str, Any]]]:
    power_points = _v100_dual_node_power_calibration_points()
    max_frequency = float(hardware.max_frequency_mhz or 1.0)
    rows = []
    for point in power_points:
        _, _, _, diagnostics = _proxy_terms(hardware, point.workload)
        dp_scale = float(diagnostics['cross_node_dp_bytes']) / max(reference_cross_node_dp_bytes, 1e-9)
        rows.append((point, dp_scale, dp_scale, diagnostics))

    best = None
    for ref_ratio_milli in range(740, 821, 2):
        reference_ratio = ref_ratio_milli / 1000.0
        values = []
        targets = []
        for point, wait_scale, byte_scale, _ in rows:
            frequency_ratio = float(point.frequency_mhz) / max(max_frequency, 1e-9)
            low_gap = max(reference_ratio - frequency_ratio, 0.0)
            scale = point.weight ** 0.5
            values.append([wait_scale * scale, byte_scale * low_gap * scale])
            targets.append((1.0 - point.observed_power_ratio) * scale)
        base_drop, low_freq_gamma = _solve_non_negative_least_squares(values, targets)
        error = 0.0
        payloads = []
        for point, wait_scale, byte_scale, diagnostics in rows:
            frequency_ratio = float(point.frequency_mhz) / max(max_frequency, 1e-9)
            low_gap = max(reference_ratio - frequency_ratio, 0.0)
            predicted_ratio = max(0.85, min(1.05, 1.0 - (base_drop * wait_scale) - (low_freq_gamma * byte_scale * low_gap)))
            payloads.append({
                'name': point.name,
                'frequency_mhz': point.frequency_mhz,
                'observed_power_ratio': point.observed_power_ratio,
                'predicted_power_ratio': predicted_ratio,
                'wait_scale': wait_scale,
                'byte_scale': byte_scale,
                'power_ratio_ape': abs(predicted_ratio - point.observed_power_ratio) / max(point.observed_power_ratio, 1e-9),
                **diagnostics,
            })
            error += point.weight * ((predicted_ratio - point.observed_power_ratio) ** 2)
        candidate = (error, base_drop, reference_ratio, low_freq_gamma, payloads)
        if best is None or candidate[0] < best[0]:
            best = candidate

    _, base_drop, reference_ratio, low_freq_gamma, payloads = best
    return base_drop, reference_ratio, low_freq_gamma, payloads


def fit_cross_node_penalty_model(
    hardware: HardwareFeatures,
) -> CrossNodeFitResult:
    """Fit holistic multi-node time penalty model.

    Fits overhead_time_per_step_s (seconds/step) as a function of cross-node
    topology features. The coefficients are in s/byte units, directly usable by
    estimate_cross_node_time_penalty_s() in model.py which adds the result to
    base_step_time_s.

    Topology interaction features capture effects beyond simple communication volume:
    - pp_dp_interaction: how PP depth affects DP allreduce hiding opportunity
    - cross_node_pressure_per_node: NIC contention from concurrent cross-node groups
    """
    rows = []
    for point in _v100_dual_node_calibration_points():
        c_pp, c_dp, c_tp, diagnostics = _proxy_terms(hardware, point.workload)
        # Target: per-step overhead in seconds (matches estimate_cross_node_time_penalty_s units)
        overhead_per_step_s = point.overhead_time_per_step_s
        rows.append((point, c_pp, c_dp, c_tp, overhead_per_step_s, diagnostics))

    # Group by topology (features that define the cross-node structure)
    grouped: Dict[tuple[float, ...], Dict[str, Any]] = {}
    for point, c_pp, c_dp, c_tp, overhead_per_step_s, diagnostics in rows:
        key = (
            round(c_pp, 6),
            round(c_dp, 6),
            round(c_tp, 6),
            round(float(diagnostics['pp_cross_node_wait_pressure']), 6),
            round(float(diagnostics['pp_cross_node_edge_fraction']), 6),
            round(float(diagnostics['pp_dp_interaction']), 6),
            round(float(diagnostics['cross_node_pressure_per_node']), 6),
        )
        bucket = grouped.setdefault(
            key,
            {
                'c_pp': c_pp,
                'c_dp': c_dp,
                'c_tp': c_tp,
                'pp_wait': float(diagnostics['pp_cross_node_wait_pressure']),
                'pp_edge': float(diagnostics['pp_cross_node_edge_fraction']),
                'pp_dp_interaction': float(diagnostics['pp_dp_interaction']),
                'cross_node_pressure_per_node': float(diagnostics['cross_node_pressure_per_node']),
                'weight_sum': 0.0,
                'overhead_weighted_sum': 0.0,
            },
        )
        bucket['weight_sum'] += point.weight
        bucket['overhead_weighted_sum'] += point.weight * overhead_per_step_s

    topology_rows = list(grouped.values())
    values = []
    targets = []
    for row in topology_rows:
        weighted_overhead = row['overhead_weighted_sum'] / max(row['weight_sum'], 1e-9)
        scale = row['weight_sum'] ** 0.5
        # Feature vector: bytes terms + topology interaction terms
        values.append([
            row['c_pp'] * scale,
            row['c_dp'] * scale,
            row['c_tp'] * scale,
            row['pp_wait'] * scale,
            row['pp_edge'] * scale,
            row['pp_dp_interaction'] * scale,
            row['cross_node_pressure_per_node'] * scale,
        ])
        targets.append(weighted_overhead * scale)

    # Fit coefficients for all features
    coeffs = _solve_non_negative_least_squares(values, targets)
    alpha_pp, alpha_dp, alpha_tp, beta_pp_wait, beta_pp_edge, gamma_pp_dp, delta_pressure = coeffs

    reference_point = next((row for row in rows if row[0].name.startswith('tp2_pp4_dp2')), rows[0])
    reference_diagnostics = reference_point[-1]
    reference_cross_node_pp_bytes = float(reference_diagnostics['cross_node_pp_bytes'])
    reference_pp_cross_node_wait_pressure = float(reference_diagnostics['pp_cross_node_wait_pressure'])
    reference_cross_node_dp_bytes = float(reference_diagnostics['cross_node_dp_bytes'])
    power_base_drop, power_low_freq_reference_ratio, power_low_freq_gamma, power_payloads = _fit_power_drop_model(
        hardware,
        reference_cross_node_dp_bytes=reference_cross_node_dp_bytes,
    )

    point_payloads: List[Dict[str, Any]] = []
    for point, c_pp, c_dp, c_tp, overhead_per_step_s, diagnostics in rows:
        # Predict per-step overhead using all features
        predicted_overhead_per_step_s = (
            (alpha_pp * c_pp)
            + (alpha_dp * c_dp)
            + (alpha_tp * c_tp)
            + (beta_pp_wait * float(diagnostics['pp_cross_node_wait_pressure']))
            + (beta_pp_edge * float(diagnostics['pp_cross_node_edge_fraction']))
            + (gamma_pp_dp * float(diagnostics['pp_dp_interaction']))
            + (delta_pressure * float(diagnostics['cross_node_pressure_per_node']))
        )
        predicted_overhead_s = predicted_overhead_per_step_s * point.workload.train_iters
        predicted_dual_time_s = point.single_node_time_s + predicted_overhead_s

        point_payloads.append(
            {
                'name': point.name,
                'observed_overhead_per_step_s': overhead_per_step_s,
                'predicted_overhead_per_step_s': predicted_overhead_per_step_s,
                'observed_dual_node_time_s': point.dual_node_time_s,
                'predicted_dual_node_time_s': predicted_dual_time_s,
                'observed_overhead_s': point.overhead_time_s,
                'predicted_overhead_s': predicted_overhead_s,
                'single_node_time_s': point.single_node_time_s,
                'overhead_per_step_ape': abs(predicted_overhead_per_step_s - overhead_per_step_s) / max(overhead_per_step_s, 1e-9),
                'overhead_ape': abs(predicted_overhead_s - point.overhead_time_s) / max(point.overhead_time_s, 1e-9),
                'weight': point.weight,
                'reference_cross_node_pp_bytes': reference_cross_node_pp_bytes,
                'reference_pp_cross_node_wait_pressure': reference_pp_cross_node_wait_pressure,
                'reference_cross_node_dp_bytes': reference_cross_node_dp_bytes,
                'beta_pp_wait_s': beta_pp_wait,
                'beta_pp_edge_s': beta_pp_edge,
                'gamma_pp_dp': gamma_pp_dp,
                'delta_pressure': delta_pressure,
                'power_base_drop': power_base_drop,
                'power_low_freq_reference_ratio': power_low_freq_reference_ratio,
                'power_low_freq_gamma': power_low_freq_gamma,
                **diagnostics,
            }
        )
    point_payloads.extend(power_payloads)

    return CrossNodeFitResult(
        alpha_pp_s_per_byte=alpha_pp,
        alpha_dp_s_per_byte=alpha_dp,
        alpha_tp_s_per_byte=alpha_tp,
        reference_cross_node_pp_bytes=reference_cross_node_pp_bytes,
        reference_pp_cross_node_wait_pressure=reference_pp_cross_node_wait_pressure,
        reference_cross_node_dp_bytes=reference_cross_node_dp_bytes,
        beta_pp_wait_s=beta_pp_wait,
        beta_pp_edge_s=beta_pp_edge,
        power_base_drop=power_base_drop,
        power_low_freq_reference_ratio=power_low_freq_reference_ratio,
        power_low_freq_gamma=power_low_freq_gamma,
        points=point_payloads,
    )
