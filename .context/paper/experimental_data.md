# Experimental Data Summary

## Reporting Rule

This file now tracks only the results that are currently supportable by recoverable or
fresh formal artifacts in this workspace. Older `98.5% -> 1.8%` IB numbers remain
useful as historical motivation, but they are not the primary formal checkpoint for the
current paper draft.

## Network Benchmark Results

### InfiniBand (live dual-node benchmark used for the fresh formal replay)

| Message Size (MB) | Bus Bandwidth (Gbps) | CV |
|-------------------|----------------------|----|
| 1                 | 6.39                 | 0.128 |
| 4                 | 9.84                 | 0.079 |
| 16                | 11.54                | 0.513 |
| 64                | 18.76                | 0.031 |

Note: a separate historical single-node benchmark reached `111.48 Gbps` at `256 MB`,
but that point is now treated as a sensitivity reference rather than the paired formal
benchmark for the current `2x4 -> 2x8` replay.

### Ethernet (formal `1x4 -> 2x4` replay benchmark)

| Message Size (MB) | Bus Bandwidth (Gbps) | CV |
|-------------------|----------------------|----|
| 1                 | 0.116                | 0.0048 |
| 4                 | 0.117                | 0.0010 |
| 16                | 0.200                | 0.0153 |
| 64                | 0.203                | 0.0072 |

## Fresh IB Formal Measurements

### Source Topology: `2x4`, `TP=4`, `PP=1`, `DP=2`, `GBS=8`

| Frequency (MHz) | Time (s) | Avg Power (W) | Energy (kJ) | Tokens/J |
|-----------------|----------|---------------|-------------|----------|
| 990             | 404.13   | 591.25        | 238.94      | 1.371 |
| 1080            | 374.76   | 628.60        | 235.57      | 1.391 |
| 1155            | 353.79   | 674.96        | 238.79      | 1.372 |

Current source calibration on this fresh source curve:

- Time MAPE: `1.02%`
- Power MAPE: `10.33%`
- Energy MAPE: `10.65%`

### Target Topology: `2x8`, `TP=4`, `PP=1`, `DP=4`, `GBS=16`

| Frequency (MHz) | Time (s) | Avg Power (W) | Energy (kJ) | Tokens/J |
|-----------------|----------|---------------|-------------|----------|
| 990             | 401.19   | 1189.17       | 477.08      | 1.374 |
| 1080            | 372.24   | 1257.39       | 468.05      | 1.400 |
| 1155            | 351.53   | 1353.14       | 475.67      | 1.378 |

Observation: source and target step times stay within about `1%`, while target
per-node power is almost exactly `2x` the source because `gpus_per_node` doubles
from `4` to `8`.

## Formal Transfer Results

### Historical Legacy Failure (motivation)

- Older slow-network assumptions produced `98.5%` time MAPE on `2x4 -> 2x8` IB transfer.
- This remains the motivating failure mode, but it should not be cited as the current
  formal paired replay result.

### Current IB Formal Replay (`2x4 -> 2x8`, transport-consistent, power-fixed)

Artifacts:

- `.context/transfer_eval_ib_2x4_to_2x8_rerun_topology_fixed_live_ib_powerfix_20260411/transfer_prediction.json`
- `.context/transfer_eval_ib_2x4_to_2x8_rerun_topology_fixed_live_ib_powerfix_20260411/transfer_prediction_report.md`

Aggregate metrics:

- Time MAPE: `11.48%`
- Step-Time MAPE: `11.48%`
- Avg-Power MAPE: `3.28%`
- Total-Energy MAPE: `7.86%`
- `alpha_dp = 2.220525e-11 s/byte`

Per-frequency replay details:

| Frequency (MHz) | Pred Time (s) | Obs Time (s) | Pred Power (W) | Obs Power (W) | Time APE | Power APE | Energy APE |
|-----------------|---------------|--------------|----------------|---------------|----------|-----------|------------|
| 990             | 435.40        | 401.19       | 1128.43        | 1189.17       | 8.53%    | 5.11%     | 2.98%      |
| 1080            | 415.90        | 372.24       | 1229.19        | 1257.39       | 11.73%   | 2.24%     | 9.22%      |
| 1155            | 401.42        | 351.53       | 1319.64        | 1353.14       | 14.19%   | 2.48%     | 11.37%     |

Sensitivity note:

- Replacing the live dual-node benchmark with the historical `111.48 Gbps`
  single-node reference only shifts IB time MAPE from `11.48%` to `11.43%`.
- Therefore the remaining runtime error is no longer primarily a benchmark
  provenance issue.

### Current Ethernet Formal Replay (`1x4 -> 2x4`)

Artifacts:

- `.context/transfer_eval_eth_qwen3b_1x4_source_curve_20260409_curated_to_eth_qwen3b_2x4_target_curve_20260408_sd-1/transfer_prediction.json`
- `.context/transfer_eval_eth_qwen3b_1x4_source_curve_20260409_curated_to_eth_qwen3b_2x4_target_curve_20260408_sd-1/transfer_prediction_report.md`

Aggregate metrics:

- Time MAPE: `5.16%`
- Step-Time MAPE: `5.16%`
- Avg-Power MAPE: `12.38%`
- Total-Energy MAPE: `10.42%`
- `alpha_dp = 7.321889e-10 s/byte`

## Penalty Coefficients

| Environment | Formal Replay Pair | `alpha_dp` (s/byte) |
|-------------|--------------------|---------------------|
| IB          | `2x4 -> 2x8`       | `2.220525e-11` |
| Ethernet    | `1x4 -> 2x4`       | `7.321889e-10` |

The current predictor therefore keeps a clear separation between fast-network and
slow-network behavior, but via continuous benchmark-scaled parameters rather than a
binary threshold switch.

## Current Paper-Safe Claims

1. The historical fixed-coefficient model can fail catastrophically (`98.5%` time MAPE)
   when network assumptions are badly mismatched.
2. On a fresh transport-consistent IB rerun, the current model reaches
   `11.48%` time MAPE, `3.28%` power MAPE, and `7.86%` energy MAPE.
3. On the formal Ethernet replay, the current model reaches `5.16%` time MAPE.
4. Average power transfer must explicitly account for `gpus_per_node` growth; after
   adding this structural scaling, IB power MAPE collapses from `51.64%` to `3.28%`.
