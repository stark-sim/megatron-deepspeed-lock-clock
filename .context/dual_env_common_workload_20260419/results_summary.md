# Dual-Environment Common Workload Summary

Date: `2026-04-19`

## Workload

- Topology: `2 nodes x 4 GPUs`
- Ethernet hosts: `sd-1/sd-2`, GPUs `0,1,2,3`
- IB hosts: `v100x16-1/v100x16-2`, GPUs `8,9,10,11`
- Model: `36L / hidden 2048 / ffn 11008 / heads 16 / kv_heads 2`
- Parallelism: `TP=1 / PP=2 / DP=4`
- Batch: `micro=1 / global=4`
- Sequence length: `2048`
- Train iters: `20`
- Optimizer/runtime: `ZeRO-1 + CPU optimizer/offload`
- Dataset: `qwen_data_text_document`

## Artifact Roots

- Ethernet: `.context/dual_env_common_workload_20260419/artifacts/eth/`
- IB: `.context/dual_env_common_workload_20260419/artifacts/ib/`

## Ethernet Results

| Mode | Time (s) | Energy (J) | Avg Power (W) | Delta Time vs Baseline | Delta Energy vs Baseline | Delta Power vs Baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 168.216 | 51438.765 | 305.790 | +0.00% | +0.00% | +0.00% |
| static 1005 | 198.728 | 40212.953 | 202.352 | +18.14% | -21.82% | -33.83% |
| static 1200 | 184.716 | 38400.352 | 207.889 | +9.81% | -25.35% | -32.02% |
| static 1395 | 181.379 | 38436.713 | 211.914 | +7.82% | -25.28% | -30.70% |

## IB Results

| Mode | Time (s) | Energy (J) | Avg Power (W) | Delta Time vs Baseline | Delta Energy vs Baseline | Delta Power vs Baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 182.482 | 114005.406 | 624.750 | +0.00% | +0.00% | +0.00% |
| static 990 | 244.728 | 88633.158 | 362.170 | +34.11% | -22.26% | -42.03% |
| static 1080 | 229.767 | 86021.344 | 374.385 | +25.91% | -24.55% | -40.07% |
| static 1155 | 212.352 | 84274.511 | 396.863 | +16.37% | -26.08% | -36.48% |
