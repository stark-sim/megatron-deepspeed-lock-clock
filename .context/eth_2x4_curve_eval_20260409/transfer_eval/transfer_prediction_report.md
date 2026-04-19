# Transfer Prediction Evaluation

- generated_at: `2026-04-09T03:03:20Z`
- source_root: `/Users/stark_sim/conductor/workspaces/megatron-deepspeed-lock-clock/los-angeles-v1/.context/eth_2x4_curve_eval_20260409/eth_qwen3b_1x4_source_curve_20260409_curated`
- target_root: `/Users/stark_sim/conductor/workspaces/megatron-deepspeed-lock-clock/los-angeles-v1/.context/eth_2x4_curve_eval_20260409/eth_qwen3b_2x4_target_curve_20260408_sd-1`
- network_benchmark_json: `/Users/stark_sim/conductor/workspaces/megatron-deepspeed-lock-clock/los-angeles-v1/.context/comm_bench_2x4_eth0_20260406_175803.json`

## Source Calibration
- sample_count: `3`
- throughput_mape: `0.1191`
- power_mape: `0.1240`
- total_time_mape: `0.1362`
- total_energy_mape: `0.0112`

## Target Transfer Accuracy
- point_count: `3`
- total_time_mape: `0.4017`
- step_time_mape: `0.4017`
- avg_power_mape: `0.2099`
- total_energy_mape: `0.1058`

## Cross-Node Coefficients
- alpha_pp_s_per_byte: `0.000000e+00`
- alpha_dp_s_per_byte: `8.411965e-10`
- alpha_tp_s_per_byte: `0.000000e+00`

## Target Points
- `1005` MHz: time observed/predicted=`217.963/313.381` s, time APE=`0.4378`, step observed/predicted=`10.898/15.669` s
- `1200` MHz: time observed/predicted=`202.778/297.100` s, time APE=`0.4652`, step observed/predicted=`10.139/14.855` s
- `1395` MHz: time observed/predicted=`217.560/283.286` s, time APE=`0.3021`, step observed/predicted=`10.878/14.164` s

## Predicted Sweet Spot
- default_frequency_mhz: `1005`
- recommended_frequencies_mhz: `[1005, 1020, 1035]`
- pareto_frontier_frequencies_mhz: `[2235, 2220, 2205, 2190, 2175, 2160, 2145, 2130, 2115, 2100, 2085, 2070, 2055, 2040, 2025, 2010, 1995, 1980, 1965, 1950, 1935, 1920, 1905, 1890, 1875, 1860, 1845, 1830, 1815, 1800, 1785, 1770, 1755, 1740, 1725, 1710, 1695, 1680, 1665, 1650, 1635, 1620, 1605, 1590, 1575, 1560, 1545, 1530, 1515, 1500, 1485, 1470, 1455, 1440, 1425, 1410, 1395, 1380, 1365, 1350, 1335, 1320, 1305, 1290, 1275, 1260, 1245, 1230, 1215, 1200, 1185, 1170, 1155, 1140, 1125, 1110, 1095, 1080, 1065, 1050, 1035, 1020, 1005]`
