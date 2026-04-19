# IB Formal Rerun Replay (Topology Fixed, Power Scale)

- source_root: `/Users/stark_sim/conductor/workspaces/megatron-deepspeed-lock-clock/los-angeles-v1/.context/ib_formal_rerun_20260410/source_curated`
- target_root: `/Users/stark_sim/conductor/workspaces/megatron-deepspeed-lock-clock/los-angeles-v1/.context/ib_formal_rerun_20260410/target_final`
- network_benchmark_json: `/Users/stark_sim/conductor/workspaces/megatron-deepspeed-lock-clock/los-angeles-v1/.context/ib_formal_replay_20260410/comm_bench_2x4_20260410_ibhist.json`
- source_override: `{'num_layers': 28, 'hidden_size': 3584, 'ffn_hidden_size': 18944, 'num_attention_heads': 28, 'num_key_value_heads': 4, 'seq_length': 2048, 'micro_batch_size': 1, 'global_batch_size': 8, 'train_iters': 20, 'tensor_model_parallel_size': 4, 'pipeline_model_parallel_size': 1, 'data_parallel_size': 2, 'zero_stage': 1, 'precision_mode': 'bf16', 'swiglu': True, 'node_count': 2, 'gpus_per_node': 4}`
- target_override: `{'num_layers': 28, 'hidden_size': 3584, 'ffn_hidden_size': 18944, 'num_attention_heads': 28, 'num_key_value_heads': 4, 'seq_length': 2048, 'micro_batch_size': 1, 'global_batch_size': 16, 'train_iters': 20, 'tensor_model_parallel_size': 4, 'pipeline_model_parallel_size': 1, 'data_parallel_size': 4, 'zero_stage': 1, 'precision_mode': 'bf16', 'swiglu': True, 'node_count': 2, 'gpus_per_node': 8}`

## Legacy
- alpha_dp: `8.411965e-10`
- target total_time_mape: `0.1381`
- target step_time_mape: `0.1381`
- target avg_power_mape: `0.0683`
- target total_energy_mape: `0.0695`

## Current
- alpha_dp: `2.220525e-11`
- target total_time_mape: `0.1148`
- target step_time_mape: `0.1148`
- target avg_power_mape: `0.0328`
- target total_energy_mape: `0.0786`

## Current Points
- `990` MHz: time `401.187/435.396` s, power `1189.172/1128.432` W, step APE `0.0853`, power APE `0.0511`, energy APE `0.0298`
- `1080` MHz: time `372.241/415.901` s, power `1257.393/1229.189` W, step APE `0.1173`, power APE `0.0224`, energy APE `0.0922`
- `1155` MHz: time `351.530/401.421` s, power `1353.139/1319.636` W, step APE `0.1419`, power APE `0.0248`, energy APE `0.1137`
