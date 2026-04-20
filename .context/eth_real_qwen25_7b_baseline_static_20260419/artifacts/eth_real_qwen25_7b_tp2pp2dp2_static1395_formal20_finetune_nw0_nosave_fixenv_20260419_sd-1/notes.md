# Experiment eth_real_qwen25_7b_tp2pp2dp2_static1395_formal20_finetune_nw0_nosave_fixenv_20260419_sd-1

## Metadata
- experiment_name: `eth_real_qwen25_7b_tp2pp2dp2_20260419`
- run_id: `eth_real_qwen25_7b_tp2pp2dp2_static1395_formal20_finetune_nw0_nosave_fixenv_20260419_sd-1`
- mode: `static`
- status: `completed`
- run_dir: `/home/user/Megatron-DeepSpeed/experiments/eth_real_qwen25_7b_tp2pp2dp2_static1395_formal20_finetune_nw0_nosave_fixenv_20260419_sd-1`
- command_sha1: `0d5d477373803aa7b25950baeb2eaa9ce6d9d945`
- framework: `Megatron-DeepSpeed + DeepSpeed launcher + Zeus`
- git_commit: `f275844f71e86e543de51c623956e5445a9232bc`
- topology: `sd-1 + sd-2`, 每机 `GPU 0,1,2,3`, `TP=2 / PP=2 / DP=2`, `world_size=8`
- workload: `Qwen2.5-7B-Instruct`, `28L / H3584 / FFN18944 / A28 / KV4`, `--load qwen25_7b_instruct_hf2megads_tp2pp2_real_main --finetune`
- dataset: `/home/user/Megatron-DeepSpeed/data/qwen_data_text_document`
- tokenizer: `Qwen2.5-7B-Instruct` snapshot `a09a35458c702b33eeacc393d103063234e8bc28`
- training_window: `train-iters=20`, `seq=2048`, `micro=1`, `global=4`, `bf16`, `ZeRO-1 + CPU optimizer/offload`, `num_workers=0`

## Command
```bash
/home/user/miniconda3/envs/tp4bit/bin/python3.10 pretrain_gpt.py --local_rank=0 --tensor-model-parallel-size 2 --pipeline-model-parallel-size 2 --num-layers 28 --hidden-size 3584 --ffn-hidden-size 18944 --num-attention-heads 28 --num-key-value-heads 4 --micro-batch-size 1 --global-batch-size 4 --num-workers 0 --seq-length 2048 --max-position-embeddings 2048 --train-iters 20 --data-path /home/user/Megatron-DeepSpeed/data/qwen_data_text_document --data-cache-path /home/user/Megatron-DeepSpeed/data/index-cache --data-impl mmap --tokenizer-type HFTokenizer --tokenizer-model /home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 --split 98,2,0 --distributed-backend nccl --lr 1e-5 --lr-decay-style cosine --min-lr 1e-6 --weight-decay 0.01 --clip-grad 1.0 --lr-warmup-iters 1 --optimizer adam --cpu-optimizer --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --log-interval 1 --save-interval 0 --eval-interval 100 --eval-iters 0 --no-query-key-layer-scaling --attention-dropout 0 --hidden-dropout 0 --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear --no-position-embedding --no-masked-softmax-fusion --no-bias-gelu-fusion --no-bias-dropout-fusion --recompute-granularity full --recompute-method uniform --deepspeed-activation-checkpointing --zero-stage=1 --deepspeed_config=/home/user/Megatron-DeepSpeed/experiments/eth_real_qwen25_7b_tp2pp2dp2_static1395_formal20_finetune_nw0_nosave_fixenv_20260419_sd-1/ds_config.json --deepspeed --experiment-run-id eth_real_qwen25_7b_tp2pp2dp2_static1395_formal20_finetune_nw0_nosave_fixenv_20260419_sd-1 --experiment-name eth_real_qwen25_7b_tp2pp2dp2_20260419 --experiment-root-dir /home/user/Megatron-DeepSpeed/experiments --bf16 --load /home/user/Megatron-DeepSpeed/checkpoints/qwen25_7b_instruct_hf2megads_tp2pp2_real_main --finetune
```

## Hypothesis
- 在真实 `Qwen2.5-7B-Instruct` checkpoint + 真实数据集 + Ethernet 双机场景下，验证 `1395 MHz` 固定频率是否仍能显著降低功率和能耗。
- 目标不是“绝对最快”，而是在不破坏 workload 公平性的前提下给出真实模型上的 baseline/static 主证据。

## Setup Notes
- GPU / node: `sd-1 + sd-2`, 每机 `GPU 0,1,2,3`
- network / backend: 仅使用 Ethernet, `NCCL_SOCKET_IFNAME=eth0`, `NCCL_IB_DISABLE=1`, `distributed-backend=nccl`
- clock policy: `static`, `STATIC_CLOCK_MHZ=1395`
- launch mode: `deepspeed --hostfile /tmp/hostfile_real_qwen7b_eth_tp2pp2dp2 --include sd-1:0,1,2,3@sd-2:0,1,2,3`
- runtime envs:
  - `DISABLE_SAVE_CHECKPOINT=1`
  - `STATIC_CLOCK_MHZ=1395`
  - `TORCH_EXTENSIONS_DIR=/dev/shm/megatron_real_qwen7b_eth_20260419/torch_extensions_tp4bit`
  - `TMPDIR=/dev/shm/megatron_real_qwen7b_eth_20260419/tmp`
  - `PYTHONPYCACHEPREFIX=/dev/shm/megatron_real_qwen7b_eth_20260419/pycache`
  - `TRITON_CACHE_DIR=/dev/shm/megatron_real_qwen7b_eth_20260419/triton_cache_static1395_nosave_fixenv`
  - `PYTHONDONTWRITEBYTECODE=1`
  - `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - `TORCH_NCCL_BLOCKING_WAIT=1`
- static-only handling:
  - preflight 在两节点都验证 `static_clock_supported=true`
  - launcher 为 static run 单独设置 `MASTER_PORT=30776`
  - 运行前会做 NVML 频点支持检查，并按可见 GPU 范围执行锁频；退出路径负责 reset
- no-save reason: `sd-2` 根分区空间不足，保存训练输出 checkpoint 会触发磁盘写满；本 run 保留 `--load` 真实初始权重，但跳过 `--save`

## Key Metrics
- final_iteration: `20`
- total_time_s: `254.418`
- avg_step_time_ms: `12718.7`
- avg_step_time_ms_excluding_step1: `12306.3`
- last_step_time_ms: `10992.0`
- total_tokens: `163840`
- tokens_per_second_window: `643.98`
- avg_power_w: `221.626`
- total_energy_j: `56385.520`
- tokens_per_j: `2.9057`
- last_loss: `4.676264E+00`
- loss curve note: `step 1` 到 `step 20` 的 `lm loss` 从 `1.080963E+01` 下降到 `4.676264E+00`，与 baseline 末步 loss 对齐，窗口内未出现 skipped / nan iterations

## Comparison
- baseline_run_id: `eth_real_qwen25_7b_tp2pp2dp2_baseline_formal20_finetune_nw0_nosave_fixenv_20260419_sd-1`
- workload parity:
  - 同一真实 checkpoint、同一模型结构、同一数据集、同一 tokenizer、同一 batch、同一拓扑、同一 `20-step` 窗口
  - baseline 与 static 的主要差异只在 `EXPERIMENT_MODE=static` 与 `STATIC_CLOCK_MHZ=1395`
- relative delta versus baseline:
  - runtime: `+10.86%`
  - avg_power: `-30.78%`
  - energy: `-23.26%`
  - tokens_per_j: `+30.31%`
- conclusion: `static 1395 MHz` 在真实模型 Ethernet 双机 workload 上把平均功率降到 `221.63 W`、总能耗降到 `56.39 kJ`，代价是总时长增加约 `10.9%`；这是当前最完整的真实模型 baseline/static 对照之一
