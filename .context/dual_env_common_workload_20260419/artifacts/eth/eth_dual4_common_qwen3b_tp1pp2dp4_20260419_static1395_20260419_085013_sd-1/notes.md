# Experiment eth_dual4_common_qwen3b_tp1pp2dp4_20260419_static1395_20260419_085013_sd-1

## Metadata
- experiment_name: `eth_dual4_common_qwen3b_tp1pp2dp4_20260419`
- status: `initialized`
- run_dir: `/home/user/Megatron-DeepSpeed/experiments/eth_dual4_common_qwen3b_tp1pp2dp4_20260419_static1395_20260419_085013_sd-1`
- command_sha1: `452435a5832ff7319d2d8fb1d0debb55f6f787fe`

## Command
```bash
/home/user/miniconda3/envs/tp4bit/bin/python3.10 pretrain_gpt.py --local_rank=0 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 --num-layers 36 --hidden-size 2048 --ffn-hidden-size 11008 --num-attention-heads 16 --num-key-value-heads 2 --micro-batch-size 1 --global-batch-size 4 --seq-length 2048 --max-position-embeddings 2048 --train-iters 20 --data-path /home/user/Megatron-DeepSpeed/data/qwen_data_text_document --data-cache-path /home/user/Megatron-DeepSpeed/data/index-cache --data-impl mmap --tokenizer-type HFTokenizer --tokenizer-model /home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 --split 98,2,0 --distributed-backend nccl --lr 1e-5 --lr-decay-style cosine --min-lr 1e-6 --weight-decay 0.01 --clip-grad 1.0 --lr-warmup-iters 19 --optimizer adam --cpu-optimizer --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --log-interval 1 --save-interval 0 --eval-interval 100 --eval-iters 0 --no-query-key-layer-scaling --attention-dropout 0 --hidden-dropout 0 --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear --no-position-embedding --no-masked-softmax-fusion --no-bias-gelu-fusion --no-bias-dropout-fusion --recompute-granularity full --recompute-method uniform --deepspeed-activation-checkpointing --zero-stage=1 --deepspeed_config=/home/user/Megatron-DeepSpeed/experiments/eth_dual4_common_qwen3b_tp1pp2dp4_20260419_static1395_20260419_085013_sd-1/ds_config.json --deepspeed --experiment-run-id eth_dual4_common_qwen3b_tp1pp2dp4_20260419_static1395_20260419_085013_sd-1 --experiment-name eth_dual4_common_qwen3b_tp1pp2dp4_20260419 --experiment-root-dir /home/user/Megatron-DeepSpeed/experiments --bf16
```

## Hypothesis
- 

## Setup Notes
- GPU / node:
- cooling / ambient:
- driver / CUDA:
- clock policy:

## Key Metrics
- step time:
- tokens per second:
- average power:
- energy per 50 steps:
- loss curve note:

## Comparison
- baseline run_id:
- compared dimension:
- conclusion:
