# Experiment ib_dual4_common_qwen3b_tp1pp2dp4_20260419_baseline_20260419_162337_v100x16-1

## Metadata
- experiment_name: `ib_dual4_common_qwen3b_tp1pp2dp4_20260419`
- status: `initialized`
- run_dir: `/home/sd/Megatron-DeepSpeed/experiments/ib_dual4_common_qwen3b_tp1pp2dp4_20260419_baseline_20260419_162337_v100x16-1`
- command_sha1: `0252283df82aebb6097a07dff181a961f29b5cf6`

## Command
```bash
/usr/bin/python3 pretrain_gpt.py --local_rank=0 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 --num-layers 36 --hidden-size 2048 --ffn-hidden-size 11008 --num-attention-heads 16 --num-key-value-heads 2 --micro-batch-size 1 --global-batch-size 4 --seq-length 2048 --max-position-embeddings 2048 --train-iters 20 --data-path /home/sd/Megatron-DeepSpeed/data/qwen_data_text_document --data-cache-path /dev/shm/megatron_common_qwen3b_20260419/index-cache --data-impl mmap --tokenizer-type HFTokenizer --tokenizer-model /home/sd/Megatron-DeepSpeed/.context/qwen25_tokenizer_flat --split 98,2,0 --distributed-backend nccl --lr 1e-5 --lr-decay-style cosine --min-lr 1e-6 --weight-decay 0.01 --clip-grad 1.0 --lr-warmup-iters 19 --optimizer adam --cpu-optimizer --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --log-interval 1 --save-interval 0 --eval-interval 100 --eval-iters 0 --no-query-key-layer-scaling --attention-dropout 0 --hidden-dropout 0 --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear --no-position-embedding --no-masked-softmax-fusion --no-bias-gelu-fusion --no-bias-dropout-fusion --recompute-granularity full --recompute-method uniform --deepspeed-activation-checkpointing --zero-stage=1 --deepspeed_config=/home/sd/Megatron-DeepSpeed/experiments/ib_dual4_common_qwen3b_tp1pp2dp4_20260419_baseline_20260419_162337_v100x16-1/ds_config.json --deepspeed --experiment-run-id ib_dual4_common_qwen3b_tp1pp2dp4_20260419_baseline_20260419_162337_v100x16-1 --experiment-name ib_dual4_common_qwen3b_tp1pp2dp4_20260419 --experiment-root-dir /home/sd/Megatron-DeepSpeed/experiments --bf16
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
