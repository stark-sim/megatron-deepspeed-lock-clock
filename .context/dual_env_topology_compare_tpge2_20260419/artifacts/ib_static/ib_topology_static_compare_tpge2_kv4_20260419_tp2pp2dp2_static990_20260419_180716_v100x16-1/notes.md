# Experiment ib_topology_static_compare_tpge2_kv4_20260419_tp2pp2dp2_static990_20260419_180716_v100x16-1

## Metadata
- experiment_name: `ib_topology_static_compare_tpge2_kv4_20260419`
- status: `initialized`
- run_dir: `/home/sd/Megatron-DeepSpeed/experiments/ib_topology_static_compare_tpge2_kv4_20260419_tp2pp2dp2_static990_20260419_180716_v100x16-1`
- command_sha1: `0e8a1be917038de428d4f0478cfe2ae481075866`

## Command
```bash
/usr/bin/python3 pretrain_gpt.py --local_rank=0 --tensor-model-parallel-size 2 --pipeline-model-parallel-size 2 --num-layers 36 --hidden-size 2048 --ffn-hidden-size 11008 --num-attention-heads 16 --num-key-value-heads 4 --micro-batch-size 1 --global-batch-size 4 --seq-length 2048 --max-position-embeddings 2048 --train-iters 20 --data-path /home/sd/Megatron-DeepSpeed/data/qwen_data_text_document --data-cache-path /home/sd/Megatron-DeepSpeed/data/index-cache --data-impl mmap --tokenizer-type HFTokenizer --tokenizer-model /home/sd/Megatron-DeepSpeed/.context/qwen25_tokenizer_flat --split 98,2,0 --distributed-backend nccl --lr 1e-5 --lr-decay-style cosine --min-lr 1e-6 --weight-decay 0.01 --clip-grad 1.0 --lr-warmup-iters 19 --optimizer adam --cpu-optimizer --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --log-interval 1 --save-interval 0 --eval-interval 100 --eval-iters 0 --no-query-key-layer-scaling --attention-dropout 0 --hidden-dropout 0 --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear --no-position-embedding --no-masked-softmax-fusion --no-bias-gelu-fusion --no-bias-dropout-fusion --recompute-granularity full --recompute-method uniform --deepspeed-activation-checkpointing --zero-stage=1 --deepspeed_config=/home/sd/Megatron-DeepSpeed/experiments/ib_topology_static_compare_tpge2_kv4_20260419_tp2pp2dp2_static990_20260419_180716_v100x16-1/ds_config.json --deepspeed --experiment-run-id ib_topology_static_compare_tpge2_kv4_20260419_tp2pp2dp2_static990_20260419_180716_v100x16-1 --experiment-name ib_topology_static_compare_tpge2_kv4_20260419 --experiment-root-dir /home/sd/Megatron-DeepSpeed/experiments --bf16
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
