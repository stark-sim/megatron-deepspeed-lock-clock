# Experiment power_static1597_20260311_20

## Metadata
- experiment_name: `v100_power_static_1597`
- status: `initialized`
- run_dir: `/home/sd/Megatron-DeepSpeed/experiments/power_static1597_20260311_20`
- command_sha1: `6519a2f8842f31ebdb5976750d2211df3fea1b37`

## Command
```bash
deepspeed --num_gpus 16 pretrain_gpt.py --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --num-layers 28 --hidden-size 3584 --ffn-hidden-size 18944 --num-attention-heads 28 --num-key-value-heads 4 --micro-batch-size 1 --global-batch-size 16 --seq-length 2048 --max-position-embeddings 2048 --train-iters 20 --save /home/sd/Megatron-DeepSpeed/checkpoints/v100_power_static_1597/power_static1597_20260311_20 --data-path /home/sd/Megatron-DeepSpeed/data/chinese_wiki_megatron_text_document --data-impl mmap --tokenizer-type HFTokenizer --tokenizer-model /home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9 --split 98\,2\,0 --distributed-backend nccl --lr 1e-5 --lr-decay-style cosine --min-lr 1e-6 --weight-decay 0.01 --clip-grad 1.0 --lr-warmup-iters 5 --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --log-interval 1 --save-interval 0 --eval-interval 21 --eval-iters 0 --no-query-key-layer-scaling --attention-dropout 0 --hidden-dropout 0 --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear --no-position-embedding --no-masked-softmax-fusion --no-bias-gelu-fusion --no-bias-dropout-fusion --recompute-granularity full --recompute-method uniform --deepspeed-activation-checkpointing --zero-stage=1 --deepspeed_config=/home/sd/Megatron-DeepSpeed/experiments/power_static1597_20260311_20/ds_config.json --deepspeed --experiment-run-id power_static1597_20260311_20 --experiment-name v100_power_static_1597 --experiment-root-dir /home/sd/Megatron-DeepSpeed/experiments --bf16 
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
