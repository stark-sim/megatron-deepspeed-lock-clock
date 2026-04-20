# Experiment ib_dual16_tp4pp1dp4_formal_1155_20260411_111702_DGX2-1

## Metadata
- experiment_name: `ib_dual16_tp4pp1dp4_formal_1155_20260410`
- status: `initialized`
- run_dir: `/home/sd/Megatron-DeepSpeed/experiments/ib_dual16_tp4pp1dp4_formal_1155_20260411_111702_DGX2-1`
- command_sha1: `c7b0c57a1977d9484f04a8247e1d59c1712a7773`

## Command
```bash
/usr/bin/python3 pretrain_gpt.py --local_rank=0 --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --num-layers 28 --hidden-size 3584 --ffn-hidden-size 18944 --num-attention-heads 28 --num-key-value-heads 4 --micro-batch-size 1 --global-batch-size 16 --seq-length 2048 --max-position-embeddings 2048 --train-iters 20 --data-path /home/sd/Megatron-DeepSpeed/data/chinese_wiki_megatron_text_document --data-impl mmap --tokenizer-type HFTokenizer --tokenizer-model /home/sd/Megatron-DeepSpeed/.context/qwen25_tokenizer_flat --split 98,2,0 --distributed-backend nccl --lr 1e-5 --lr-decay-style cosine --min-lr 1e-6 --weight-decay 0.01 --clip-grad 1.0 --lr-warmup-iters 19 --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --log-interval 1 --save-interval 0 --eval-interval 21 --eval-iters 0 --no-query-key-layer-scaling --attention-dropout 0 --hidden-dropout 0 --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear --no-position-embedding --no-masked-softmax-fusion --no-bias-gelu-fusion --no-bias-dropout-fusion --recompute-granularity full --recompute-method uniform --deepspeed-activation-checkpointing --zero-stage=1 --deepspeed_config=/home/sd/Megatron-DeepSpeed/experiments/ib_dual16_tp4pp1dp4_formal_1155_20260411_111702_DGX2-1/ds_config.json --deepspeed --experiment-run-id ib_dual16_tp4pp1dp4_formal_1155_20260411_111702_DGX2-1 --experiment-name ib_dual16_tp4pp1dp4_formal_1155_20260410 --experiment-root-dir /home/sd/Megatron-DeepSpeed/experiments --bf16
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
