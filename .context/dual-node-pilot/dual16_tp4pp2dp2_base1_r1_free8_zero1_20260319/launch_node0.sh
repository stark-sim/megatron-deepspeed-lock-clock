#!/bin/bash
set -euo pipefail
export PATH=/home/sd/.local/bin:/usr/local/cuda/bin:/opt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH=/home/sd/.local/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/home/sd/.local/lib/python3.10/site-packages/torch/lib:/home/sd/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/sd/.local/lib/python3.10/site-packages/nvidia/cublas/lib:/home/sd/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/sd/.local/lib/python3.10/site-packages/numpy.libs:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export OMP_NUM_THREADS=1
export MASTER_ADDR=100.64.0.90
export MASTER_PORT=29523
export NNODES=2
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=tailscale0
export GLOO_SOCKET_IFNAME=tailscale0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
cd /home/sd/Megatron-DeepSpeed
OUT=/home/sd/Megatron-DeepSpeed/.context/dual-node-pilot/dual16_tp4pp2dp2_base1_r1_free8_zero1_20260319
cat > "$OUT/ds_config.json" <<JSON
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 1,
  "steps_per_print": 1,
  "zero_optimization": {"stage": 1},
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0,
  "prescale_gradients": false,
  "wall_clock_breakdown": false
}
JSON
/home/sd/.local/bin/torchrun --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" pretrain_gpt.py --tensor-model-parallel-size 4 --pipeline-model-parallel-size 2 --num-layers 28 --hidden-size 3584 --ffn-hidden-size 18944 --num-attention-heads 28 --num-key-value-heads 4 --micro-batch-size 1 --global-batch-size 16 --seq-length 2048 --max-position-embeddings 2048 --train-iters 1 --data-path /home/sd/Megatron-DeepSpeed/data/chinese_wiki_megatron_text_document --data-impl mmap --tokenizer-type HFTokenizer --tokenizer-model /home/sd/Megatron-DeepSpeed/.context/qwen25_tokenizer_flat --split 98,2,0 --distributed-backend nccl --lr 1e-5 --lr-decay-style cosine --min-lr 1e-6 --weight-decay 0.01 --clip-grad 1.0 --lr-warmup-iters 0 --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --log-interval 1 --save-interval 0 --eval-interval 2 --eval-iters 0 --bf16 --no-query-key-layer-scaling --attention-dropout 0 --hidden-dropout 0 --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear --no-position-embedding --no-masked-softmax-fusion --no-bias-gelu-fusion --no-bias-dropout-fusion --recompute-granularity full --recompute-method uniform --deepspeed-activation-checkpointing --zero-stage=1 --deepspeed_config="$OUT/ds_config.json" --deepspeed 2>&1 | tee "$OUT/node0.log"
