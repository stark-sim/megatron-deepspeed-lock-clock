#!/bin/bash
# 数据预处理脚本 - 将JSONL格式转换为Megatron二进制格式
set -ex

BASE_PATH=/home/sd/Megatron-DeepSpeed
INPUT_FILE="/home/sd/Megatron-DeepSpeed/data/chinese_wiki.jsonl"
OUTPUT_PREFIX="/home/sd/Megatron-DeepSpeed/data/chinese_wiki_megatron"
TOKENIZER_PATH=/home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9

cd ${BASE_PATH}

python3 tools/preprocess_data.py \
    --input ${INPUT_FILE} \
    --output-prefix ${OUTPUT_PREFIX} \
    --tokenizer-type HFTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --seq-length 2048 \
    --json-keys text \
    --workers 4 \
    --append-eod
