# GPU 频率调节节能指南

## 概述

在 NCCL 通信期间动态降低 GPU 核心频率以节省能耗。经测试，最优配置可实现 **-3.9% 能耗** 和 **-2.4% 训练时间** 的改善。

## 最优配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `--comm-low-freq` | 1200 MHz | 通信期间的低频率 |
| `--comm-min-elements` | 300M | 触发调频的最小元素数 |
| `--comm-high-freq` | 1597 MHz | 计算期间的高频率（默认最高） |

## 单节点训练启动方式

### 环境要求

- **Root 权限**：NVML 调频需要 root/sudo
- **pynvml**：`pip install pynvml`
- **Zeus 监控**（可选）：`pip install zeus-ml`

### Qwen2.5-7B 训练（TP=4, 16 GPUs）

```bash
# 清理之前的 checkpoint
sudo rm -rf checkpoints/qwen7b_tp4/*

# 重置 GPU 频率
sudo python3 -c "
import pynvml
pynvml.nvmlInit()
for i in range(16):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    pynvml.nvmlDeviceResetGpuLockedClocks(h)
    pynvml.nvmlDeviceResetApplicationsClocks(h)
pynvml.nvmlShutdown()
"

# 启动训练（需要 sudo）
sudo -E bash -c "
export PYTHONPATH=/home/sd/.local/lib/python3.10/site-packages:/home/sd/Megatron-DeepSpeed:\$PYTHONPATH
export PATH=/home/sd/.local/bin:\$PATH
cd /home/sd/Megatron-DeepSpeed
bash scripts/pretrain_qwen7b_tp4_freq.sh
"
```

### 关键启动脚本参数

```bash
/home/sd/.local/bin/torchrun \
    --nproc_per_node 16 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 29500 \
    pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    # ... 模型参数 ...
    --enable-comm-freq-scaling \
    --comm-low-freq 1200 \
    --comm-min-elements 300000000 \
    --deepspeed \
    --deepspeed_config=scripts/ds_config_7b_tp4.json
```

## 测试结果对比

| 配置 | 能耗 | 功率 | 时间 | vs 对照组 |
|------|------|------|------|----------|
| **1200MHz (最优)** | 983 Wh | 4009 W | 883s | **-3.9% 能耗, -2.4% 时间** |
| 1000MHz | 993 Wh | 3794 W | 942s | -2.9% 能耗, +4.2% 时间 |
| 对照组 (无调频) | 1023 Wh | 4072 W | 904s | - |

## 实现原理

1. **Monkey-patch `torch.distributed`**：自动包装 `all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`
2. **阈值过滤**：只对 >300M 元素的大通信降频，小通信跳过
3. **并行调频**：使用 `ThreadPoolExecutor` 并行设置 16 GPU 频率，减少切换开销

## 文件说明

- `megatron/gpu_freq_manager.py`：GPU 频率管理器核心实现
- `megatron/power_monitor.py`：功耗监控（Zeus）
- `scripts/pretrain_qwen7b_tp4_freq.sh`：7B 模型训练脚本
- `scripts/ds_config_7b_tp4.json`：DeepSpeed 配置

## 调试选项

```bash
# Dry-run 模式：保留逻辑但不执行调频，用于测量开销
--comm-freq-dry-run
```

## 注意事项

1. 频率过低（如 500-1000MHz）会显著降低 NCCL 性能
2. 1200MHz 是 V100 上的最佳平衡点，几乎不影响 NCCL 性能
3. 每个节点只需 local_rank=0 进程管理所有 GPU
