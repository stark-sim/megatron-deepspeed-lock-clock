# Megatron-DeepSpeed 锁频训练能效实验项目

本仓库以 `Megatron-DeepSpeed` 作为大模型分布式训练的 baseline 框架，在此基础上实现并验证面向训练能效优化的一套实验体系：固定频点策略、Zeus 功率统计、统一启动与预检流程，以及用于辅助选频的**网络感知型预测层**。

> 📄 **论文工作**: 当前正在进行中的研究论文位于 [`.context/paper/`](.context/paper/)，主题：动态网络感知的跨节点性能预测。
> 
> 📊 **实验记录**: 详细进展见 [`memory-bank/progress.md`](memory-bank/progress.md)，技术决策见 [`memory-bank/techContext.md`](memory-bank/techContext.md)。

---

## 1. 核心创新点

### 1.1 动态网络感知预测

传统分布式训练预测器使用**固定的跨节点通信惩罚系数**，在高速网络环境（如 InfiniBand）下会出现 **98.5%** 的预测误差。

**我们的解决方案**:
- 运行轻量级 NCCL 基准测试测量实际带宽
- 根据网络速度动态调整惩罚系数（IB 环境降低 1700 倍）
- 预测误差从 98.5% 降至 **11.48%**

### 1.2 跨网络可移植性

模型支持多种网络环境：
- **InfiniBand** (100+ Gbps) - 高性能计算
- **RoCE** (25-100 Gbps) - 企业数据中心
- **Ethernet** (1-10 Gbps) - 通用集群
- **Tailscale/VPN** (<1 Gbps) - 远程访问

---

## 2. 项目结构

```
.
├── analysis/freq_model/       # 预测模型核心代码
│   ├── cross_node.py         # 网络感知惩罚模型
│   ├── calibrate.py          # 校准流程
│   └── model.py              # 性能预测模型
├── scripts/
│   ├── run_experiment.sh     # 统一实验启动器
│   └── preflight_check.sh    # 预检脚本
├── .context/
│   ├── paper/                # 📄 论文 LaTeX 源文件
│   │   ├── main.tex
│   │   ├── sections/         # 各章节
│   │   └── figures/          # 图表
│   ├── torch_nccl_comm_bench.py  # 网络基准测试
│   └── run_comm_bench.sh     # 多节点基准启动
├── memory-bank/              # 📊 项目记忆库
│   ├── progress.md           # 进展记录
│   ├── techContext.md        # 技术决策
│   ├── multi_topology_roadmap.md  # 扩展路线图
│   └── ...
└── experiments/              # 实验数据
```

---

## 3. 快速开始

### 3.1 网络基准测试

```bash
# 单节点测试
cd .context
python torch_nccl_comm_bench.py --sizes-mb 1 4 16 64 256

# 多节点测试 (2x4)
bash run_comm_bench.sh 2x4
```

### 3.2 运行实验

```bash
# 预检
bash scripts/preflight_check.sh --verbose

# 定频实验
EXPERIMENT_MODE=static \
STATIC_CLOCK_MHZ=1080 \
bash scripts/run_experiment.sh
```

### 3.3 预测与校准

```python
from analysis.freq_model.calibrate import calibrate_frequency_model

# 使用网络基准结果进行校准
result = calibrate_frequency_model(
    samples, hardware, features,
    network_bench_result=bench_data  # 动态网络感知
)
```

---

## 4. 当前研究成果

### 4.1 预测准确性

| 环境 | 时间 MAPE | 功率 MAPE | 能量 MAPE |
|------|-----------|-----------|-----------|
| InfiniBand (2x4→2x8) | **11.48%** | 3.28% | 7.86% |
| Ethernet (1x4→2x4) | **5.16%** | 12.38% | 10.42% |

### 4.2 能效收益

| 拓扑 | Baseline | 优化频点 | 功率下降 |
|------|----------|----------|----------|
| TP=1, PP=4, DP=4 | 3170.6 W | 1252-1267 MHz | **24-25%** |
| TP=2, PP=2, DP=4 | 3686.1 W | 1072-1125 MHz | **33-36%** |

---

## 5. 扩展路线图

### Phase 1: 网络层 (当前 - Q2 2025)
- ✅ InfiniBand 验证完成
- 🔄 RoCE 验证进行中 (sd-1, sd-2)
- 📝 Ethernet 待验证

### Phase 2: 拓扑层 (Q2-Q3 2025)
- TP/PP/DP 组合变化
- 2x4 / 2x8 / 2x16 / 4x8 规模扩展

### Phase 3: 硬件层 (Q3-Q4 2025)
- NVIDIA A100
- NVIDIA H100

详见 [`memory-bank/multi_topology_roadmap.md`](memory-bank/multi_topology_roadmap.md)

---

## 6. 技术决策记录

关键设计决策：
- **网络检测**: 实测带宽 >50 Gbps 判定为高速网络
- **惩罚系数**: 高速网络 5e-13 s/byte，标准网络 8.41e-10 s/byte
- **阈值选择**: 50 Gbps 平衡了准确性和鲁棒性

详见 [`memory-bank/systemPatterns.md`](memory-bank/systemPatterns.md)

---

## 7. 贡献与引用

如果本项目对您有帮助，请引用：

```bibtex
@inproceedings{network2025aware,
  title={Dynamic Network-Aware Cross-Node Performance Prediction for Distributed Deep Learning},
  author={[Authors]},
  booktitle={[Conference]},
  year={2025}
}
```

---

## 8. 许可证

本项目基于 Megatron-DeepSpeed 进行修改，遵循原项目的许可证。
