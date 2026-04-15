# Context Directory

本目录存放与具体实验运行相关的上下文文件、论文工作区和临时脚本。

## 目录结构

```
.context/
├── README.md                   # 本文件
├── paper/                      # 📄 论文工作区
│   ├── README.md              # 论文构建说明
│   ├── main.tex               # LaTeX 主文件
│   ├── sections/              # 论文各章节
│   ├── figures/               # 生成的图表
│   └── experimental_data.md   # 实验数据汇总
│
├── torch_nccl_comm_bench.py   # NCCL 网络基准测试
├── run_comm_bench.sh          # 多节点基准启动脚本
│
├── IB_NETWORK_FIX_SUMMARY.md  # IB 网络修复总结
│
├── dual8_*/                   # 2x4 实验数据目录
├── dual16_*/                  # 2x8 实验数据目录
├── dual32_*/                  # 2x16 实验相关
└── anomaly_analysis_*/        # 异常分析数据
```

## 快速命令

### 网络基准测试

```bash
# 2x4 配置 (4 GPUs per node)
bash run_comm_bench.sh 2x4

# 2x8 配置 (8 GPUs per node)
bash run_comm_bench.sh 2x8
```

### 论文构建

```bash
cd paper
make figures    # 生成图表
make quick      # 快速编译
make            # 完整编译
```

## 命名规范

### 实验数据目录
```
dual{total_gpus}_tp{tp}pp{pp}dp{dp}_{description}_{timestamp}_{hostname}/
```

示例: `dual8_tp4pp1dp2_static990_20260401_222055_DGX2-1`

### 网络基准结果
```
comm_bench_{topology}_{timestamp}.json
```

示例: `comm_bench_2x4_20260403_184932.json`

## 重要文件说明

| 文件 | 用途 | 更新频率 |
|------|------|----------|
| `paper/` | 论文工作区 | 每日 |
| `torch_nccl_comm_bench.py` | 网络基准测试 | 稳定 |
| `IB_NETWORK_FIX_SUMMARY.md` | 技术修复记录 | 归档 |
| `experimental_data.md` | 数据汇总 | 每次实验后 |

## 数据归档

- 已完成实验的数据目录不要删除
- 每个目录应包含 `run.json` 或等效的实验记录
- 重要结果同步到 `paper/experimental_data.md`
