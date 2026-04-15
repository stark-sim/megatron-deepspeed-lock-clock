# Megatron-DeepSpeed 锁频训练能效实验项目

本仓库以 `Megatron-DeepSpeed` 作为大模型分布式训练的 baseline 框架，在此基础上实现并验证面向训练能效优化的一套实验体系：固定频点策略、Zeus 功率统计、统一启动与预检流程，以及用于辅助选频的预测层。

当前 repo 首页只保留与本项目直接相关的内容，不再展开上游 README 的长篇介绍。

## 1. Baseline 框架是什么

`Megatron-DeepSpeed` 是本项目的训练底座，用于承载：

- GPT 类模型训练
- `TP / PP / DP` 分布式并行
- DeepSpeed ZeRO
- 单机与多机启动

在本项目里，`baseline` 的含义非常明确：

- 不额外锁定 GPU 频率
- 保持 Megatron-DeepSpeed 的默认运行方式
- 作为后续 `static` 定频实验的公平对照组

## 2. 本项目新增了什么

相对于“直接跑 Megatron-DeepSpeed baseline”，本项目补充了 4 类能力：

### 2.1 定频对照实验

- `EXPERIMENT_MODE=baseline`：默认 GPU 行为
- `EXPERIMENT_MODE=static`：固定到指定 `STATIC_CLOCK_MHZ`

核心目标不是“单纯降频”，而是：

- 在相同 workload 和相同拓扑下
- 找到一个不会明显拖慢训练时间的固定频点
- 尽量降低平均功率和总能耗

### 2.2 Zeus 统一功率统计

训练过程中统一使用 Zeus 记录关键指标，包括：

- 总训练时间
- 平均功率
- 总能耗

这样 baseline 与 static 的对比不再依赖零散日志，而是落到统一的结构化工件中。

### 2.3 预测层

预测层的定位是：

- 根据已有 source 曲线和当前拓扑
- 辅助缩小值得验证的频点范围
- 帮助用户更快找到“合适的固定频点”

它的作用不是直接替代真实实验，也不是把 benchmark 时间直接当成最终罚时。

### 2.4 统一 launcher / preflight / artifact

本项目围绕 `scripts/run_experiment.sh` 建立统一实验流程，补充了：

- 启动前预检
- 多机 hostfile 快照
- topology 元数据
- `run.json` / `events.jsonl` / `preflight.json`
- baseline 与 static 的统一入口

## 3. 当前可支撑的结论

当前最稳妥的 headline 是：

- 在相同 workload 与拓扑下，将 GPU 固定在合适频点，可以在不显著增加训练时间的前提下，降低约 `20%` 量级的平均功率
- 在已经完成的 V100 双机实验中，部分拓扑可达到 `25%` 到 `35%+` 的平均功率下降
- 预测层的价值是帮助更快找到合适频点，而不是直接“制造”节能收益

代表性历史结果如下。

| 拓扑 | baseline | static | 结论 |
| --- | --- | --- | --- |
| `TP=1, PP=4, DP=4` | `1316.7 s / 3170.6 W / 4174627.4 J` | `1252-1267 MHz` 下约 `1286-1300 s / 2367.9-2407.7 W / 3067834.2-3097007.8 J` | 时长基本不变甚至略优，平均功率下降约 `24%-25%` |
| `TP=2, PP=2, DP=4` | `1030.6 s / 3686.1 W / 3798903.8 J` | `1072-1125 MHz` 下约 `1006-1054 s / 2349.8-2477.3 W / 2471497.2-2492545.3 J` | 时长变化约 `-2.4%` 到 `+2.2%`，平均功率下降约 `33%-36%` |

这些数字当前主要用于说明：

- `baseline` 不是能效最优点
- 合适的 `static` 频点确实存在
- 值得做“先预测、再少量验证”的工作流

## 4. 启动条件

运行本项目前，至少需要满足下面这些条件。

| 条件 | 说明 |
| --- | --- |
| NVIDIA GPU 环境 | 需要 `nvidia-smi` 可用 |
| 训练 launcher 可用 | 当前默认优先 `deepspeed`，也支持 `torchrun` |
| 数据集可用 | `DATASET` 必须对应 `.bin/.idx` 前缀 |
| tokenizer 可用 | `TOKENIZER_PATH` 指向含 `tokenizer.json` 或 `tokenizer_config.json` 的目录 |
| static 模式具备锁频权限 | 需要 NVML + `sudo -n` 可执行 GPU 时钟锁定/复位 |
| 多机启动条件齐全 | 需要 SSH 可达、`HOSTFILE`、`MASTER_ADDR`、`MASTER_PORT` |

当前 launcher 会自动检查：

- GPU 可见性
- `TP / PP / WORLD_SIZE` 是否匹配
- dataset 与 tokenizer 是否存在
- launcher 是否可用
- static 模式下锁频是否受支持

## 5. 启动流程

推荐流程固定为 4 步：

1. 先做预检
2. 跑 baseline
3. 跑 static 对照点或小范围 sweep
4. 用 Zeus 指标做统一比较

### 5.1 先做预检

最简单的演示入口：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VALIDATE_ONLY=1 \
bash 汇报总结_20260415/脚本/运行1p5b演示.sh
```

更通用的 canonical launcher 入口：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TP=4
export PP=1
export EXPERIMENT_MODE=baseline
export EXPERIMENT_NAME=qwen7b_validate
export VALIDATE_ONLY=1

bash scripts/run_experiment.sh
```

如果是多机环境，还建议单独用：

```bash
bash scripts/preflight_check.sh \
  --node-list "node1,node2" \
  --gpu-indices "0,1,2,3" \
  --verbose
```

### 5.2 跑 baseline

```bash
CUDA_VISIBLE_DEVICES=0,1 \
EXPERIMENT_MODE=baseline \
TRAIN_STEPS=20 \
bash 汇报总结_20260415/脚本/运行1p5b演示.sh
```

或使用通用入口：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TP=4
export PP=1
export EXPERIMENT_MODE=baseline
export EXPERIMENT_NAME=qwen7b_baseline
export TRAIN_STEPS=20

bash scripts/run_experiment.sh
```

### 5.3 跑 static 定频对照

```bash
CUDA_VISIBLE_DEVICES=0,1 \
EXPERIMENT_MODE=static \
STATIC_CLOCK_MHZ=1080 \
TRAIN_STEPS=20 \
bash 汇报总结_20260415/脚本/运行1p5b演示.sh
```

或使用通用入口：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TP=4
export PP=1
export EXPERIMENT_MODE=static
export STATIC_CLOCK_MHZ=1080
export EXPERIMENT_NAME=qwen7b_static1080
export TRAIN_STEPS=20

bash scripts/run_experiment.sh
```

### 5.4 做 baseline vs static 对比

如果希望直接演示一组 baseline + 多个 static：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
STATIC_FREQS="1080 1200 1320" \
bash 汇报总结_20260415/脚本/批量运行1p5b对比.sh
```

如果只想把已完成 run 的 Zeus 指标汇总成表：

```bash
python3 汇报总结_20260415/脚本/对比Zeus结果.py \
  experiments/<baseline_run_id> \
  experiments/<static_run_id_1> \
  experiments/<static_run_id_2>
```

## 6. 运行产物怎么看

每次实验都会落到 `experiments/<run_id>/`，常用文件包括：

- `run.json`：完整运行元数据
- `events.jsonl`：训练阶段事件与 Zeus 区间统计
- `preflight.json`：本地/远端预检结果
- `topology.json`：当前解析到的拓扑
- `hostfile_snapshot.json`：多机 hostfile 快照
- `command.sh`：实际启动命令
- `logs/<run_id>.log`：launcher 输出

做功耗对比时，最常用的是：

- `run.json` 中的 Zeus 汇总
- `events.jsonl` 中的 interval/finalized 记录

## 7. 推荐的项目使用方式

对实际使用者，推荐按下面的顺序使用本项目：

1. 用 `baseline` 跑通目标模型和目标拓扑
2. 用 `static` 做少量频点对照，确认存在收益区间
3. 用预测层缩小频点搜索范围
4. 再用真实运行验证最终候选频点

也就是说，本项目的核心价值不是替代 Megatron-DeepSpeed，而是围绕它补齐：

- baseline 对照口径
- 定频实验能力
- Zeus 统一观测
- 预检与工件化流程
- 预测辅助选频

## 8. 关键入口

- `scripts/run_experiment.sh`：统一 launcher
- `scripts/preflight_check.sh`：预检脚本
- `scripts/predict_freq_sweet_spot.py`：频点预测入口
- `docs/experiment-tracking.md`：实验记录与工件说明
- `汇报总结_20260415/`：中文汇报、demo 与对比脚本

如果你只想快速展示完整流程，直接从 `汇报总结_20260415/脚本/运行1p5b演示.sh` 和 `汇报总结_20260415/脚本/批量运行1p5b对比.sh` 开始即可。
