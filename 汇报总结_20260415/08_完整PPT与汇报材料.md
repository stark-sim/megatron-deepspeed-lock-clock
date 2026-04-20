# 完整 PPT 与汇报材料

## 文件位置

| 文件 | 说明 | 格式 |
|------|------|------|
| `GPU能效优化_20%+功耗节省方案.pptx` | **PowerPoint 演示文稿（主要交付物）** | PPT |
| `generate_ppt.py` | Python 生成脚本 | Python |

## PPT 概览

这是一份更偏学术汇报风格、并已强化为图表版的 PowerPoint 演示文稿，重点不再是直接放大“20%+”成果，而是先建立证据链，再给出 headline。

### 核心信息

| 项目 | 内容 |
|------|------|
| **标题** | 分布式训练 GPU 能效优化：基于受控锁频、Zeus 统计与网络感知预测的实验研究 |
| **页数** | 13 页 |
| **语言** | 中文 |
| **格式** | PowerPoint (.pptx) |
| **风格** | 学术汇报 / 组会报告 / 中期答辩风格 |

## 当前内容结构

### 第 1 页：标题页

- 项目标题
- 副标题
- 汇报重点提示

### 第 2 页：研究问题与汇报主线

- baseline 为什么值得比较
- 为什么不能一开始只摆“20%+ 节能”
- 先证据、后结论、再 predictor

### 第 3 页：公平对比口径

- baseline vs static 的唯一区别
- 需要保持一致的条件
- Zeus 统一观测口径

### 第 4 页：系统与方法

- 实验层
- 观测层
- 预测层
- continuous alpha 思路

### 第 5 页：实验环境与证据来源

- artifact-backed local results
- memory-bank preserved summaries
- 避免过强表述

### 第 6 页：案例 A 对照结果

- `TP=1, PP=4, DP=4`
- baseline 锚点 + `1252/1260/1267 MHz` 相对 baseline 柱状图
- 右侧保留精简 `ΔTime / ΔPower / ΔEnergy` 对照表

### 第 7 页：案例 B 对照结果

- `TP=2, PP=2, DP=4`
- baseline 锚点 + `1072/1080/1087/1125 MHz` 相对 baseline 柱状图
- 当前最强的受控节能证据

### 第 8 页：受控对照结论

- 汇总 runtime / power / energy 相对 baseline 的变化范围
- 新增 `runtime delta vs power delta` trade-off 象限图
- 给出可以稳妥写出的 headline
- 给出当前不能直接说死的内容

### 第 9 页：为什么还需要 predictor

- baseline/static 负责给真实证据
- predictor 负责减少 sweep 成本
- 明确 predictor 的定位是 `frequency-selection assistant`

### 第 10 页：IB formal replay

- aggregate metrics：`11.48% / 3.28% / 7.86%`
- 逐频点 `observed vs predicted` 时间 / 功率图表
- 解释剩余误差主要在 runtime

### 第 11 页：Ethernet formal replay

- aggregate metrics：`5.16% / 12.38% / 10.42%`
- 逐频点 `observed vs predicted` 时间图 + APE 图
- 强调 topology 与 IB 不同
- 说明更适合支撑“predictor 可迁移到慢网络”

### 第 12 页：结论与边界

- 这份汇报真正证明了什么
- 当前仍需避免的过强表述
- 下一步工作

### 第 13 页：Q&A

- 谢谢
- Q&A

## 当前最重要的数据支撑

### Baseline vs Static 受控对照

| 拓扑 | runtime 相对 baseline | avg power 相对 baseline | energy 相对 baseline |
|------|------------------------|-------------------------|----------------------|
| `TP=1, PP=4, DP=4` | `-2.3%` 到 `-1.3%` | `-25.3%` 到 `-24.1%` | `-26.5%` 到 `-25.8%` |
| `TP=2, PP=2, DP=4` | `-2.4%` 到 `+2.2%` | `-36.3%` 到 `-32.8%` | `-34.9%` 到 `-34.4%` |

这两组结果是当前支撑“合适固定频点可以显著降功耗，而 runtime 基本不变”的核心证据。

### Predictor formal replay

| 环境 | 时间 MAPE | 功率 MAPE | 能耗 MAPE | 说明 |
|------|-----------|-----------|-----------|------|
| IB `2x4 -> 2x8` | `11.48%` | `3.28%` | `7.86%` | 当前 predictor 的主要 formal 精度证据 |
| Ethernet `1x4 -> 2x4` | `5.16%` | `12.38%` | `10.42%` | topology 不同，更适合支撑可迁移性而非直接横向排名 |

## 当前风格特点

- 偏学术汇报，不偏宣传页
- 先给实验口径，再给数据图表，再给结论
- 明确区分受控节能证据与 predictor 结果
- 明确写出当前边界与不可过度宣称的内容
- 关键结果页已改为柱状图 / trade-off 图 / 观测-预测对照图

## 如何使用

### 直接使用
用 Microsoft PowerPoint 或 WPS 打开：
```
GPU能效优化_20%+功耗节省方案.pptx
```

### 重新生成
如果需要修改内容，编辑 `generate_ppt.py` 后运行：
```bash
cd 汇报总结_20260415
/Users/stark_sim/miniconda3/bin/python3 generate_ppt.py
```

**依赖**：
```bash
pip3 install python-pptx
```

## 与现有材料的关系

- 本 PPT 继续复用 `01_实验口径与主线.md` 到 `07_实现说明.md` 中已经沉淀的中文叙事
- 受控对照数据主要来自 `05_证据清单.md`
- predictor formal 指标主要来自 `.context/paper/experimental_data.md`
- `02_PPT提纲.md` 与 `03_PPT页面文案.md` 现已按新的学术化结构同步更新

## 文件结构

```
汇报总结_20260415/
├── GPU能效优化_20%+功耗节省方案.pptx    # PowerPoint 文件
├── generate_ppt.py                       # Python 生成脚本
└── 08_完整PPT与汇报材料.md               # 本说明文件
```

---

**制作**：Megatron-DeepSpeed 能效实验项目团队  
**工具**：Python + python-pptx  
**日期**：2026-04-16
