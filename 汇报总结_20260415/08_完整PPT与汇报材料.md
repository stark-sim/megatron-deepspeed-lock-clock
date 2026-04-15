# 完整 PPT 与汇报材料

## 文件位置

| 文件 | 说明 | 语言 |
|------|------|------|
| `GPU能效优化_20%+功耗节省方案.pdf` | **中文版 PPT（主要版本）** | 中文为主，关键技术概念保留英文 |
| `main_zh.tex` | 中文版 LaTeX 源文件 | 中文 |
| `Makefile` | 编译脚本 | - |

**注意**: 原英文版 PPT (`GPU_Energy_Optimization_20Percent_Savings.pdf`) 已归档，当前主要使用中文版。

## 如何编译中文版

```bash
cd 汇报总结_20260415

# 需要安装 XeLaTeX（支持中文）
make          # 编译中文版
make view     # 在 macOS 上查看
make clean    # 清理辅助文件
```

**依赖**: TeX Live 或 MiKTeX，包含 XeLaTeX 编译器

## PPT 概览

这是一份 16 页的正式汇报材料，采用 LaTeX Beamer 制作，主题风格为暖色调（米色背景 + 深棕色文字）。

### 核心信息

| 项目 | 内容 |
|------|------|
| **标题** | 分布式训练 GPU 能效优化：基于锁频策略的 20%+ 功耗节省方案 |
| **页数** | 16 页 |
| **语言** | 英文（main_en.tex）/ 中文（main.tex） |
| **文件大小** | 181 KB |

## 16 页内容结构

### 第一部分：背景与挑战（2 页）

**第 1 页 - 标题页**
- 项目名称：GPU Energy Optimization for Distributed Training
- 副标题：Lock-Frequency Strategy for 20%+ Power Savings
- 项目归属：Megatron-DeepSpeed Energy Efficiency Project

**第 2 页 - Outline**
- 汇报提纲：Background / Solution / Results / Business Value / Conclusion

**第 3 页 - Energy Challenges in LLM Training**
- 行业痛点：GPT-4 训练耗电 1200 万度，电费占 40-60%
- 传统方案局限：动态频率复杂、默认配置非最优
- 我们的洞察：静态频率策略可实现显著节能

**第 4 页 - Core Technical Challenges**
- 挑战 1：频率-性能非线性关系（收益递减）
- 挑战 2：跨节点通信开销（IB vs VPN）
- 挑战 3：拓扑多样性（TP/PP/DP 组合）

### 第二部分：技术方案（2 页）

**第 5 页 - System Architecture: Three-Layer Stack**
- 实验层：统一启动、预检流程
- 监控层：Zeus 功率统计、实时能耗追踪
- 预测层：网络感知模型、跨拓扑迁移

**第 6 页 - Key Innovation: Dynamic Network-Aware Prediction**
- 传统方案：固定惩罚系数，IB 环境下 98.5% 误差
- 我们的方案：实时带宽测量、动态惩罚调整
- 关键成果：误差降至 11.48%，1700 倍系数优化

### 第三部分：实验结果（3 页）

**第 7 页 - Core Metric: Over 20% Power Savings**
- 大标题：**Average Power Savings: 24-35%**
- 实验配置：V100 GPU、Qwen2.5-7B、IB HDR 网络
- 测试场景：单机 16 卡、双机 16 卡、多拓扑对比

**第 8 页 - Detailed Results: Energy Efficiency by Topology**

| 拓扑 | Baseline 功率 | 优化频点 | 优化后功率 | 节省 |
|------|--------------|----------|-----------|------|
| TP=1, PP=4, DP=4 | 3170.6 W | 1252-1267 MHz | 2407.7 W | **24%** |
| TP=2, PP=2, DP=4 | 3686.1 W | 1072-1125 MHz | 2477.3 W | **33%** |
| TP=4, PP=1, DP=4 | 3613.0 W | 1080-1155 MHz | 2363.6 W | **35%** |

- 关键发现：所有拓扑均实现 20%+ 节省，训练时间变化 ±3% 以内

**第 9 页 - Prediction Accuracy: From 98.5% to 11.48%**
- 修复前（固定系数）：98.5% 误差，严重高估跨节点开销
- 修复后（网络感知）：11.48% 误差，准确识别网络特性
- 技术价值：零样本优化成为可能，实验次数减少 50%+

### 第四部分：应用价值（2 页）

**第 10 页 - Cost Savings Estimation**

以 100 卡集群训练 GPT-3 级别模型为例：
- 训练时长：30 天
- Baseline 功率：300W/卡
- 优化后功率：210W/卡（-30%）
- 节省电能：6,480 kWh
- **节省费用：$972（约 6,500 元）**
- **年化节省：$11,664（约 77,000 元）**

**第 11 页 - Deployment Scenarios & Extensibility**
- 适用场景：企业 AI 训练集群、云 GPU 实例、超算中心、边缘节点
- 部署要求：NVIDIA GPU、NVML 支持、DeepSpeed/PyTorch
- 扩展路线：
  - 网络层：IB → RoCE → Ethernet
  - 拓扑层：2×4 → 2×8 → 2×16 → 4×8
  - 硬件层：V100 → A100 → H100

### 第五部分：总结与展望（2 页）

**第 12 页 - Core Achievements Summary**

三大核心指标：
1. **20%+ Power Savings** - 所有测试拓扑均达标
2. **11.48% Prediction Error** - 网络感知模型高精度
3. **1700× Coefficient Optimization** - 动态适配多网络类型

技术亮点：
- 首创动态网络感知跨节点性能预测框架
- 预测精度飞跃：98.5% → 11.48%
- 支持零样本频率优化，实验成本降低 50%+

**第 13 页 - Next Steps**
- 近期（1-2 月）：RoCE 验证、论文投稿、开源发布
- 中期（3-6 月）：A100/H100 适配、更大规模验证
- 长期（6-12 月）：多租户优化、异构集群支持、云服务集成

### 第六部分：附录（3 页）

**第 14 页 - Thank You**
- Q&A
- 联系方式和项目地址

**第 15 页 - Appendix: Experimental Environment**

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA V100-SXM3-32GB × 16 |
| CPU | Intel Xeon Platinum 8168 @ 2.70GHz |
| 内存 | 1.5 TB DRAM per node |
| 网络 | Mellanox ConnectX-5 (100 Gbps IB) |
| 软件 | CUDA 12.1 / PyTorch 2.1.0 / DeepSpeed 0.12 |
| 模型 | Qwen2.5-7B (7B 参数, 28 层) |

**第 16 页 - Appendix: Prediction Model Formulas**

跨节点时间惩罚公式：
```
T_cross_node(f) = α_pp × V_pp + α_dp × V_dp + α_tp × V_tp
```

惩罚系数选择：
```
α_dp = 5×10^-13  (if B_eff > 50 Gbps, IB)
α_dp = 8.41×10^-10 (otherwise, standard network)
```

## 使用建议

### 不同汇报时长

| 时长 | 重点页面 | 内容重点 |
|------|---------|----------|
| 5 分钟 | 第 7、12 页 | 核心指标 + 成果总结 |
| 15 分钟 | 第 5-9 页 | 技术方案 + 实验结果 |
| 30 分钟 | 完整 16 页 | 包含附录详细公式 |

### 不同观众

- **技术观众**：重点讲解第 5-6 页的创新技术
- **管理层**：重点展示第 7、10、12 页的业务价值
- **投资人**：强调第 11 页的扩展性和市场潜力

## 与现有材料的关系

- 本 PPT 是对 `01_实验口径与主线.md` 到 `07_实现说明.md` 的可视化呈现
- 数据来源于 `05_证据清单.md` 中的核心数字
- 汇报主线与 `02_PPT提纲.md` 保持一致

## 如何修改

如需修改内容，请编辑 `presentation/` 目录下的源文件：
- `main_en.tex` - 英文版
- `main.tex` - 中文版
- 然后运行 `make` 重新编译 PDF

## 说明

这份 PPT 是正式汇报版本，与当前目录内的材料（实验口径、证据清单、讲稿）形成互补：
- 本目录材料：详细论证、口径定义、演示脚本
- PPT：高层概览、可视化展示、对外汇报
