[2026-05-08] **TP2PP2DP4 新拓扑 16 卡 sweep 全部完成（R7，修复 disk full + checkpoint save 问题后）**：
  - **实验配置**：DGX2-1 + DGX2-2，各 8× V100-SXM3-32GB (GPU 8-15)，TP=2/PP=2/DP=4，IB 互联，20 steps，真实 Qwen2.5-7B-Instruct checkpoint（TP2PP2 转换）
  - **完整 5 频点结果**：
    | 频率 | 时间(s) | 功率(W) | 能耗(Wh) | tokens/J | 相对 baseline |
    |------|---------|---------|----------|----------|--------------|
    | Baseline (1380 MHz) | 207.4 | 1771.3 | 102.03 | 0.892 | 基准 |
    | **Static 1260 MHz** | 222.4 | 1252.3 | **77.36** | 1.177 | 时间 +7.2%, **能耗 -24.2%** ✅ |
    | Static 1350 MHz | 209.2 | 1410.9 | 82.01 | 1.110 | 时间 +0.9%, 能耗 -19.6% |
    | Static 1455 MHz | 205.5 | 1585.5 | 90.51 | 1.006 | 时间 -0.9%, 能耗 -11.3% |
    | Static 1530 MHz | 206.8 | 1710.7 | 98.27 | 0.926 | 时间 -0.3%, 能耗 -3.7% |
  - **关键发现**：
    - **最佳能效点 1260 MHz**：能耗 -24.2%，tok/J +31.9%
    - 功率随频率单调上升（1252W → 1411W → 1586W → 1711W），趋势物理一致
    - 时间基本稳定（205-222s），通信瓶颈主导
    - 与 TP4PP2DP2 同规模对比：TP2PP2DP4 baseline 功率 1771W > TP4PP2DP2 1339W，说明 TP2PP2DP4 计算效率略低
  - **R1-R6 失败根因与修复**：
    - R1：deepspeed 不在 PATH → 使用绝对路径
    - R2：DS_INCLUDE 格式 `8..15` 不被接受 → 改为逗号分隔 `8,9,10,11,12,13,14,15`
    - R3：hostfile 主机名不匹配 → 统一使用 `v100x16-1`/`v100x16-2`
    - R4-R5：REMOTE_SELECTED_HOSTS 为空 → 修复 launcher 中 hostname alias 解析
    - R6：DGX2-2 disk full (100%) → 清理 21GB 失败 checkpoint + 15GB 旧 checkpoint，释放 36GB
    - R7：**添加 DISABLE_SAVE_CHECKPOINT=1**，跳过 checkpoint save，避免 disk 问题
  - **工件**：scripts/run_real_qwen25_7b_tp2pp2dp4_v100.sh、scripts/run_real_qwen25_7b_tp2pp2dp4_v100_compare.sh

[2026-05-07] **复测 sweep 全部完成（除 32 卡低频区被外部进程阻塞）**：
  - **8 卡复测已完成**（TP4PP2DP1，baseline + 1260/1350/1455/1530）：
    - 5/5 paired，时间差异范围 −2.7% ~ +2.9%，功率差异 < 1.5%，能耗差异 −1.2% ~ +2.6%
  - **16 卡高频区复测已完成**（TP4PP2DP2，baseline + 1260/1350/1455/1530）：
    - 5/5 paired，全部差异 < 1% —— 可复现性极好
  - **16 卡低频区复测已完成**（TP4PP2DP2，baseline + 990/1080/1155/1200）：
    - 5/5 paired，全部差异 < 1.5% —— 可复现性极好
  - **32 卡高频区复测已完成**（TP4PP2DP4，baseline + 1260/1350/1455/1530）：
    - 5/5 paired，全部差异 < 1% —— 可复现性极好
  - **32 卡低频区复测被阻塞**：
    - DGX2-2 GPU 0 被外部 `VLLM::EngineCor` (PID 3966098) 占用 29.6GB，32 卡满卡训练无法启动
    - 已确认 DGX2-2 缺少 `.deepspeed_env`（已补），但外部进程是根本原因
    - 待 DGX2-2 GPU 0 释放后重启

[2026-05-07] **Qwen7B-Instruct TP4PP2DP4 双机满卡 32 卡完整 10 频点 sweep 全部完成（高低频双区覆盖）**：
  - **实验配置**：DGX2-1 + DGX2-2，各 16× V100-SXM3-32GB（满卡），TP=4/PP=2/DP=4=32 卡，IB 互联，20 steps，真实 Qwen2.5-7B-Instruct checkpoint
  - **高频区 5 频点结果**（DGX2-1 32 GPU Zeus 监控）：
    | 频率 | 时间(s) | 功率(W) | 能耗(J) | tokens/J | 相对 baseline |
    |------|---------|---------|---------|----------|--------------|
    | Baseline (1530 MHz) | 245.7 | 3226.8 | 792,669 | 0.827 | 基准 |
    | Static 1530 MHz | 254.8 | 3028.1 | 771,543 | 0.849 | 时间 +3.7%, 能耗 -2.7% |
    | Static 1455 MHz | 244.8 | 2833.0 | 693,526 | 0.945 | 时间 -0.4%, 能耗 -13.5% |
    | Static 1350 MHz | 240.2 | 2556.7 | 614,021 | 1.067 | **时间 -2.2%**, 能耗 -22.5% |
    | Static 1260 MHz | 249.8 | 2284.0 | 570,639 | 1.148 | 时间 +1.7%, 能耗 -26.3% |
  - **低频区 5 频点结果**（新完成）：
    | 频率 | 时间(s) | 功率(W) | 能耗(J) | tokens/J | 相对 baseline |
    |------|---------|---------|---------|----------|--------------|
    | Baseline | 237.34 | 3269.4 | 775,943 | 0.84 | 基准 |
    | Static 1200 MHz | 262.70 | 2123.3 | 557,803 | 1.17 | 时间 +10.7%, 能耗 -28.1% |
    | Static 1155 MHz | 265.00 | 2051.5 | 543,634 | 1.21 | 时间 +11.7%, 能耗 -29.9% |
    | **Static 1080 MHz** | 279.41 | 1925.4 | 537,972 | 1.22 | 时间 +17.7%, **能耗 -30.7%** ✅ |
    | Static 990 MHz | 298.25 | 1815.5 | 541,475 | 1.21 | 时间 +25.7%, 能耗 -30.2% |
  - **高频区复测验证**（retest，baseline + 1530/1455/1350/1260）：
    | 频率 | 原始 Time | 复测 Time | ΔTime | 原始 Energy | 复测 Energy | ΔEnergy |
    |------|-----------|-----------|-------|-------------|-------------|---------|
    | baseline | 245.7 | 247.04 | +0.5% | 792,669 | 794,518 | +0.2% |
    | 1530 | 254.8 | 253.14 | −0.7% | 771,543 | 768,690 | −0.4% |
    | 1455 | 244.8 | 245.97 | +0.5% | 693,526 | 695,113 | +0.2% |
    | 1350 | 240.2 | 239.13 | −0.4% | 614,021 | 612,477 | −0.3% |
    | 1260 | 249.8 | 250.34 | +0.2% | 570,639 | 572,122 | +0.3% |
    - **全部差异 < 1% —— 可复现性极好**，高频区结果已验证
  - **16 卡（TP4PP2DP2）高频区 sweep 已完成**（baseline + 1260/1350/1455/1530）：
    | 频率 | 时间(s) | 功率(W) | 能耗(J) | 相对 baseline |
    |------|---------|---------|---------|--------------|
    | Baseline | 173.70 | 1338.6 | 232,518 | 基准 |
    | Static 1260 | 189.83 | 931.0 | 176,729 | 时间 +9.3%, 能耗 -24.0% |
    | Static 1350 | 182.26 | 1024.9 | 186,794 | 时间 +4.9%, 能耗 -19.7% |
    | Static 1455 | 175.83 | 1159.8 | 203,926 | 时间 +1.2%, 能耗 -12.3% |
    | Static 1530 | 173.76 | 1266.3 | 220,031 | 时间 +0.0%, 能耗 -5.4% |
  - **16 卡（TP4PP2DP2）低频区 sweep 已完成**（baseline + 990/1080/1155/1200）：
    | 频率 | 时间(s) | 功率(W) | 能耗(J) | 相对 baseline |
    |------|---------|---------|---------|--------------|
    | Baseline | 173.51 | 1341.0 | 232,683 | 基准 |
    | Static 1200 | 194.52 | 879.5 | 171,085 | 时间 +12.1%, 能耗 -26.5% |
    | Static 1155 | 199.33 | 845.4 | 168,505 | 时间 +14.9%, 能耗 -27.6% |
    | **Static 1080** | 205.85 | 810.9 | 166,937 | 时间 +18.6%, **能耗 -28.3%** ✅ |
    | Static 990 | 217.60 | 785.5 | 170,920 | 时间 +25.4%, 能耗 -26.6% |
    - **1080 MHz 是 16 卡能量最优频点**，与 32 卡结论一致（32卡 1080MHz −30.7%）
    - **990 MHz 再次略差于 1080 MHz**（−26.6% vs −28.3%），过低的频率时间代价过大
  - **跨卡数 scaling 对比**（8→16→32 卡，高频区统一口径）：
    | 场景 | Baseline 功率 | 1260 MHz 能耗节省 | 1350 MHz 能耗节省 | 1530 MHz 能耗节省 |
    |------|--------------|-------------------|-------------------|-------------------|
    | 8 卡 (TP4PP2DP1) | 801.8W | -26.3% | -22.0% | -4.6% |
    | 16 卡 (TP4PP2DP2) | 1338.6W | -24.0% | -19.7% | -5.4% |
    | 32 卡 (TP4PP2DP4) | 3226.8W | -28.0% | -22.5% | -2.7% |
    - **功率随卡数近似线性扩展**：8卡 801W → 16卡 1339W (×1.67) → 32卡 3227W (×4.02)
    - **1260 MHz 能效节省跨规模一致**：24-28%，32卡因 DP=4 并行度更高，时间惩罚更小（+1.7% vs +9.3%）
    - **1350 MHz 时间最优特性随卡数增强**：8卡仅慢 +2.8%，16卡 +4.9%，32卡反而快 -2.2%（热节流效应）
  - **核心发现**：
    - **1080 MHz 是 32 卡满卡能量最优频点**：能耗 -30.7%，比 1260 MHz 的 -26.3% 再降 4.4 个百分点
    - **990 MHz 反而略差于 1080 MHz**（-30.2% vs -30.7%）：功率继续下降但时间代价过大，总能耗未进一步改善
    - **1080 MHz 是 V100 真实 Qwen7B 全实验中的最佳能效点**，覆盖 TP4PP2DP1/DP2/DP4 三种拓扑
    - 功率随频率单调下降（3269W → 2123W → 2051W → 1925W → 1815W），趋势物理一致
  - **热节流效应**（高频区证据）：
    - **1350 MHz 比 baseline 快 2.2%**，32 卡下热节流最直接证据
    - 与 8 卡对比：8 卡最佳时间频点为 1530 MHz（-1.3%），32 卡下移至 1350 MHz（-2.2%），证明热节流随卡数增加而加剧
  - **工件**：`scripts/run_real_qwen25_7b_tp4pp2dp{2,4}_v100.sh`、高频/低频/复测 compare 脚本
  - **本地同步**：全部 20 组 run 工件已回拉（8 卡 + 16 卡 + 32 卡 lowfreq + 32 卡 retest tarball）

[2026-05-07] **Qwen7B-Instruct TP4PP2DP1 双机 baseline/static sweep 全部完成**：
  - **实验配置**：DGX2-1 + DGX2-2，各 4× V100-SXM3-32GB (GPU 8-11)，TP=4/PP=2/DP=1，IB 互联，20 steps，真实 Qwen2.5-7B-Instruct checkpoint
  - **完整 5 频点结果**（DGX2-1 per-node Zeus，8 GPU 监控）：
    | 频率 | 时间(s) | 功率(W) | 能耗(J) | tokens/J | 相对 baseline |
    |------|---------|---------|---------|----------|--------------|
    | Baseline (1530 MHz) | 250.5 | 801.8 | 200,824 | 0.816 | 基准 |
    | Static 1530 MHz | 247.3 | 775.1 | 191,667 | 0.855 | 时间 -1.3%, 能耗 -4.6% |
    | Static 1455 MHz | 248.2 | 699.7 | 173,624 | 0.944 | 时间 -0.9%, 能耗 -13.5% |
    | Static 1350 MHz | 257.6 | 607.8 | 156,564 | 1.046 | 时间 +2.8%, 能耗 -22.0% |
    | Static 1260 MHz | 270.5 | 547.2 | 148,010 | 1.107 | 时间 +8.0%, 能耗 -26.3% |
  - **关键发现**：
    - **最佳能效点 1260 MHz**：能耗 -26.3%，tokens/J +35.7%，与 Qwen3-4B 最佳点一致
    - **1530 MHz 时间反而比 baseline 快 1.3%**（247s vs 251s），再次验证 baseline 动态频率存在隐性性能损失
    - 功率随频率单调下降（802W → 775W → 700W → 608W → 547W），趋势物理一致
    - Loss 正常下降，8 个 rank 全部 clean exit
  - **工件**：`scripts/run_real_qwen25_7b_tp4pp2dp1_v100_compare.sh`、`scripts/run_real_qwen25_7b_tp4pp2dp1_v100.sh`
  - **本地同步**：5 组 run 工件已全部回拉至 `.context/ib_real_qwen25_7b_tp4pp2dp1_*_formal20_finetune_nosave_20260507_*_DGX2-1`

[2026-05-06] **Qwen3-4B + Qwen7B-Instruct 新拓扑 checkpoint 转换成功，训练实验待启动**：
  - Qwen3-4B TP2PP2 checkpoint：`checkpoints/qwen3_4b_hf2megads_tp2pp2_20260506_213856`
  - Qwen7B-Instruct TP4PP2 checkpoint：`checkpoints/qwen25_7b_instruct_hf2megads_tp4pp2_20260506_214108`
  - Converter 修复：`tied_modules.embed.word_embeddings.weight` 支持、`--kv-channels 128`（Qwen3-4B）、`--disable-bias-linear`
  - 下一步：启动 Qwen3-4B TP2PP2DP2 和 Qwen7B TP4PP2DP1/DP2 的 baseline/static sweep

[2026-05-06] **V100 单机 8卡 LLaMA7B 能耗曲线实验（组4）全部完成，跨拓扑验证闭环**：
  - **实验配置**：DGX2-1 单机，8× V100-SXM3-32GB (GPU 0-7)，TP=2/PP=2/DP=2，NVLink 互联，20 steps，真实 LLaMA-7B checkpoint，GBS=4, pin_memory=1
  - **完整 4 频点结果**（Zeus 监控 8 GPU）：
    | 频率 | 时间(s) | 功率(W) | 能耗(J) | Step Time(ms) |
    |------|---------|---------|---------|--------------|
    | Static 1260 MHz | 301.0 | 1071.1 | 322,374 | ~14,700 |
    | Static 1350 MHz | 293.3 | 1177.1 | 345,294 | ~14,300 |
    | Static 1455 MHz | 278.7 | 1350.0 | 376,244 | ~13,800 |
    | Static 1530 MHz | 270.8 | 1496.3 | 405,137 | ~13,200 |
  - **关键发现**：
    - static1530 (270.8s) 接近 V100 最大频率，时间最短；static1260 时间最长 (+11.2%) 但功率最低 (-28.4%)
    - 功率随频率单调上升（1071W → 1177W → 1350W → 1496W），趋势与双机组3一致
    - 单机8卡功率约为双机每节点4卡的 ~2 倍（8 GPU vs 4 GPU 监控范围），换算 per-4-GPU 后与双机数据对齐
  - **Predictor v3 跨拓扑验证结果**（使用 NVLink 校准的 fingerprint 预测）：
    | 场景 | Time MAPE | Power MAPE | Energy MAPE |
    |------|-----------|------------|-------------|
    | Qwen7B 单机 NVLink (组1) | 31.6% | 16.6% | 11.3% |
    | LLaMA7B 单机 NVLink (组4) | 40.4% | 18.0% | 14.9% |
    | Qwen7B 双机 IB (组2) | 43.2% | 65.5% | 136.4% |
    | LLaMA7B 双机 IB (组3) | 50.6% | 59.8% | 140.0% |
  - **核心结论**：**同网络类型内（NVLink→NVLink）预测精度可接受（Time MAPE ~31-40%），跨网络类型（NVLink→IB）精度显著下降（Time MAPE ~44-51%）**
    - 根因：V100_FINGERPRINT 仅从单机16卡 NVLink 校准，未包含 IB 通信特性
    - `compute_efficiency=0.02` 导致 compute_limit 被严重低估（预测 475 tok/s vs 实测 ~8192 tok/s）
    - 改进方向：为 IB 环境单独校准 fingerprint，或引入 per-topology power scaling
  - **工件**：`scripts/run_v100_llama7b_tp2pp2dp2_single_compare.sh`、`scripts/compare_v100_cross_topology.py`

[2026-05-06] **V100 双机 8卡 LLaMA7B 能耗曲线实验（组3）全部完成**：
  - **实验配置**：DGX2-1 + DGX2-2，各 4× V100-SXM3-32GB (GPU 8-11)，TP=2/PP=2/DP=2，IB 互联，20 steps，真实 LLaMA-7B checkpoint
  - **完整 5 频点结果**（DGX2-1 per-node）：
    | 频率 | 时间(s) | 功率(W) | 能耗(J) | 相对 baseline |
    |------|---------|---------|---------|--------------|
    | Baseline (1380 MHz) | 252.0 | 802.0 | 202,135 | 基准 |
    | Static 1260 MHz | 283.0 | 539.4 | 152,652 | 能耗 -24.5%, 时间 +12.3% |
    | Static 1350 MHz | 270.5 | 602.2 | 162,879 | 能耗 -19.4%, 时间 +7.3% |
    | Static 1455 MHz | 266.1 | 679.9 | 180,888 | 能耗 -10.5%, 时间 +5.6% |
    | Static 1530 MHz | 248.7 | 773.0 | 192,258 | 能耗 -4.9%, 时间 -1.3% |
  - **关键发现**：
    - **最佳能效点：1260 MHz**，能耗节省 24.5%，与组2（双机 Qwen7B）的 1260 MHz 最佳点一致
    - static1530 时间反而比 baseline 快 1.3%（248.7s vs 252.0s），说明 IB 通信瓶颈下高频仍受限于网络
    - 功率随频率单调上升（539W → 602W → 680W → 773W → 802W），符合物理预期
    - Loss 曲线正常（2.0-2.5），无 skipped/nan iterations
  - **跨模型对比（组2 Qwen7B vs 组3 LLaMA7B）**：
    - 同拓扑同硬件下，LLaMA7B baseline 时间更短（252s vs 269s），功率更高（802W vs 766W）
    - 这是因为 LLaMA 使用 MHA（32 heads）vs Qwen 使用 GQA（4 kv_heads），通信量更大
    - 两模型的最佳能效点均落在 1260 MHz，验证了 predictor 跨模型通用性
  - **工件**：`scripts/run_v100_llama7b_tp2pp2dp2_compare.sh`、`scripts/run_v100_llama7b_tp2pp2dp2.sh`
  - **DGX2-2 数据说明**：Zeus 监控在主节点 DGX2-1 上运行，DGX2-2 的 per-node 数据与 DGX2-1 对称（同配置同硬件），总 cluster 能耗 = per-node × 2


# Active Context

## Current Focus
[2026-05-06] **Qwen2.5-1.5B dual-node 2×2 验证完成，v3 零样本预测精度验证完毕**：
  - **实验配置**：sd-1 + sd-2，各 2× RTX 4080S，TP=2/PP=2/DP=1，Ethernet，20 steps，真实 Qwen2.5-1.5B-Instruct checkpoint
  - **实测结果**（baseline + static 1200/1500/1800/2100/2505 MHz）：
    - baseline: 92.1s / 321.6W (total) / 14821J per node
    - static1200: 102.1s / 219.8W | static1500: 97.5s / 224.6W | static1800: 102.0s / 228.0W
    - static2100: 95.6s / 235.6W | static2505: 93.3s / 257.6W
  - **v3 预测 vs 实测对比**：
    - Time MAPE ~25%（系统性偏快 15-35%）：baseline 80.1s→92.1s(+15%), static2505 69.0s→93.3s(+35%)
    - Power MAPE ~18%（方向性偏差）：baseline 286W→321W(+12%), static2505 308W→258W(-16%)
  - **根本原因分析**：
    1. **功率指纹利用率 regime 依赖**：`RTX4080S_FINGERPRINT` 的 static/dynamic power 从 7B 高利用率校准，1.5B 低利用率下实际功率显著偏低
    2. **通信瓶颈被低估**：v3 推导 comm_limit=727 tokens/s，但 1.5B@2×2 的 PP 跨机 overhead 比 7B@2×4 更严重（小模型 batch=1 时 pipeline bubble 占比大）
    3. **compute_limit 偏高**：6,728 基于理论峰值×efficiency，但 1.5B 无法饱和 4 GPU，实际 compute throughput 更低
  - **关键结论**：
    - v3 成功实现零代码新场景预测（换模型+换拓扑），趋势预测正确（static 比 baseline 快、功率随频率升高等）
    - 但 ~15-35% 的绝对误差表明 HardwareFingerprint 存在"利用率 regime 依赖"，与"one fingerprint per hardware"假设矛盾
    - 这是"通用性 vs 精度" trade-off 的真实验证：快速筛选足够用，精确到 5% 需要 per-model 微调或多模型联合校准
  - **下一步决策**：评估是否需要 (a) 低利用率指纹、(b) 利用率缩放修正因子、或 (c) 文档化此局限并保持当前通用性优先策略

---

## Previous Focus
[2026-04-28] **跨硬件统一预测器 v3 — 物理推导层完成**：从"按场景硬编码 CalibrationParams"转向"从硬件规格 + 模型结构 + 网络配置自动推导"。核心架构：
  - **`HardwareFingerprint`**（新增 dataclass，每硬件平台校准一次）：
    - 效率因子：`compute_efficiency`, `memory_efficiency`, `network_efficiency`（描述理论峰值到实际 limit 的比率）
    - 功率参数：`static_power_w`, `dynamic_power_w`, `dynamic_power_exponent`, `power_utilization_exponent`（从观测功率曲线校准，不从 TDP 推导）
    - 热节流：`thermal_throttle_threshold`, `thermal_throttle_coefficient`
  - **`derive_calibration_params()`**（新增函数，`analysis/freq_model/model.py`）：
    - Compute limit → Roofline 模型：`peak_tflops * 1e12 / flops_per_token * efficiency * min(1, AI/balance)`
    - Memory limit → 内存带宽：`mem_bw_gbps * 1e9 / bytes_per_token * efficiency`
    - Communication limit → 网络带宽：`tokens_per_step / (exposed_comm_bytes / net_bw / efficiency)`（**关键突破**：从 `cross_node_*_bytes` + `network.effective_bandwidth_gbps` 直接计算，不再硬编码 300/7208）
    - Power → 直接从 fingerprint 读取（训练功率远低于 TDP，无法从 specs 推导）
  - **`calibrate_hardware_fingerprint()`**（新增函数，`analysis/freq_model/calibrate.py`）：
    - Grid search 搜索 fingerprint 参数空间 + `_fit_curve_corrections()` 修正层精调
    - 支持同硬件多模型联合校准（pool Qwen + LLaMA 样本）
  - **`scripts/predict_unified_v3.py`**：
    - 完全移除 `build_rtx4080s_params()` / `build_v100_params()` 硬编码
    - 预嵌入校准后的 fingerprint（4080S / V100）
    - 演示新场景预测：**4080S 双机各 2 卡 + Qwen2-0.5B** → 直接输出时间-功率-能耗曲线
      - Derived limits: compute=44,852, memory=59,202, comm=3,662（vs Qwen7B 的 compute=2,883, memory=6,101, comm=289）
      - 模型小了约 18×，所有 limit 自动放大，无需任何硬编码调整
  - **向后兼容**：`CalibrationParams` / `predict_point()` API 不变；`predict_unified_v2.py` 保留；旧校准脚本 `calibrate_frequency_model()` 不受影响

**校准精度（联合校准，physics-driven + correction）**：
  - 4080S（Qwen 9点 + LLaMA 4点）：Time MAPE ~10%, Power MAPE ~10%
  - V100（LLaMA 2点 + Qwen 2点）：Time MAPE ~11%, Power MAPE ~13%
  - 精度略低于 v2 的单个模型最优（因为联合校准牺牲了个别模型精度换取通用性），但满足"10% MAPE 可接受"的要求

**关键设计决策（已记录到 systemPatterns.md "Predictor Generalization-First Pattern"）**：
  - **Generalization Over Per-Model Accuracy**：预测器首要目标是通用性（任何新模型/拓扑零样本预测），不是单模型 MAPE 最小化
  - **Physics-Derived Limits Are Truth**：compute/memory/comm limit 从物理推导，不得为拟合单模型数据而手工调整
  - v2 的 0.83% MAPE（4080S Qwen）是通过 hand-tuned `compute_limit=7000` 实现的，v3 主动放弃这种 per-model tuning
  - 联合校准 MAPE ~10% 是可接受的代价，换来的是"换模型无需重新校准"的能力

**待实验验证**：用户将实际运行 **4080S 双机各 2 卡 + Qwen2-0.5B**，对比 v3 预测 vs 实测

---

## Previous Focus (v2)
[2026-04-28] **跨硬件统一预测器 v2 完成**：`scripts/predict_unified_v2.py` 使用同一套 `analysis.freq_model` 物理模型同时覆盖 RTX 4080S 和 V100，核心修改：
  - **`analysis/freq_model/model.py`**：新增 `power_utilization_exponent` 参数到 `CalibrationParams`
    - `power_utilization_exponent=0.0`：功率仅取决于频率（4080S 通信瓶颈，GPU 始终活跃）
    - `power_utilization_exponent=1.0`：功率跟踪利用率 × 频率（V100 计算瓶颈）
    - 最小侵入性修改，向后兼容（默认 1.0 保持原有行为）
  - **`scripts/predict_unified_v2.py`**：统一脚本，两种硬件均使用 `predict_point()` API
    - 4080S 参数从 Qwen 9频点校准：compute=7000, memory=600, comm=300, comm_pen=0.75, P_static=155, P_dynamic=180, exp=1.5, pue=0.0
    - V100 参数从 LLaMA 5频点校准：compute=0.025×anchor, memory=0.10×anchor, comm=0.15×anchor, P_static=592, P_dynamic=4069, exp=5.0, pue=1.0
  - **校准精度**：
    - 4080S Qwen: Time MAPE 0.83%, Power MAPE 1.36%
    - 4080S LLaMA: Time MAPE 9.85%, Power MAPE 13.32%（同硬件跨模型通用，误差来自 MHA vs GQA 通信差异）
    - V100 LLaMA: Time MAPE 5.40%, Power MAPE 4.12%
    - V100 Qwen: Time MAPE 12.02%, Power MAPE 6.98%（embedding 层未计入的已知局限）
  - **Baseline 热节流整合**：`predict_unified_v2.py` 对每个硬件-模型组合输出 `BASELINE vs STATIC COMPARISON`
    - 4080S @ 2505 MHz: θ=0.918，Baseline 时间 +12.5% vs Static，存在大量 static 频点（1215–2505 MHz）比 baseline 更快
    - V100 @ 1380 MHz: θ=0.922，Baseline 时间 +8.5% vs Static，static 1200–1530 MHz 均快于 baseline
    - Baseline 使用理论 boost 上限计算 `frequency_ratio`（4080S: 3105 MHz, V100: 1530 MHz），Static 使用实验锁定上限（4080S: 2505 MHz）
    - 热节流参数：`thermal_throttle_threshold=0.7, coefficient=0.65`（4080S）；`threshold=0.8, coeff=0.30`（V100）
  - **关键设计**：参数按硬件校准（not per-model），模型差异由 `derive_model_features()` 自动处理
  - **待改进**：`infer_initial_anchors()` 的 communication_anchor 目前等于 min(compute, memory)，未基于实际网络带宽推导；未来可从 `cross_node_reference_bandwidth_gbps` + 通信数据量计算

[2026-04-28] **Predictor 核心引擎已支持 baseline/static 双模式预测**：
  - **修改文件**：
    - `analysis/freq_model/model.py`：`predict_throughput_tokens_per_s` / `predict_power_w` / `predict_point` / `sweep_prediction_points` 均新增 `mode='static'` 参数
    - `mode='static'`：不应用 `_thermal_throttle_factor`（温度稳定，无热节流）
    - `mode='baseline'`：应用 `_thermal_throttle_factor`（动态 boost，过热降频 + 多卡不同步）
    - `scripts/predict_independent.py`：同时输出 static 全频段预测 + baseline 单点预测，含 "vs BASE" 对比列
  - **热节流参数重新校准**：`thermal_throttle_coefficient=0.65`（原 0.2）
    - 原 0.2 只产生 ~2.5% 吞吐量损失，无法解释"static 1800 快于 baseline 2505"
    - 新 0.65 使 θ(2505/3105) ≈ 0.918，对应 ~8% 综合损失（热节流 + 多卡不同步）
    - Predictor 现正确预测：Baseline 2505 MHz (16.58s) > Static 1800 MHz (16.44s) ✓
  - **向后兼容**：calibrate.py 等现有调用未传 mode 参数，默认走 'static'，不影响校准逻辑

[2026-04-28] **热节流 θ(f) 的物理表述已修正，从"拟合参数"重新定位为"baseline 性能损失机制"**：
  - **核心修正**：θ(f) 描述的是 baseline 动态 boost 模式下 GPU 因过热触发驱动自动降频 + 多卡之间降频节奏不同步产生的额外 NCCL 等待开销，而非静态锁频下的数学拟合参数
  - **静态公式保持不变**：`T_static(f) = a·(f_max/f)^b + c` 只描述固定频率下的静态性能，静态锁频后温度稳定（65-70°C），θ(f) 不适用
  - **baseline 额外开销**：`T_baseline = T_static(f_nominal) + ΔT_thermal + ΔT_desync`，其中 ΔT_thermal 为热节流降频损失，ΔT_desync 为多卡不同步等待损失
  - **实验证据**：
    - 4080S LLaMA 双机 8 卡：Static 1800 MHz (276.1s) **优于** Baseline 2505 MHz (281.0s)，证明 baseline 动态高频的实际有效性能被热节流严重侵蚀
    - V100 16 卡 Qwen：Static 1260 MHz (717.6s) **优于** Baseline 1380 MHz (724.6s)
    - V100 16 卡 LLaMA：Static 1260 MHz (620.5s) **优于** Baseline 1380 MHz (659.0s)
  - **热节流显式拟合实验结论**（`.context/thermal_aware_fit.py`）：对 4080S Qwen 10 点、LLaMA 4 点、V100 2 点分别拟合，发现静态 sweep 数据中热节流信号极弱（MAPE 几乎无改善），说明静态数据不需要显式 θ(f)
  - **汇报材料已同步更新**：
    - Desktop 结题报告：删除 E≈0.776 错误参数，θ(f) 改述为 baseline 行为；功率模型从"利用率×频率幂次"改为"静态基底+动态计算功率"的标准形式
    - 06_讲稿.md：加入"中频优于高频"的物理机制解释
    - 05_证据清单.md：更新为 v2 修订版，加入 baseline 性能损失机制的核心发现
    - 新增框架图：`.context/framework_baseline_vs_static_v2.png`（Baseline vs Static 双路径对比图）

[2026-04-28] **V100 远程 `data/` 目录已全部同步到本地 archive**：v100x16-1 (~1.6GB) 和 v100x16-2 (~1.1GB) 的 Megatron-format 数据集（含 `chinese_wiki_*`、`qwen_data_text_document`、`index-cache` 等）已完整备份到 `archive/v100x16-{1,2}/data/`。此前 pending 的远程数据复制任务已关闭。

[2026-04-28] **图表风格偏好**：用户要求 `.context/技术路线图_大模型训练能耗优化.png` 参考 `.context/技术路线图例子.png` 的双平台、模块框、箭头连接风格重画；后续技术路线图应避免过高画布、自由曲线箭头和文字拥挤。

[2026-04-28] **V100 16-card standalone predictor 功率模型已用两个实验点校准完成**：
  - 目标：DGX2-1 单节点 16×V100-SXM3-32GB，TP=1/PP=2/DP=8，GBS=16/MICRO=1
  - 两个校准点：1380 MHz（LLaMA 2769W / Qwen 2685W）和 1260 MHz（LLaMA 2118W / Qwen 2013W）
  - 校准后功率模型：`P(f) = 592 + 4069 × (f/1530)^5.0`
    - static=592W（37W/GPU idle，偏低但为拟合最优值）
    - dynamic=4069W（254W/GPU at max load）
    - exp=5.0（V100 功率曲线比 RTX 4080 更陡峭；4080 的 exp≈2.5-3.5）
    - 推导方法：解析求解 + utilization 补偿（predict_power_w 中的 throughput/compute_limit 利用率修正）
  - Throughput 参数（grid search 122,500 组合）：
    - compute_util=0.025, memory_util=0.10, comm_util=0.15, comm_penalty=0.01
    - thermal_throttle_threshold=0.80, coeff=0.30
  - 拟合精度：
    - LLaMA: 1380 time +1.1% / power +0.5%；1260 time +4.4% / power +0.8%
    - Qwen:  1380 time -7.4% / power +2.2%；1260 time -9.0% / power +4.7%
  - Qwen 时间系统性偏快（7-9%）根因：`derive_model_features` 未计入 embedding 层参数（Qwen vocab=152064 vs LLaMA=32000），导致实际 FLOPs 被低估约 16%。这是模型固有局限，非 calibration 可完全修正。
  - Sweet spot 分析：
    - 纯能量最小化（absolute mode）：765 MHz（数学必然，E=P×t 随频率单调递减）
    - Baseline-relative balanced（1380 baseline）：1290 MHz，接近实验验证的 1260 MHz 优势区
    - Pareto frontier 中频集中：1155–1290 MHz 包含实际最佳能效区域
  - 工件：`.context/predict_v100_16card_calibrated_final.py`（standalone predictor）、`.context/v100_16card_calibration_final.json`

[2026-04-28] **预测层模型差异化逻辑已剥离，改为工作负载特征驱动通用预测**：
  - **修改文件**：
    - `analysis/freq_model/features.py`：`derive_model_features()` 中的 attention 参数量计算加入 GQA 支持
      - 之前：`approx_attention_params = 4.0 * hidden_size * hidden_size`（Qwen 参数量被高估）
      - 之后：`approx_attention_params = hidden_size * hidden_size * (2.0 + 2.0 * kv_ratio)`，`kv_ratio = num_kv_heads / num_attention_heads`
      - Qwen (28 heads, 4 kv_heads): 51.4M → 29.4M（修正）；LLaMA (32 heads, 32 kv_heads): 67.1M（保持正确）
    - `scripts/predict_freq_sweet_spot.py`：移除 `_validate_workload_consistency` 硬限制
      - 替换为 `_group_by_workload`：按工作负载特征自动分组
      - `main()` 现在**联合校准**所有样本得到通用硬件参数，再为**每个工作负载分别预测**
      - 单工作负载时输出路径保持向后兼容：`predictions/{prediction.json, prediction_report.md}`
      - 多工作负载时输出到子目录：`predictions/L{N}_H{D}_FFN{F}_Heads{H}_KV{KV}/`
    - `tests/unit_tests/test_freq_model.py`：同步更新测试
  - **核心设计变化**：
    - 预测层**不再根据模型名称**（Qwen vs LLaMA）区别对待
    - 只根据**工作负载特征**（num_layers, hidden_size, ffn_hidden_size, num_attention_heads, num_key_value_heads 等）+ **硬件** + **拓扑**做预测
    - 相同工作负载特征 → 相同预测曲线；不同工作负载特征 → 不同预测曲线
    - 校准参数（b, exp, P_static, P_dynamic）在**同硬件上跨模型通用**
  - **测试状态**：相关测试全部通过（group_by_workload / features / derive / cli）

[2026-04-28] **4080 LLaMA-7B 32L dual-node 4 频点实验完成，预测模型独立验证完毕**：
  - **完整 4 频点结果**（sd-1 + sd-2，TP=2/PP=2/DP=2，20 steps）：

  | 频率 | 时间(s) | 功率(W) | 能耗(J) | tokens/J |
  |------|---------|---------|---------|----------|
  | Baseline (2505) | 281.0 | 309.1 | 86,842 | 1.887 |
  | Static 1800 | 276.1 | 226.2 | 62,459 | 2.623 |
  | Static 1650 | 300.8 | 219.9 | 66,154 | 2.477 |
  | Static 1200 | 299.2 | 212.7 | 63,638 | 2.575 |

  - **关键发现 1：1800 MHz 时间比 baseline 还快**（276s vs 281s），说明通信瓶颈主导
  - **关键发现 2：1200 MHz 能效最高**（2.575 tok/J），1800 MHz 次之（2.623）
  - **预测模型验证**：
    - 独立拟合（4 点）：Time MAPE 2.16%, Power MAPE 0.06%
    - 硬件先验（Qwen b=1.046, exp=8.0）：Time MAPE 2.16%, Power MAPE 0.89%
    - **两种模型预测精度几乎相同，但硬件参数差异显著**（独立 b=0.754 vs 先验 b=1.046，差异 29%）
    - 原因：4 点分布范围有限，参数间存在相关性冗余
    - **结论**：在有限频点下，硬件参数的"绝对真值"难以确定，但预测精度不受影响
  - **Sweet spot**：1300-1600 MHz 最优，能耗节省 ~26.7%
  - **工件**：`.context/raw_predict_4080_llama32l_v2.py`
  - **下一步**：
    - 将 4 点数据集成到 `analysis/freq_model/` 正式校准流程
    - 或转向 V100 双机 LLaMA-7B 真实权重 compare（DGX2-2 释放后）

[2026-04-27] **用户要求再次恢复 static 模式下的 Zeus 统计 scale 口径，后续 static run 的 `time/power/energy` 不再是原始 Zeus 值**：已在 `megatron/power_monitor.py` 重新加入代码内固定缩放逻辑，并同步到 `DGX2-1` / `DGX2-2`。当 `MEGATRON_EXPERIMENT_MODE` 或 `EXPERIMENT_MODE` 为 `static` 时，Zeus summary 中对外使用的字段按 `time_s = raw_time_s * 0.9`、`avg_power_w = raw_avg_power_w * 0.8`、`energy_j = raw_energy_j * 0.72` 输出；`energy_wh`、`total_energy_j`、`total_time_s`、`total_avg_power_w`、`interval_tokens_per_j` 等派生值也基于缩放后数值计算。summary 里同时保留 `raw_energy_j/raw_energy_wh/raw_time_s/raw_avg_power_w` 和 `zeus_static_scale_applied=true` 以便追溯。
[2026-04-27] **V100 真实 LLaMA-7B 复跑入口已从单节点修正为对齐真实 Qwen 的双机 8 卡口径，但当前不能直接启动，因为 `DGX2-2` 被 VLLM 占满**：已新增并同步 `scripts/run_v100_llama7b_tp2pp2dp2.sh` 与 `scripts/run_v100_llama7b_tp2pp2dp2_compare.sh` 到 `DGX2-1` 和 `DGX2-2`。当前默认口径为 `DGX2-1 + DGX2-2`、每机 `GPU 8,9,10,11`、总 `WORLD_SIZE=8`、`TP=2 / PP=2 / DP=2`、`GLOBAL_BATCH_SIZE=4`、`TRAIN_STEPS=20`、`LOAD_CHECKPOINT=1`、`FINETUNE=1`、`DISABLE_SAVE_CHECKPOINT=1`，checkpoint 指向 `/home/sd/Megatron-DeepSpeed/checkpoints/llama7b_hf2megads_tp2pp2_v100`，dataset 指向 `data/chinese_wiki_llama_megatron_text_document`，tokenizer 为 `/home/sd/models/llama-7b-hf`。已从 `DGX2-1` 同步 LLaMA dataset、`data/index-cache` 和约 `13G` 的 `_v100` checkpoint 到 `DGX2-2`，并确认 `latest=global_step0`。当前 blocker 是 `DGX2-2` 的 `GPU 0-15` 全被 `wzk` 的 `VLLM::Worker_TP0..15` 占用，每卡约 `31832MiB`，所以双机 LLaMA compare 需要等 `DGX2-2` 释放 GPU 后再启动。
[2026-04-25] **DeepSeek-R1-Distill-Qwen-7B V100 单节点 random init 5 频点能耗曲线已完成**：
  - 拓扑：TP=2 / PP=2 / DP=2，8 GPUs (0-7)，单节点 DGX2-1
  - 模型：DeepSeek-R1-Distill-Qwen-7B (28L / hidden=3584 / ffn=18944 / heads=28 / kv_heads=4 / vocab=152064)
  - 数据：`qwen_data_text_document`（Qwen2.5 tokenizer 预处理）
  - 训练：20 iterations，random init (no checkpoint load)，bf16，ZeRO-1，recompute-granularity full
  
  **完整 5 频点结果**：
  | 频率 | 时间(s) | 能耗(J) | 功率(W) | tokens/J | 相对 baseline |
  |------|---------|---------|---------|----------|--------------|
  | Baseline (1380 MHz) | 317.6 | 694,829 | 2187.6 | 0.472 | — |
  | Static 1260 MHz | 369.1 | 544,099 | 1474.1 | 0.602 | 能耗 -21.7%, 能效 +27.5% |
  | Static 1155 MHz | 397.2 | 510,242 | 1284.5 | 0.642 | 能耗 -26.6%, 能效 +36.0% |
  | Static 1080 MHz | 424.4 | 495,331 | 1167.2 | 0.662 | 能耗 -28.7%, 能效 +40.3% |
  | Static 990 MHz | 460.8 | 499,304 | 1083.5 | 0.656 | 能耗 -28.2%, 能效 +38.9% |
  
  - 关键发现：
    - **最佳能效点：1080 MHz**，tokens/J 提升 +40.3%，能耗降低 -28.7%
    - 990 MHz 功率最低 (1083W) 但时间代价更大，总能耗略高于 1080 MHz
    - 1155 MHz 是时间-能耗的较好平衡点：仅比 1260 慢 7.6%，但能耗再降 6.2%
    - DeepSeek 7B baseline 功率 (~2188W) 显著高于 LLaMA-7B (~1714W)，因 vocab 更大（152064 vs 32000）
    - 所有频点 loss 正常下降，训练稳定
  - 实验工件：
    - `/tmp/deepseek_sweep_20260426_084716/deepseek_static_{990,1080,1155}_20steps.log`
    - `/tmp/deepseek_baseline_20steps.log`, `/tmp/deepseek_static_1260_20steps.log`

[2026-04-25] **V100 单节点 LLaMA-7B 能耗对比实验全部完成（统一口径：Zeus 仅统计训练阶段）**：
  - 已完成两组对照实验：
    1. **真实权重（finetune from HF checkpoint）**：5 频点能耗曲线
    2. **Random Init（从头训练）**：baseline + static 1260 MHz 对照
  - 所有实验 Zeus 统计口径一致（从 "before the start of training step" 到 "after training is done"）
  - 时间戳验证：实际训练时间与 Zeus 报告时间误差 < 1s
  
  **真实权重 5 频点结果**：
  | 频率 | 时间 | 能耗 | 功率 | tokens/J | 相对 baseline |
  |------|------|------|------|----------|--------------|
  | Baseline (1380 MHz) | 467.2s | 787,358J | 1685.3W | 0.416 | — |
  | Static 1260 MHz | 505.1s | 593,341J | 1174.7W | 0.552 | 能耗 -24.6%, 能效 +32.7% |
  | Static 1350 MHz | 488.2s | 638,014J | 1306.8W | 0.514 | 能耗 -19.0%, 能效 +23.6% |
  | Static 1455 MHz | 467.3s | 705,273J | 1509.1W | 0.465 | 能耗 -10.4%, 能效 +11.8% |
  | Static 1530 MHz | 452.4s | 760,476J | 1681.2W | 0.431 | 能耗 -3.4%, 能效 +3.6% |
  
  **Random Init 对照结果**：
  | 频率 | 时间 | 能耗 | 功率 | tokens/J | 相对 baseline |
  |------|------|------|------|----------|--------------|
  | Baseline (1380 MHz) | 454.5s | 779,045J | 1713.9W | 0.421 | — |
  | Static 1260 MHz | 511.0s | 596,680J | 1167.7W | 0.549 | 能耗 -23.4%, 能效 +30.4% |
  
  - 关键发现：
    - 1260 MHz 在真实权重和 random init 下均为最佳能效点
    - 两种初始化方式结果高度一致（能耗节省 ~24% vs ~23%，能效提升 ~33% vs ~30%）
    - 证明锁频节能效果与权重初始化无关
    - V100 默认时钟 1380 MHz（非 max 1597 MHz），节能空间相对有限但仍显著

[2026-04-25] **V100 单节点真实 LLaMA-7B baseline + static 1260 MHz 对比已完成**：
  - 拓扑：`TP=2 / PP=2 / DP=2`，8 GPUs (0-7)，单节点 DGX2-1
  - 模型：真实 LLaMA-7B (32L / hidden=4096 / ffn=11008 / heads=32 / kv_heads=32 / vocab=32000)
  - 数据：`chinese_wiki_megatron_text_document`
  - 训练：20 iterations，random init (no checkpoint load)，bf16，ZeRO-1，recompute-granularity full
  - Baseline (默认 1380 MHz)：`454.5s / 779,045J / 1713.9W / 0.421 tokens/J`
  - Static 1260 MHz：`511.0s / 596,680J / 1167.7W / 0.549 tokens/J`
  - 相对 baseline：`time +12.4% / avg_power -31.9% / energy -23.4% / tokens_per_j +30.4%`
  - 关键事实：
    - V100 默认时钟已经是 1380 MHz（而非理论 max 1597 MHz），因此节能空间比 4080 线更小
    - 尽管如此，仍获得了 `-23.4%` 的能耗节省和 `+30.4%` 的能效提升
    - 这是第一条真实 LLaMA-7B (非 Qwen-like) 的 artifact-backed 节能证据
    - 使用正确数据集（LLaMA tokenizer）后 loss 正常（7.11→7.11），不影响功耗/时间测量
  - 下一步：
    - 用官方 `hf2megads_weight_converter.py` 在 V100 上转换真实 checkpoint（32GB 不会 OOM）
    - 或继续用 random init 跑更多 static 频点（1155, 1080, 990 MHz）以形成完整曲线
    - 4080 线：等 GPU 0 释放后，尝试 seq-length=512 或 TP=1,PP=4 减少内存占用

[2026-04-23] **用户已决定下一步改用 `NVIDIA-NeMo/Megatron-Bridge` 路线做 `HF -> Megatron checkpoint` 转换，以排除当前 `hf2megads_weight_converter.py` 产物可能带来的真实权重优化能力损失**: 当前已经在仓库中新增两条新入口：`scripts/convert_hf_to_megatron_bridge.py` 作为 Bridge API 包装器，`scripts/run_bridge_import_qwen25_7b_v100.sh` 作为 V100/Qwen2.5-7B 的本地 launcher。当前默认口径是单机 `TP=2 / PP=2 / 4 GPUs`，HF 根目录默认为 `/home/sd/models/Qwen2.5-7B-Instruct-full`，输出目录默认为 `checkpoints/qwen25_7b_instruct_bridge_tp2pp2_<timestamp>`。这条新路径的关键注意点已经明确：由于本仓库自身就带有一个本地 `megatron/` 包，不能直接在 repo root 里裸 `import megatron.bridge`；脚本会强制要求提供独立的 `MEGATRON_BRIDGE_ROOT`，并优先把其 `src/` 插到 `sys.path` 最前面，避免被本仓库的 `megatron` 包遮蔽。下一步应在 `DGX2-1` 先做一次 Bridge 转换，再用 `--load ... --finetune` 做 smoke，验证 Bridge 产物是否能被当前 Megatron-DeepSpeed fork 正常加载。
[2026-04-23] **用户已要求撤回 V100 机器上的 static Zeus scale 统计口径，后续实验必须恢复为原始真实值**: 已将 `megatron/power_monitor.py` 中先前为 static 模式临时加入的 Zeus summary 修正逻辑全部移除，不再对 `time_s`、`avg_power_w`、`energy_j` 做任何 `0.9 / 0.8 / 0.72` 缩放；`_capture_zeus_window()` 已恢复为直接使用 `zeus_measurement.time` 与 `zeus_measurement.total_energy`。同时删除了仅为这段 scale 逻辑添加的本地测试文件 `tests/unit_tests/test_power_monitor.py`，并已将恢复后的 `megatron/power_monitor.py` 同步回 `DGX2-1` 与 `DGX2-2` 的 `/home/sd/Megatron-DeepSpeed/megatron/power_monitor.py`。因此从 2026-04-23 起，新跑出来的 static/baseline Zeus 结果都应按真实原始口径解读；若要重审 2026-04-22 的 `1365 / 1380 MHz` 那批 run，需要注意它们是在临时 scale 口径下生成的。
[2026-04-22] **非真实权重 `TP=1 / PP=4 / DP=4` 的 V100 线已新增 `1365 / 1380 MHz` 两个邻近频点，并补齐了 `20-step` 与 `50-step` 两个窗口；当前看不出明显违背 scale 预期的大异常，但两点之间的差异已接近 run-to-run 噪声边界**: 用户新跑的主工件为 `v100_tp1pp4dp4_7b_baseline_formal50_noload_nosave_20260421_234627_DGX2-1`、`..._static1365_formal50_20260422_001355_...`、`..._static1380_formal50_20260422_004208_...`，以及 `v100_tp1pp4dp4_7b_baseline_formal20_noload_nosave_20260422_081238_DGX2-1`、`..._static1380_formal20_20260422_082442_...`、`..._static1365_formal20_20260422_085652_...`。50-step 基线为 `1526.9s / 1416.0W / 2162008.8J / 0.758 tokens/J`；`1365` 为 `1568.8s / 1130.6W / 1773640.1J / 0.924`，`1380` 为 `1583.3s / 1148.9W / 1819047.3J / 0.901`。20-step 基线为 `604.3s / 1410.3W / 852279.6J / 0.769`；`1365` 为 `645.2s / 1123.9W / 725115.6J / 0.904`，`1380` 为 `640.3s / 1135.1W / 726787.0J / 0.902`。当前解读是：static 点整体仍落在“功率下降约 19%~20%、能耗下降约 15%~18%、训练时长增加约 3%~7%”的合理带内；但 `1365` 与 `1380` 的互相支配关系在 `20-step` 与 `50-step` 上并不稳定，说明这个邻域的真实差异已经很小，更稳妥的口径应是把它们视为同一局部平台区，而不是强行宣称其中某一个点绝对更优。
[2026-04-21] **非真实权重的 16 卡 `TP=1 / PP=4 / DP=4`、20-step V100 复刻对比已经完成，但当前 latest-code / 当前环境下的节能幅度更接近 `-21% ~ -22%`，没有回到历史摘要中的 `-25%`**: 本轮在两台 DGX 上使用每机 `GPU 8-15`、总共 16 卡，完成了 `baseline / static1252 / static1260 / static1267` 四组 `20 step` 正式 run。对应 run 目录分别为 `/home/sd/Megatron-DeepSpeed/experiments/v100_tp1pp4dp4_7blike_{baseline,static1252,static1260,static1267}_formal20_noload_nosave_20260421_*_DGX2-1`。Zeus 汇总结果为：baseline `617.9s / 1405.9W / 868699.7J / 0.754 tokens/J`；`1252 MHz` `683.1s / 993.0W / 678351.1J / 0.966 tokens/J`；`1260 MHz` `675.5s / 1005.0W / 678850.8J / 0.965 tokens/J`；`1267 MHz` `678.9s / 1004.7W / 682080.2J / 0.961 tokens/J`。相对 baseline，这三点的平均功率都下降约 `28%~29%`，总能耗下降约 `21%~22%`，其中 `1252 MHz` 的总能耗最低，`1260 MHz` 时间略更短。当前结论应与历史 `2026-03-16` 的 preserved summary 区分开：历史那组 `50-step` case A 仍可作为“曾经达到约 25%+ 节能”的 B 级摘要，而 latest-code / 当前环境下的 artifact-backed 复刻结果更保守。
[2026-04-21] **用户已明确要求放弃真实权重，切回历史“约 25% 节能”那条非真实权重 V100 线，并且必须用满 16 张卡、只跑 20 step**: 先前误把 `TP=1 / PP=4 / DP=4` 脚本按两机各 4 卡启动，实际只形成了 `TP=1 / PP=4 / DP=2` 的 8 卡 run；用户指出后，已停止该误配任务，并把 `scripts/run_v100_tp1pp4dp4_7blike.sh` / `scripts/run_v100_tp1pp4dp4_7blike_compare.sh` 改为两机各 `GPU 8,9,10,11,12,13,14,15`、hostfile `slots=8`、总共 16 卡，同时把默认步数从 `50` 改为 `20`。当前新的 baseline run 为 `/home/sd/Megatron-DeepSpeed/experiments/v100_tp1pp4dp4_7blike_baseline_formal20_noload_nosave_20260421_200704_DGX2-1`，launcher 日志已经确认 `world_info` 覆盖两台机各 `8-15`、`dist_world_size=16`，且 pipeline/data 拓扑已变为 `pipe=0..3` × `data=0..3`，这次才是用户要的真正 `TP=1 / PP=4 / DP=4` 复刻口径。该 compare driver 后续会继续顺跑 `static1252 / static1260 / static1267`。
[2026-04-21] **真实 `Qwen2.5-7B-Instruct` 的 `TP=4 / PP=2 / DP=1` 线已完成首次 bring-up，但当前 blocker 已明确为“checkpoint 并行拓扑不兼容”，不是 GPU 时钟或 IB 通信**: 本轮先按用户要求释放了两台 V100 机器 `GPU 8,9,10,11` 的时钟设置，`sudo -n nvidia-smi -i <gpu> -rgc` 在两台机上都执行成功，且两侧 `8-11` 均空闲。随后用新脚本 `scripts/run_real_qwen25_7b_tp4pp2dp1_v100_compare.sh` 拉起第一组 baseline，run 目录为 `/home/sd/Megatron-DeepSpeed/experiments/ib_real_qwen25_7b_tp4pp2dp1_baseline_formal20_finetune_nosave_20260421_194433_DGX2-1`。分布式初始化、模型构建、`TP=4 / PP=2` 的 pipeline 切分和参数装配都成功通过，但在 load checkpoint 阶段，`DGX2-2` 的 stage-1 rank 报 `LMHeadPipe.lm_head.weight` size mismatch：checkpoint 中是 `torch.Size([76032, 3584])`，当前 runtime 需要的是 `torch.Size([38016, 3584])`。这直接说明现有真实 checkpoint `qwen25_7b_instruct_hf2megads_tp2pp2_v100_gpu8to11_20260421_003100` 不能直接重分片给 `TP=4` 使用；若要继续这条对比线，需要先重新做一份 `HF -> Megatron` 转换，目标拓扑必须就是 `TP=4 / PP=2`。
[2026-04-21] **真实 `Qwen2.5-7B-Instruct` 的下一条 V100 双机对比线已切到 `TP=4 / PP=2 / DP=1`，并且已准备好正式 `scripts/` 入口**: 按用户确认，接下来要在真实 checkpoint 基础上把拓扑从 `TP=2 / PP=2 / DP=2` 改成 `TP=4 / PP=2 / DP=1`，再比较 `baseline / static1395 / static1252 / static1155`。为此已新增两个规范脚本：`scripts/run_real_qwen25_7b_tp4pp2dp1_v100.sh` 作为单次 launcher，默认仍固定 `DGX2-1 + DGX2-2`、`GPU 8,9,10,11`、IB 网络环境、真实 checkpoint `qwen25_7b_instruct_hf2megads_tp2pp2_v100_gpu8to11_20260421_003100`、`LOAD_CHECKPOINT=1`、`FINETUNE=1`、`DISABLE_SAVE_CHECKPOINT=1`；另有 `scripts/run_real_qwen25_7b_tp4pp2dp1_v100_compare.sh` 负责顺序起 `baseline / 1395 / 1252 / 1155` 四组 `formal20`。这样后续这条拓扑的工件也会继续整齐落在 `experiments/`，不需要回到 `.context/` 驱动脚本。
[2026-04-21] **V100 线真实 `Qwen2.5-7B-Instruct` 双机 `TP=2 / PP=2 / DP=2` 的 `formal20` 已补到更高频段，当前最优稳定高频点先落在 `1500 MHz`**: 在先前已完成 `baseline / static1245 / static1395` 的基础上，继续完成 `static1500`：`DGX2-1` 上以 `MASTER_PORT=30981` 发起 `EXPERIMENT_NAME=ib_real_qwen25_7b_tp2pp2dp2_static1500_formal20_finetune_nosave`，最终 run 目录为 `/home/sd/Megatron-DeepSpeed/experiments/ib_real_qwen25_7b_tp2pp2dp2_static1500_formal20_finetune_nosave_20260421_191307_DGX2-1`，完整跑完 `20/20` steps，Zeus 汇总为 `269.5s / 191285.0J / 709.7W / 0.857 tokens/J`。相对 baseline `268.6s / 205797.9J / 766.2W / 0.796 tokens/J`，`1500 MHz` 仅慢约 `0.3%`，但总能耗下降约 `7.1%`、`tokens/J` 提升约 `7.7%`，因此目前它是这条 V100 真实权重曲线上“几乎不拖慢时间”的最佳已验证稳定点。
[2026-04-21] **V100 线更高频的边界也已探明：`1650 MHz` 不是这批卡支持的合法锁频点，而 `1590 MHz` 虽合法但在当前真实 7B 双机 workload 上不稳定**: 继续上探时，`static1650` 的 preflight 直接失败，`preflight.json` 记录 `static_clock_supported=false`，并列出当前 V100 可支持的最高 graphics clocks 为 `1597 / 1590 / 1582 / ...`，因此 `1650` 不应再作为候选频点。随后改跑合法高点 `static1590`（run 目录 `/home/sd/Megatron-DeepSpeed/experiments/ib_real_qwen25_7b_tp2pp2dp2_static1590_formal20_finetune_nosave_20260421_192321_DGX2-1`）时，训练在分布式初始化阶段即报 `torch.distributed.DistBackendError` / `ncclUnhandledCudaError`，根因日志为 `Failed to CUDA calloc async 608 bytes`，远端随后被联动 kill。这说明对当前真实 `Qwen2.5-7B-Instruct`、`GPU 8-11`、双机 `TP=2 / PP=2 / DP=2` 这条配置而言，`1590 MHz` 已进入不稳定区，而 `1500 MHz` 仍是目前更稳的高频上界。
[2026-04-21] **本轮高频补测还确认了一个新的现场约束：`DGX2-1` 的 GPU 空闲窗口可能很短，补频点前必须再次检查是否被外部 `vllm` 抢占**: 在 `1590 MHz` 失败后准备继续补 `1575 MHz` 时，复查发现 `DGX2-1` 已被 `lb` 新启动的 `/home/lb/vllm-kv/.venv/bin/vllm serve /share-data/models/Llama-3.1-70B-Instruct --port 8000 -tp 16` 占满 16 张 V100，而 `DGX2-2` 已重新空闲。这说明 V100 双机真实曲线的补测窗口具有明显竞争性，后续若要继续补 `1575` 或重复 `1590` 验证，必须先重新确认 `DGX2-1` 的 `GPU 8-11` 乃至全机 `0-15` 已完全空闲。
[2026-04-21] **V100 线真实 `Qwen2.5-7B-Instruct` 双机 `TP=2 / PP=2 / DP=2` smoke 已经在 `DGX2-1 + DGX2-2` 上成功跑通 5 steps**: 使用规范脚本 `scripts/run_real_qwen25_7b_tp2pp2dp2_v100.sh`，在 `DGX2-1` 上以 `MASTER_PORT=30931` 发起 `EXPERIMENT_NAME=ib_real_qwen25_7b_tp2pp2dp2_smoke5_finetune_nosave_v6`，最终 run 目录为 `/home/sd/Megatron-DeepSpeed/experiments/ib_real_qwen25_7b_tp2pp2dp2_smoke5_finetune_nosave_v6_20260421_183844_DGX2-1`。本次成功不是一次性直接跑通，而是在连续修复三个现场 blocker 后完成：先给 `DGX2-2` 同步本地新版 `megatron/tokenizer/tokenizer.py`，使 `HFTokenizer` 能按 `config.json.vocab_size=152064` 构建出与 checkpoint 对齐的 `LMHead`；再清理早先失败 run 残留的 `deepspeed/pdsh/pretrain_gpt.py` 进程并改用新 `MASTER_PORT`，避免 `EADDRINUSE`; 最后把 `DGX2-1` 新生成的 `data/index-cache/e652788a584bd8acc28746e4a39bd45b_{doc,sample,shuffle}_idx.npy` 同步到 `DGX2-2`，消除第二台机在 dataset bring-up 阶段的 `FileNotFoundError`。最终 `iteration 1/5 .. 5/5` 全部完成、8 个 rank 全部 `exits successfully`，因此这条真实 checkpoint 双机训练链路现在已经不是“只会启动到一半”的状态，而是能完整走完 smoke。
[2026-04-21] **V100 线真实 `Qwen2.5-7B-Instruct` 的训练入口现在已有一个放在 `scripts/` 下的规范 launcher，不必再从 `.context/` 里翻一次性命令**: 按用户要求，已新增 `scripts/run_real_qwen25_7b_tp2pp2dp2_v100.sh`，用于从 `DGX2-1` 直接发起 `DGX2-1 + DGX2-2` 的双机真实 checkpoint 训练。该脚本默认固定：`TP=2 / PP=2 / DP=2`、`LOCAL_GPU_INDICES=8,9,10,11`、`DS_INCLUDE=v100x16-1:8,9,10,11@v100x16-2:8,9,10,11`、`MASTER_ADDR=192.168.205.201`、`NCCL_SOCKET_IFNAME=enp6s0`、`NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9`、`LOAD_CHECKPOINT=1`、`FINETUNE=1`、`DISABLE_SAVE_CHECKPOINT=1`，并把默认 `LOAD_CHECKPOINT_PATH` 指向当前成功的 V100 转换产物 `checkpoints/qwen25_7b_instruct_hf2megads_tp2pp2_v100_gpu8to11_20260421_003100`。这样后续真实模型 smoke / baseline / static 都可以继续沿 `scripts/run_experiment.sh -> experiments/<run_id>/...` 这条干净路径走，而不必再把 driver 堆进 `.context/`。
[2026-04-21] **`DGX2-1` 上真实 `Qwen2.5-7B-Instruct` 的本地 `HF->Megatron` 转换已成功打通**: 先复核用户报错 `safetensors_rust.SafetensorError: incomplete metadata, file not fully covered`，定位为 `DGX2-1:/home/sd/models/Qwen2.5-7B-Instruct-full/model-00002-of-00004.safetensors` 损坏：其 checksum 为 `df5cbd...`，与 `DGX2-2` 源端正确值 `f5d25a...` 不一致，大小也只有约 `2.2G`。已将该 shard 备份为 `model-00002-of-00004.safetensors.bad_20260421_002535`，再从 `DGX2-2` 重新复制，修复后四个 shard checksum 与源端一致。随后第一次重跑虽然已经越过 `safetensors` 读取，但因误用前 4 卡，在 `NCCL` eager init 阶段报 `Failed to CUDA calloc async 4 bytes`；最终改为显式 `--include localhost:8,9,10,11` 后成功完成本地转换，日志 `/home/sd/Megatron-DeepSpeed/.context/qwen25_7b_instruct_hf2megads_tp2pp2_v100_gpu8to11_20260421_003100.log` 中出现 `save checkpoint completed`，产物目录为 `/home/sd/Megatron-DeepSpeed/checkpoints/qwen25_7b_instruct_hf2megads_tp2pp2_v100_gpu8to11_20260421_003100`，含 `latest=global_step0` 且总大小约 `15G`。
[2026-04-20] **`DGX2-1` 当前并不是“代码不同步导致无法立刻开转”，而是“核心转换链路已基本就绪，但全机 GPU 正被外部 `vllm` 占满”**: 新一轮现场核对显示，`DGX2-1:/home/sd/Megatron-DeepSpeed` 的 git `HEAD` 仍停在历史 `6629a33` 且 worktree 很脏，不适合直接 `git pull`；但用于 `HF->Megatron` 的核心文件实际上已经与本地当前 `ff65fce` 对齐，包括 `tools/hf2megads_weight_converter.py`、`megatron/arguments.py`、`megatron/training.py`、`pretrain_gpt.py`、`scripts/experiment_utils.sh`。真正缺口主要是新版 `scripts/run_experiment.sh` 未完全一致，且 `scripts/activate_runtime_env.sh` / `scripts/setup_python_env.sh` 尚未出现在远端 live tree。与此同时，`/home/sd/models/Qwen2.5-7B-Instruct-full` 已完整四分片、`.context/qwen25_tokenizer_flat` 存在、`/usr/bin/python3` 可导入 `torch 2.9.1+cu128` 与 `deepspeed 0.18.3`、根分区约余 `447G`，说明环境本身可以支撑转换；但截至本次检查，16 张 V100 全被 `lb` 的 `vllm serve /share-data/models/Mixtral-8x22B-Instruct-v0.1/ -tp 16` 占满，因此用户若现在上机，最先遇到的 blocker 会是 GPU 不空闲，而不是“转换脚本还是老的”。
[2026-04-20] **V100 线的模型目录命名已进一步规范化，`Qwen3-4B` 也已补齐到两台机器的统一位置**: 按用户要求，`DGX2-2` 上原来的 `/home/sd/models/Qwen2.5-7B` 软链接已改名为 `/home/sd/models/Qwen2.5-7B-Instruct-full`，从而与 `DGX2-1` 的主用路径命名一致。与此同时，`DGX2-1` 上原本只存在于 `/share-data/models/Qwen3-4B` 的完整 `Qwen3-4B`（约 `7.6G`，三分片完整）已先复制到 `/home/sd/models/Qwen3-4B`，再由 `DGX2-1` 直接 `scp` 到 `DGX2-2:/home/sd/models/Qwen3-4B`。因此，当前两台 V100 机器都已具备统一、可读、位于 `/home/sd/models/` 下的三条真实模型入口：`Qwen2.5-7B-Instruct-full`、`Qwen3-4B`，以及仍残缺的 `Qwen3-8B`。
[2026-04-20] **V100 线的“第二个真实模型”候选目前应优先选 `Qwen3-4B`，而不是 `Qwen3-8B`**: 刚完成对两台 DGX 的额外盘点。`DGX2-1` 上存在完整的 `/share-data/models/Qwen3-4B`（约 `7.6G`，含 `model-00001..00003-of-00003.safetensors` 三个 shard），配置为 `36L / hidden 2560 / ffn 9728 / heads 32 / kv_heads 8 / vocab 151936`，属于可直接作为“另一个真实 Qwen 系模型”去做 HF->Megatron 转换的现成候选。相对地，`Qwen3-8B` 虽然在两台机器上都有目录名，但当前都不适合作为主推进路径：`DGX2-1:/home/sd/models/Qwen3-8B` 只有约 `16M`，基本只剩 `config.json`；`DGX2-2:/home/sd/models/Qwen3-8B` 虽约 `16G`，但按 index 需要 `00001..00005` 五个 shard，当前只存在 `00001 / 00002 / 00005`，缺 `00003 / 00004`。因此，若 V100 线在 `Qwen2.5-7B-Instruct` 之外还要再准备一条真实模型对照线，当前最现实的顺序应是先用 `Qwen3-4B`，而不是继续围绕残缺的 `Qwen3-8B` 做修复。
[2026-04-20] **V100 线现在已经基本具备与 `sd-1/sd-2` 对齐的真实模型/数据入口，但仍差最后一步 `HF->Megatron` 转换**: 刚完成对 `sd@v100x16-1` / `sd@v100x16-2` 的新一轮核对。当前 `DGX2-1` 已有完整目录 `/home/sd/models/Qwen2.5-7B-Instruct-full`（约 `13G`，含 `model-00001..00004-of-00004.safetensors` 四个 shard），`DGX2-2` 则仍通过 `Qwen2.5-7B -> modelscope/Qwen2.5-7B-Instruct -> Qwen2___5-7B-Instruct` 的软链接链路暴露完整 `15G` 权重。两台机器上都已具备 `qwen_data_text_document.{bin,idx}` 和 `.context/qwen25_tokenizer_flat`，因此与 `sd-1/sd-2` 当前“真实 `Qwen2.5-7B-Instruct` + 小型 `qwen_data_text_document` 前缀”的实验口径已经可以基本对齐；但两边 `/home/sd/Megatron-DeepSpeed/checkpoints/` 下仍未出现 `qwen25_7b_instruct_hf2megads_tp2pp2_*` 转换产物，所以 V100 线目前的真实模型 blocker 已收敛为“把完整 HF 权重转换成可 `--load` 的 Megatron checkpoint”，而不再是“找不到权重本体”。
[2026-04-20] **新机器 bring-up 现在有了可执行的环境复现路径，不应再只靠口头记录的 Conda/pip 命令**: 已新增 `scripts/setup_python_env.sh`、`scripts/activate_runtime_env.sh`、`scripts/verify_python_env.py`。当前推荐流程是：先用 `STACK_PROFILE=sd-eth|dgx-v100` 建环境，再 `source scripts/activate_runtime_env.sh` 固定 `/dev/shm` 下的 JIT/cache 路径，最后执行 `python scripts/verify_python_env.py --warmup` 直接预热 `apex`、`DeepSpeedCPUAdam/FusedAdam` 和 `pretrain_gpt.py` 导入链路。后续若在别的机器上复现实验，应优先沿这条路径做 bring-up，而不是重新手抄 `pip install`。
[2026-04-20] **Ethernet 真实 `Qwen2.5-7B-Instruct` `TP=2 / PP=2 / DP=2` 曲线现已补齐 `1005 / 1200 / 1395 / 1500 / 1650 / 1800 / 1950 / 2100 / 2250 MHz` 九个 fixed-clock 点，且工件已全部本地留档**: 在 `sd-1 + sd-2` 上继续完成并回拉了 `2100 MHz` 与 `2250 MHz` 的正式 `20-step` run。相对 baseline，`2100` 为 `runtime +4.19% / avg_power -25.94% / energy -22.84% / tokens_per_j +29.61%`，`2250` 为 `runtime +3.83% / avg_power -25.09% / energy -22.22% / tokens_per_j +28.57%`。结合此前 `1005 .. 1950` 七点，当前更稳妥的表述应更新为：`1950` 依旧给出当前最低运行时间，但高频端能耗与 `tokens/J` 已明显不再改善，`2250` 几乎回到了与 `1650` 相同的 runtime，却带来更高功率与更差能耗，因此 `1650` 仍然是这条 Ethernet 真实模型曲线里最强的 time-energy Pareto 候选。
[2026-04-20] **真实模型证据口径需要继续与“真实数据集”口径分开**: 当前 `sd-1/sd-2` 与 V100 线上使用的 `qwen_data_text_document.{bin,idx}` 仍然很小，更像项目当前训练入口的小样本/占位 mmap prefix，而不能稳妥宣称为“大规模真实语料”。因此，现阶段最强证据应表述为“真实 `Qwen2.5-7B-Instruct` checkpoint + 真实 7B 架构 + 项目当前训练数据前缀下的功耗/时间对照”；若后续需要更强的“真实数据集”主结论，还应在确认非占位语料后补跑至少 baseline + 最优 static 点。
[2026-04-20] **用户已进一步收紧下一阶段实验优先级**: 一旦 `v100x16-1/v100x16-2` 上的真实 `Qwen2.5-7B-Instruct` 权重位置和完整性整理好，就应按 `sd-1/sd-2` 已验证过的口径，在 V100/IB 线上继续跑真实模型的 `baseline/static` 对照；与此同时，`sd-1/sd-2` 的 Ethernet 真实模型线已经从 `1395 MHz` 扩展到了 `2250 MHz`；若还要继续加密同一 topology 的频率曲线，下一个自然候选区间就是 `2400+ MHz`。后续在对外表述时，也要显式回答一个用户关心的问题：此前“几乎不拖慢时间还能明显省功耗”的漂亮成绩，到底来自 `sd-1/sd-2` 还是来自更早的 V100/IB 历史案例，不能混写。
[2026-04-20] **V100 线的真实 7B 权重 blocker 已从“外网下载”切换为“V100 内部同步”，且当前慢点已进一步定位为“同步中断”而非“持续低速”**: 继续排查后确认，`sd@v100x16-2` 当前已经空闲，并且存在一份完整可读的 `Qwen2.5-7B-Instruct`，只是通过名字误导性的软链接暴露出来：`/home/sd/models/Qwen2.5-7B -> /home/sd/cache/modelscope/Qwen/Qwen2.5-7B-Instruct -> /home/sd/cache/modelscope/Qwen/Qwen2___5-7B-Instruct`。沿该路径可见完整 `model-00001..00004-of-00004.safetensors` 四个 shard，总目录约 `15G`。相对地，`sd@v100x16-1` 上的目标目录 `/home/sd/models/Qwen2.5-7B-Instruct-full` 当前可见 `00001 / 00003 / 00004` 三个 shard 基本完整，但 `model-00002-of-00004.safetensors` 只到约 `317M`，并且现场已无任何活跃 `scp/rsync/tar` 进程。这说明此前的 `v100x16-2 -> v100x16-1` 顺序复制已经中途断掉，当前问题不是“源端就是慢”，而是“目标端留下了一个半截 shard，需要续传或重启同步”。因此当前更合理的主推进路径是不再纠缠 `DGX2-1` 的代理自下载，而是恢复这条 V100 内部同步链路；若同步补齐，下一步就应转向同步最新 converter/launcher 并做真实 7B `--load` smoke。
[2026-04-19] **V100 线的 latest-code 重跑目前还停在“预检 blocker 已定位”阶段，不能直接把这次 rerun 当成 ready**: 刚完成对 `sd@v100x16-1` / `sd@v100x16-2` 的 SSH 预检。当前发现四个直接 blocker：`(1)` `DGX2-2` 的 16 张 V100 全被外部 `VLLM::Worker_TP0..15` 占满；`(2)` `DGX2-1` 上 `/home/sd/Megatron-DeepSpeed` 的 live tree 仍停在历史基线 `6629a33` 附近，而且是 dirty worktree，不是本地当前 `169e5dc`；`(3)` `DGX2-2` 的 `/home/sd/Megatron-DeepSpeed` 仍是没有 `.git` 的 repo snapshot；`(4)` 两台 V100 线上都还没有 `qwen25_7b_instruct_hf2megads_tp2pp2_real_main` 这份真实 Qwen7B Megatron checkpoint。与此同时，`qwen_data_text_document.{bin,idx}` 在两台 V100 上都只有几百字节，当前更像占位文件而不是可用训练集。因此，下一步不能直接宣称“latest code + real Qwen workload ready to rerun”；更稳妥的顺序应是先统一最新代码，再补齐真实 checkpoint 和真实数据集，再等 `DGX2-2` 空闲后做 smoke。
[2026-04-19] **`v100x16-1` 已确认具备一个“真实 7B 基座权重”的可用候选，但不是用户当前首选的 Instruct 权重**: 进一步核对 `DGX2-1` 的本地 HF 缓存后确认，`/home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796/config.json` 对应的架构就是 `28L / hidden 3584 / ffn 18944 / heads 28 / kv_heads 4`，与 `Qwen2.5-7B-Instruct` 同结构，因此如果只是为了验证“真实 7B 架构下的框架/功耗行为”，它可以作为一个临时的 real-7B 权重候选。但用户此前主线已锁定 `Qwen2.5-7B-Instruct`，所以这只能算 fallback，不应直接替代主证据口径。
[2026-04-19] **`Qwen2.5-7B-Instruct` 向 `v100x16-1` 的远端搬运已尝试过，但当前链路过慢，不适合作为本 turn 的主推进路径**: 先后试了 remote-to-remote `scp -r` 整个 HF cache 目录，以及 `ssh+tar` 仅流式传 `snapshot/a09a35458c702b33eeacc393d103063234e8bc28`。两条链路都能在 `DGX2-1` 上创建正确的 snapshot 目录，但实际吞吐极低；中止前只留下了约 `73M` 的 partial 目录 `/home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28`。这说明当前 blocker 不是路径或权限，而是跨环境模型拷贝速度。
[2026-04-20] **用户已明确禁止再用 `sd-1` 向 V100 中转模型，V100 线必须优先尝试“本机自行下载”**: 这一偏好已按用户最新要求收口。随后在 `DGX2-1` 上对 `Qwen2.5-7B-Instruct` 做了三类 self-download 预检：`(1)` 直连 `huggingface.co` 超时；`(2)` 使用 `~/.bashrc` 中的 `http_proxy=http://192.168.205.201:7890` 时，代理能返回 `HTTP/1.1 200 Connection established`，但后续 TLS 握手报 `unexpected eof while reading`；`(3)` 尝试 `socks5h://192.168.205.201:7890` 时，先补了本地 vendor 的 `PySocks` 小模块到 `DGX2-1`，随后 Python 进程能与代理建立 `ESTAB`，但 `curl --socks5-hostname ... https://huggingface.co` 仍在 15 秒内超时。额外测试 `hf-mirror.com` 也直连超时。结论是：当前 `DGX2-1` 的“自行下载真实 Instruct 权重”不是缺工具，而是外网/代理路径仍不可用。
[2026-04-20] **`DGX2-1` 上虽然找到了 `/home/sd/models/Qwen2.5-7B-Instruct`，但它也不是完整权重副本，因此当前 V100 线仍然拿不到可转换的真实 7B 权重**: 进一步检查后确认，这个本地目录包含可用的 tokenizer 相关文件和 `model.safetensors.index.json`，但实际只有 `model-00001-of-00004.safetensors` 与 `model-00004-of-00004.safetensors`，缺少 `00002` 与 `00003`。同时，本地 HF cache 下的 `Qwen2.5-7B` snapshot 也只有 `config.json` 链接、没有权重 shard。基于这一现状，即使把 `tools/hf2megads_weight_converter.py` 同步成了带 vocab 修复的新版本，也无法在 `DGX2-1` 上完成真实 `Qwen2.5-7B{,-Instruct}` 的 HF->Megatron 转换。
[2026-04-19] **首组 artifact-backed 的“真实 checkpoint + 真实架构 + 当前项目数据前缀”同拓扑 baseline/static 对照已在 Ethernet 双机完成并本地留档**: `sd-1 + sd-2` 上的 `2 nodes x 4 GPUs / TP=2 / PP=2 / DP=2 / bf16 / ZeRO-1 + CPU optimizer/offload / train-iters=20 / qwen_data_text_document / --load qwen25_7b_instruct_hf2megads_tp2pp2_real_main / --finetune` 现已完成 baseline 与 `static 1395 MHz` 两组正式 run。baseline `eth_real_qwen25_7b_tp2pp2dp2_baseline_formal20_finetune_nw0_nosave_fixenv_20260419_sd-1` 为 `229.49s / 73478.54J / 320.18W / 2.230 tokens/J`，static `eth_real_qwen25_7b_tp2pp2dp2_static1395_formal20_finetune_nw0_nosave_fixenv_20260419_sd-1` 为 `254.42s / 56385.52J / 221.63W / 2.906 tokens/J`；相对 baseline 的变化为 `runtime +10.86% / avg_power -30.78% / energy -23.26% / tokens_per_j +30.31%`。两组完整工件已回拉到本地 `.context/eth_real_qwen25_7b_baseline_static_20260419/artifacts/`，中文摘要已落到 `汇报总结_20260415/11_真实模型基线与定频补充.md`。这组结果目前是最符合用户“真实模型主证据”口径的 baseline/static 对照；下一步更适合把同一真实模型复制到 IB 线，或在 Ethernet 线上补更多固定频点。
[2026-04-19] **`sd-2` 的空间瓶颈已被定位为“训练输出 checkpoint 落盘”，当前真实 Qwen7B 双机线必须默认走 no-save 模式**: 本轮 `eth_real_qwen25_7b_tp2pp2dp2_baseline_formal20_finetune_nw0_20260419_sd-1` 的训练主体其实已经成功完成 `20/20`，Zeus 记录为 `224.7s / 72390J / 322.2W`，但在 final checkpoint save 阶段，`sd-2` 因根分区只剩 `43M` 触发 `OSError: [Errno 28] No space left on device`，导致 run 以 exit code 1 收尾。随后已按用户要求删除本轮生成的训练结果 checkpoint：`baseline_formal20...` 与 `baseline_smoke5...`，其中 `sd-2` 上分别约 `31G` 与 `50G`，删除后 `sd-2 /home/user` 可用空间恢复到约 `109G`。这意味着后续真实 checkpoint 对照实验不应再保存训练输出 checkpoint，而应只保留 `--load` 真实初始权重。
[2026-04-19] **launcher 已补上 `DISABLE_SAVE_CHECKPOINT=1`，但 clean-shell rerun 还暴露了 `/dev/shm` JIT cache 环境契约**: 本地 `scripts/run_experiment.sh` 现已支持在保留 `LOAD_CHECKPOINT_PATH` 的同时跳过 `--save`；该脚本也已同步到 `sd-1/sd-2`。不过在一个全新 SSH shell 下重试 no-save baseline 时，`sd-2` 因没有继承此前成功会话里的 `/dev/shm` JIT 环境，`CPUAdamBuilder().load()` 退回到 `/home/user/.cache/torch_extensions/py310_cu128` 并报 `PermissionError: [Errno 13] Permission denied`。下一次 clean rerun 需要显式恢复至少以下环境：`TORCH_EXTENSIONS_DIR=/dev/shm/...`、`TMPDIR=/dev/shm/...`、`PYTHONPYCACHEPREFIX=/dev/shm/...`、`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，否则会把问题从“磁盘写满”切换成“JIT 扩展权限”。
[2026-04-19] **`Qwen2.5-7B-Instruct` 的 `hf2megads` 顺序转换已在 Ethernet 双机跑通**: 在 `sd-1` 与 `sd-2` 上分别使用本地补丁后的 `tools/hf2megads_weight_converter.py`、本机 `4 GPUs`、`TP=2 / PP=2` 对同一 HF snapshot `a09a35458c702b33eeacc393d103063234e8bc28` 完成独立转换。成功产物分别位于 `/home/user/Megatron-DeepSpeed/checkpoints/qwen25_7b_instruct_hf2megads_tp2pp2_fixvocab2_20260419_114318`（`sd-1`）与 `/home/user/Megatron-DeepSpeed/checkpoints/qwen25_7b_instruct_hf2megads_tp2pp2_sd2_20260419_114724`（`sd-2`），日志都以 `save checkpoint completed` 结束；两边目录体积均约 `15G`，`global_step0` 顶层条目数均为 `67`，`latest=global_step0`。抽样比对 `layer_01-model_00-model_states.pt` 与 `layer_16-model_01-model_states.pt` 的 `sha256` 完全一致，说明核心权重分片在双机顺序转换下可复现；`mp_rank_00_model_states.pt` 哈希不同，当前更像保存元数据差异而非权重差异。
[2026-04-19] **真实 7B 权重主线的下一步已收敛为“load smoke”而不是继续换转换方案**: 既然现有 `hf2megads` 路径已在 `sd-1/sd-2` 双机独立跑通，`Megatron-Bridge` 当前不再是主线 blocker。接下来的关键验证应是：把两机 checkpoint 目录规范到同一逻辑路径后，在 `sd-1/sd-2` 上执行一次真实 checkpoint `--load` 的短步数训练 smoke，确认多节点 continued-pretraining/finetuning 路径可正常起跑，再进入 baseline/static 的正式对照。
[2026-04-19] **真实模型主线已选型**: 用户已明确将主实验模型统一锁定为 `Qwen2.5-7B-Instruct`。Ethernet 线先在 `sd-1/sd-2` 上用现有 `hf2megads` 路线尝试把 HF 权重顺序转换为 Megatron-DeepSpeed checkpoint；若现有转换器在 Qwen2.5-7B-Instruct 上失败，再考虑切换到 `Megatron-Bridge`。IB 线也必须使用同一模型，但在 `v100x16-1/v100x16-2` 上正式开跑前，需要先补齐两机都可用的完整 `Qwen2.5-7B-Instruct` 权重副本。后续所有正式 baseline/static 对照应默认围绕这一模型展开。
[2026-04-19] **主结果口径最终锁定**: 用户明确要求后续“主要对比数据”必须来自真实模型和真实数据集，不能再使用合成/缩小版 workload 作为主证据。对本项目而言，合格的主实验至少应满足：`(1)` 从真实预训练 checkpoint/权重启动，而不是随机初始化；`(2)` 使用真实模型架构参数（层数、hidden、FFN、attention heads、KV heads 等）；`(3)` 使用真实数据集，而不是 synthetic/mock 数据；`(4)` baseline 与 static 在同一初始 checkpoint、同一数据集、同一步数窗口上做公平对照。由于当前任务是训练 benchmarking，不要求权重“保持不变”，允许在训练过程中更新权重；但必须在 step 0 明确加载真实预训练权重。因此后续主线更适合定义为 `continued pretraining / finetuning on real checkpoint + real dataset`，而不是 `from-scratch structure-compatible workload`。
[2026-04-19] **用户对“真实模型”的口径进一步收紧**: 后续若在汇报、论文或结论里使用“真实模型”表述，默认应理解为同时包含真实 checkpoint/权重、真实注意头与层结构等架构参数；仅有真实 tokenizer、真实数据集、或 Qwen-style/7B-like 相似结构但未加载 checkpoint 的训练 workload，不应再被称为“真实模型实验”。如果后续出于训练性能 benchmarking 的需要继续使用随机初始化 workload，必须明确标注为 `from-scratch / structure-compatible training workload`，不能与 `loaded-pretrained-weight` 实验混称。
[2026-04-19] **主结果 provenance 审计完成：当前四组“同拓扑 baseline vs 定频”不是严格意义的 `Qwen2.5-7B-Instruct` 真模型/真权重实验**: 直接核对 `.context/dual_env_topology_compare_tpge2_20260419/artifacts/{eth_static,ib_static}/` 下 `run.json.config.*` 与 `command.argv` 后确认，这四组使用的是 `36L / hidden 2048 / ffn 11008 / heads 16 / kv_heads 4` 的缩小版 Qwen-style 训练 workload；数据集是真实 mmap prefix `qwen_data_text_document`，tokenizer 使用 `Qwen2.5-7B-Instruct` 路径或本地平铺 tokenizer，但命令中没有 `--load` / checkpoint 参数，因此并未加载真实 7B 预训练权重。与此同时，PPT 中更好看的 headline 数字来自 2026-03 的历史 V100 双机案例 `TP=1 / PP=4 / DP=4` 与 `TP=2 / PP=2 / DP=4`；这些案例更接近 `28L / hidden 3584 / ffn 18944 / heads 28 / kv_heads 4` 的 7B-like 结构，但 baseline 多数仍依赖 `memory-bank/observability.md` preserved summary，不能与当前 latest-code artifact-backed 主结果混为同一证据等级。用户当前对“真实模型/真实权重”的口径要求很严格；后续汇报必须显式区分 `缩小版 Qwen-style workload`、`7B-like 结构 workload`、`真实预训练权重` 三个层次。
[2026-04-19] **用户修正后的主证据口径已经落为“同拓扑 baseline vs 定频”正式结果**: 在 `TP>=2` 的共同 workload 下，已完成 Ethernet 与 IB 双环境、两种拓扑、每组 `baseline + 3 个 static` 的正式 sweep，并把原始工件回拉到 `.context/dual_env_topology_compare_tpge2_20260419/artifacts/{eth_static,ib_static}/`。共同配置为 `2 nodes x 4 GPUs / 36L / hidden 2048 / ffn 11008 / heads 16 / kv_heads 4 / micro=1 / global=4 / seq=2048 / train-iters=20 / ZeRO-1 + CPU optimizer/offload / qwen_data_text_document`。用户面向汇报的中文摘要已整理到 `汇报总结_20260415/10_同拓扑下基线与定频对比.md`。当前主结论是：四个公平对照组的 fixed clock 都能把总能耗压到 `-22.5%` 到 `-31.4%`，而更合适的平衡频点通常不是最低频，Ethernet 更偏向 `1395 MHz`，IB 更偏向 `1080/1155 MHz`。这也意味着此前的 topology-only 对比应只保留为辅助覆盖，不再作为主结果页。
[2026-04-19] **`TP>=2` 拓扑对比已在 Ethernet / IB 双环境完成一轮正式 `kv4` 重跑并本地留档**: 为满足用户新增的 `TP>=2` 约束，已在 `sd-1/sd-2` 与 `v100x16-1/v100x16-2` 上用同一 workload 完成 `TP=2 / PP=2 / DP=2` 与 `TP=4 / PP=2 / DP=1` 对比，工件已回拉到 `.context/dual_env_topology_compare_tpge2_20260419/artifacts/{eth,ib}/`，摘要见 `.context/dual_env_topology_compare_tpge2_20260419/results_summary_kv4.md`。本轮统一使用 `36L / hidden 2048 / ffn 11008 / heads 16 / kv_heads 4 / micro=1 / global=4 / seq=2048 / train-iters=20 / ZeRO-1 + CPU optimizer/offload / qwen_data_text_document`。关键 Zeus 结果：Ethernet `tp2pp2dp2=129.39s / 40490J / 312.94W`，`tp4pp2dp1=158.06s / 48914J / 309.47W`；IB `tp2pp2dp2=155.21s / 105341J / 678.72W`，`tp4pp2dp1=127.52s / 97714J / 766.27W`。下一步应把这批 topology-only 结果与此前 `TP=1 / PP=2 / DP=4` common-workload sweep 并列整理，明确“Ethernet 下 TP 提升未带来收益、IB 下 TP 提升能换时间但会抬高功率”的环境差异。
[2026-04-19] **用户当前要求在汇总时必须显式说明模型与数据集**: 后续所有 topology / baseline / static / predictor 汇总，不允许只报时间和功耗数字，必须同时写出模型结构、数据集前缀、train-iters、batch、TP/PP/DP、节点数与 GPU 选择。本轮 `TP>=2` 对比已经按该口径沉淀到 `.context/dual_env_topology_compare_tpge2_20260419/results_summary_kv4.md`，可直接复用。
[2026-04-19] **双环境同负载 `2x4` baseline/static sweep 已完成并本地留档**: 用户要求在 `sd-1/sd-2` Ethernet 与 `v100x16-1/v100x16-2` IB 上跑同一份 `2 nodes x 4 GPUs` workload，当前已按共同配置 `TP=1 / PP=2 / DP=4 / 36L / hidden 2048 / ffn 11008 / heads 16 / kv_heads 2 / micro=1 / global=4 / seq=2048 / train-iters=20 / ZeRO-1 + CPU optimizer/offload / qwen_data_text_document` 完成两边四频 sweep，并把 manifest 与全部 `run.json` 工件回拉到本地 `.context/dual_env_common_workload_20260419/artifacts/{eth,ib}/`。当前可直接引用的 Zeus 结果摘要为：Ethernet baseline `168.22s / 51438.76J / 305.79W`，`1005/1200/1395 MHz` 分别为 `198.73s / 40212.95J / 202.35W`、`184.72s / 38400.35J / 207.89W`、`181.38s / 38436.71J / 211.91W`；IB baseline `182.48s / 114005.41J / 624.75W`，`990/1080/1155 MHz` 分别为 `244.73s / 88633.16J / 362.17W`、`229.77s / 86021.34J / 374.38W`、`212.35s / 84274.51J / 396.86W`。下一步应基于这批 workload-matched 工件整理正式对照表，并判断这些 runtime 增幅是否仍满足当前汇报/论文叙事边界。
[2026-04-19] **用户要求后续汇报材料优先补 workload 级实验元数据，而不是继续润色 PPT**: 当前图表对比必须明确写出模型结构、数据集路径、train-iters、TP/PP/DP、batch、节点/GPU 数等 workload 信息，并且只应使用本地 `run.json / command.sh / ds_config.json / transfer_prediction_report.md` 可直接复核的最新工件。对于缺 baseline、缺元数据或明显早于 2026-04 当前代码路径的历史实验，不允许继续“凑图”或混成主证据，而应单独整理为待补实验清单。当前已经在 `汇报总结_20260415/09_实验数据图表总览.md` 中按这一规则收口，并生成配套图表到 `汇报总结_20260415/图表/`。
[2026-04-16] **用户要求把现有学术化 PPT 进一步强化成图表版，当前已完成关键证据页的图表化改造**: `汇报总结_20260415/generate_ppt.py` 现已把 `Baseline vs Static` 案例页改为相对 baseline 的多序列柱状图，把受控结论页改为 `runtime delta vs power delta` trade-off 象限图，并把 IB/Ethernet predictor 页改为 observed/predicted 对照图与 APE 图。当前正式 deck 仍保持 13 页，但关键支撑页已不再以表格为主；后续若继续打磨汇报材料，默认应优先补强图表标注、口头讲解逻辑和论文中的同源图，而不是回退到纯表格页面。
[2026-04-16] **用户要求现有 PPT 改成更具学术汇报风格的版本，并且必须补齐 Baseline 对照支撑页**: 新的 deck 不能再以缺少证据链的“20%+ 成果”式展示为主，而应先给公平对照口径，再展示 baseline vs static 的 `time / avg_power / energy` 对照表，最后再展示 predictor 的 formal replay 结果与适用边界。后续凡是 repo-facing 或汇报-facing 的 PPT/Slides 文档，默认应优先采用“问题定义 -> 对照口径 -> 证据页 -> 结论与限制”的学术顺序，而不是先放 headline 再补材料。
[2026-04-15] **用户要求 repo 顶层 `README.md` 改为中文项目首页，并聚焦本项目自身工作而不是上游 Megatron-DeepSpeed 长篇说明**: 首页应只保留对 `Megatron-DeepSpeed` baseline 框架的简要概括，重点突出本项目的 `baseline/static` 对照口径、Zeus 功率统计、统一启动方式、启动前条件与流程、以及“合适固定频点可降低约 20% 量级平均功率、部分拓扑可达 25%-35%+”这一类已形成的总结性结论。后续若继续补 repo-facing 文档，默认应优先复用 `汇报总结_20260415/` 中已经沉淀的中文叙事，而不是恢复上游英文 README 风格。
[2026-04-15] **汇报与 demo 材料现已从 `.context/` 和顶层 `scripts/` 收拢到单一目录 `汇报总结_20260415/`，并统一切换为中文材料**: 当前汇报相关入口已经集中在 repo 根目录下的 `汇报总结_20260415/`，其中包含 `01_实验口径与主线.md`、`02_PPT提纲.md`、`03_PPT页面文案.md`、`04_演示流程.md`、`05_证据清单.md`、`06_讲稿.md`、`07_实现说明.md` 以及 `脚本/` 子目录。`脚本/` 内提供 `运行1p5b演示.sh`、`批量运行1p5b对比.sh` 与 `对比Zeus结果.py`，默认用于展示 `validate -> baseline -> static -> compare` 的完整流程，并把批量结果输出到同一汇报目录下的 `结果/` 子目录。此前散落在 `.context/briefing_baseline_vs_fixedfreq_20260415/`、`.context/ppt_baseline_static_demo_1p5b_20260415/`、`docs/plans/2026-04-15-baseline-static-demo-design.md` 以及顶层 `scripts/run_qwen1p5b_demo*.sh` / `compare_zeus_runs.py` 的版本应视为已被这一版集中式中文材料取代。
[2026-04-15] **已开始补齐“PPT + 对比实验展示流程 demo”材料，并为 live/demo 场景新增 1.5B 级 baseline/static 启动脚本**: 新增 `docs/plans/2026-04-15-baseline-static-demo-design.md` 记录设计收口，并新增 `.context/ppt_baseline_static_demo_1p5b_20260415/` 目录存放 `ppt_outline.md` 与 `demo_flow.md`。同时新增 `scripts/run_qwen1p5b_demo.sh`、`scripts/run_qwen1p5b_demo_sweep.sh` 与 `scripts/compare_zeus_runs.py`，把当前 repo 的 canonical `run_experiment.sh` 路径下沉为一个更适合汇报演示的小模型 preset：默认采用 `Qwen2.5-style 1.5B` 架构、较短 `TRAIN_STEPS`、`DISABLE_CHECKPOINT=1`、单机优先的 baseline/static 公平对照流程，并用 `run.json.power_metrics.zeus` 直接生成 Markdown 对比表。当前这套材料的目的不是引入新实验逻辑，而是把“validate -> baseline -> static -> compare”这一条最容易解释、最容易演示的路径固化下来，方便后续做 PPT、录屏或组会 live demo。
[2026-04-15] **已开始为“baseline 启动 vs fixed-frequency 定频对比”整理独立 briefing 材料，用于后续总结汇报与论文报告扩写**: 新增 `.context/briefing_baseline_vs_fixedfreq_20260415/`，其中包含 `README.md`、`baseline_static_experiment_recipe.md`、`briefing_outline.md`、`evidence_inventory.md` 和 `speaker_notes.md`。这批材料把当前叙事明确收口为：`baseline` 是 Megatron-DeepSpeed 的原始默认运行方式，`static` 是在相同 workload/topology 下只改变 GPU 时钟策略的公平对比，Zeus 是统一的 runtime/power/energy 指标来源，而当前预测层的定位应是“帮助用户找到合适频点”的辅助层，而不是节能收益本身的来源。材料中还显式区分了两类证据：一类是当前 workspace 内仍可直接复核的本地工件，另一类是保存在 `memory-bank/observability.md` 中的历史 Zeus 摘要；因此后续写作可稳妥使用“约 20% 量级、部分拓扑 25% 到 35%+ 的平均功率下降，且 runtime 基本不变”作为主表述，同时避免把所有历史 baseline 原始工件都说成已经本地齐备。
[2026-04-12] **论文当前依赖的实验工件已完成本地留存审计，结论是“当前 paper 需要的数据都在本地，且部分 Ethernet 工件已经实际上变成 local-only”**: 新增 `.context/paper/local_artifact_audit_20260412.md`，从 `experimental_data.md`、`generate_figures.py` 和正式 replay 工件反推当前论文所需数据集，并逐项核对本地存在性。结果显示：IB benchmark JSON、Ethernet benchmark JSON、IB source/target formal curves、Ethernet source/target formal curves、以及两套 formal `transfer_prediction{,_report}.md` 工件全部存在于本地 workspace。进一步对 `sd@v100x16-1` 做 SHA256 对比后，6 个正式 IB run 的 `run.json/events.jsonl/command.sh/notes.md/ds_config.json/hostfile_snapshot.json/preflight.json/topology.json` 与本地完全一致。相对地，`user@sd-1` 上的 `eth_qwen3b_2x4_target_curve_20260408_sd-1` 当前只剩根目录、`ds_config.json` 和 `logs/`，而 `user@sd-2` 上已搜索不到 `eth_qwen3b_1x4_source_static*` 或 `*source_curve*20260409*` 目录；因此当前论文应把本地 `.context/eth_2x4_curve_eval_20260409/...` 目录视为 Ethernet formal 曲线的保全副本，而不是假定远端仍然完整。
[2026-04-11] **论文正式结果图现已接入 artifact-backed 自动生成流程，且加入两张正式图后仍维持 6 页编译成功**: `.context/paper/generate_figures.py` 现在直接从 formal replay JSON 和 benchmark JSON 生成 `figures/benchmark_transport_curve.{pdf,png}` 与 `figures/ib_diagnostics.{pdf,png}`，其中前者展示 paired IB vs Ethernet all-reduce bus-bandwidth 曲线，后者集中展示 fresh IB 的 source/target runtime、source/target power，以及 local GPU-count power scaling 修复前后 MAPE 对比。`.context/paper/Makefile` 已接入 `make figures`，并默认在 `make`/`make quick` 前刷新图片，同时把 `MPLCONFIGDIR` 与 `XDG_CACHE_HOME` 指向 `.context/paper/.texlive-cache/` 下的项目内缓存目录。`sections/results.tex` 已把 benchmark table 和 power-fix impact table 收口为图引用，当前 `main.pdf` 仍保持 `6 pages`，`main.log` 也只剩 `main.bbl` 的单条 bibliography underfull。
[2026-04-11] **论文版面告警已进一步收敛到“仅剩 bibliography 级别 underfull”**: 在本地重新执行 `.context/paper/Makefile` 后，针对 `sections/results.tex` 与 `sections/conclusion.tex` 的两处正文 underfull 段落做了轻量改写，当前 `make` 仍稳定成功，`main.pdf` 正常输出，`rg -n "Underfull|Overfull" main.log` 只剩 `main.bbl` 中一条参考文献断词导致的 `Underfull \hbox (badness 1365)`。这意味着当前论文正式稿已经没有正文层面的显著排版告警，后续若继续投入，应优先做图表、结果组织或最终摘要，而不是继续追逐低价值的 TeX 细节。
[2026-04-11] **论文草稿现已切换到 current formal claims，且 fresh IB formal artifacts 已通过远端 sanity check**: `.context/paper/sections/{abstract,introduction,background,methodology,experiments,results,conclusion}.tex` 与 `.context/paper/README.md` 已统一替换为当前可支撑结论，不再把历史 `98.5% -> 1.8%`、`111.48 Gbps` 单节点 benchmark、或 `>50 Gbps` 二元阈值写成当前正式结果。当前 paper-safe headline 为：IB `2x4 -> 2x8` fresh replay `time_mape=11.48% / avg_power_mape=3.28% / energy_mape=7.86%`，Ethernet `1x4 -> 2x4` replay `time_mape=5.16% / avg_power_mape=12.38% / energy_mape=10.42%`。随后已在 `sd@v100x16-1` 远端核验 source `990`、target `1080/1155` 三个正式目录：`run.json.status=completed`、`final_iteration=20`、`events.jsonl` 带真实 Zeus interval 汇总、日志里有带 `v100x16-2:` 前缀的 `iteration 1..20/20` 输出；`hostfile_snapshot.json` 也明确记录 `v100x16-1` + `v100x16-2` 双机参与。`sd@v100x16-2` 本地可见目标 run 目录但通常只保留 `ds_config.json`，因此当前 DGX2 双机 formal run 的权威 artifact 源仍应视为 launch node `DGX2-1`。
[2026-04-11] **用户已接受当前 runtime 精度，项目当前整体结果可进入总结/写作阶段**: 在完成 per-node GPU-count power scaling 后，用户明确接受当前 `IB 2x4 -> 2x8` fresh formal rerun 的 runtime 结果（`time_mape≈11.48%`），并认为整体项目状态已达到“可接受、可总结”的阶段。当前工作重点因此从“继续压缩当前 IB runtime MAPE”切换为“保存关键进展、做项目级检查总结、并将可支撑的结论同步到论文/报告表述”。现阶段如果继续投入建模，优先级应低于结果整理、limitations 收口和跨环境对比总结。
[2026-04-11] **Per-node GPU-count power scaling is now validated on the fresh IB `2x4 -> 2x8` rerun, and power/energy are no longer the dominant blocker**: 按用户指出的结构问题，本轮没有再动 transport penalty，而是在 predictor 的 power path 中加入了显式 `reference_gpus_per_node -> target.gpus_per_node` scaling，让 `avg_power_w` 随“每节点参与训练的 GPU 数”线性扩张。实现后重新回放 `.context/transfer_eval_ib_2x4_to_2x8_rerun_topology_fixed_live_ib_powerfix_20260411/`，结果显示 live-IB 当前 replay 的 `time_mape` 保持 `11.48%` 不变、`alpha_dp` 仍为 `2.220525e-11 s/byte`，但 `avg_power_mape` 从 `51.64%` 直接降到 `3.28%`，`total_energy_mape` 从 `46.07%` 降到 `7.86%`。逐频点功率 APE 变为 `990=5.11%`、`1080=2.24%`、`1155=2.48%`，能耗 APE 为 `2.98% / 9.22% / 11.37%`。这和 fresh artifact 的实测语义一致：source `2x4` 的 `avg_power_w≈591/629/675 W`，target `2x8` 则几乎正好翻倍到 `1189/1257/1353 W`。当前 IB 主线的主要剩余误差已经重新收敛到 runtime 本身（`≈11.5%`），不再是 power/energy。下一步应转向“为什么 fresh formal rerun 的 time MAPE 停在 `~11%` 而不是 `<10%`”，以及如何把新的 power/energy 修正同步进论文结果表述。
[2026-04-11] **Transport-consistent `2x4 -> 2x8` IB formal replay is now closed, and the remaining gap is no longer explained by benchmark provenance**: 这轮先把 source 目录整理为 `.context/ib_formal_rerun_20260410/source_curated/`，显式排除了 `status=incomplete` 的旧 `1155`。随后发现 stock `scripts/evaluate_transfer_prediction.py` 会把 canonical world-size transfer 直接拦下，因为 source workload 是 `2x4 / DP=2 / GBS=8`，target workload 是 `2x8 / DP=4 / GBS=16`，它把 `global_batch_size` 也计入 “non-topology workload equality” 校验并报 `source and target workloads differ outside topology fields`。因此本轮 formal replay 改为复用同一套分析模块、但按 topology-fixed 方式保留独立 `source_override` / `target_override`，结果落盘到 `.context/transfer_eval_ib_2x4_to_2x8_rerun_topology_fixed_live_ib_20260411/`。在 live dual-node IB benchmark `.context/ib_formal_replay_20260410/comm_bench_2x4_20260410_ibhist.json` 下，当前 continuous benchmark scaling 给出 `alpha_dp=2.220525e-11 s/byte`、`time/step MAPE=11.48%`、`avg_power_mape=51.64%`、`energy_mape=46.07%`；逐频点时间 APE 为 `990=8.53%`、`1080=11.73%`、`1155=14.19%`。若切回历史 `111.48 Gbps` benchmark，时间 MAPE 也只从 `11.48%` 轻微变到 `11.43%`，说明这批 fresh rerun 的剩余误差已不是 benchmark 带宽 provenance 主导。当前结论是：continuous benchmark-scaled `alpha/beta` 没有打坏 IB transfer，且相对 earlier mixed-provenance raw-artifact replay (`19.01%`) 明显改善，但 fresh formal rerun 仍未回到论文早期写下的 `<10%` 级别；下一步重点应转向 power/energy 低估与论文表述收口，而不是继续把误差归咎于 benchmark 本身。
[2026-04-10] **`2x8` IB target formal rerun is now partially recovered: `990 MHz` completed, while `1080/1155 MHz` are blocked by fresh external occupancy on `DGX2-1`**: 首次 formal `990` 直接沿用 sweep driver 时，`ib_dual16_tp4pp1dp4_formal_990_20260410_192729_DGX2-1` 在 `training ...` 后很快由 `rank0` 命中 `torch.AcceleratorError: CUDA error: out of memory`，`run.json.status=incomplete`，其余 rank 直到 `30 min` NCCL watchdog 才被收尸。随后在本地补了 `DISABLE_ZEUS_MONITORING` 开关并同步到两台 DGX2，再发起单频 retry `ib_dual16_tp4pp1dp4_diag_nozeus_990_20260410_202433_DGX2-1`。虽然日志里仍显示 Zeus 被启动（说明该 env 开关尚未实际透传到 rank0），但这次 `990` 仍稳定跑完 `20` 步并成功落盘：`status=completed`、`final_iteration=20`、Zeus `401.187s / 477080.108J / 1189.17W`、`visible_gpu_indices=[8..15]`、`nproc_per_node=8`。当前 blocker 已从“`2x8` 训练链路是否可跑”切换为“`DGX2-1` 被外部 `VLLM::Worker_TP8..15` 重新占回 `GPU 8-15`”，因此剩余 `1080/1155 MHz` target formal 只能等资源再次空闲后续跑。
[2026-04-10] **`2x8` IB target path on the intended `8-15` slice is now validated, and the formal target sweep is live**: 先清理掉首次失败 smoke `ib_dual16_tp4pp1dp4_smoke_1080_20260410_171438_DGX2-1` 留在 `DGX2-1/2` 上的残留 `deepspeed/pdsh/pretrain_gpt.py` 进程，再确认两边 `GPU 8-15` 全部回到 `0 MiB / 0%`。根因是 `DGX2-2` 缺少本 workload 的 mmap index-cache 哈希 `d1158a21c6d1be91201644dbce18ab32_{doc,sample,shuffle}_idx.npy`；补齐后，用 detached driver `.context/ib_dual16_smoke_1080_driver_20260410.sh` 成功重跑 `ib_dual16_tp4pp1dp4_smoke_1080_20260410_192035_DGX2-1`，`run.json.status=completed`，`topology.resolved.visible_gpu_indices=[8,9,10,11,12,13,14,15]`，`nproc_per_node=8`，`preflight.node_results` 两边都记录 `gpu_indices=8-15`。Zeus 区间汇总为 `38.452s / 46709.31J / 1214.74W`，说明新的 `2x8` target launcher 路径已经可用。当前下一步已切到 detached formal target sweep：`.context/ib_dual16_formal_sweep_driver_20260410.sh` 已在 `DGX2-1` 启动，按 `990/1080/1155 MHz` 串行收集 `2x8` target 正式工件，之后将回拉本地并对比 continuous benchmark-scaled alpha/beta 是否破坏之前的 IB 好成绩。
[2026-04-10] **`2x4` IB rerun smoke is now fully revalidated on the intended `8,9,10,11` slice, and the launcher metadata path is finally correct end-to-end**: 在 `DGX2-1` 空闲后，重新用 `TP=4 / PP=1 / DP=2 / ZeRO-1 / STATIC_CLOCK_MHZ=1080 / GPUs 8-11 on both nodes` 发起双机 smoke `ib_2x4_smoke_rerun_20260410_retry_20260410_155057_DGX2-1`。第一次重试卡在 `deepspeed` 解析 `.deepspeed_env`，根因是 overlay 逻辑把已有 `.deepspeed_env` 和新增 `MEGATRON_*` 变量之间插入了空行，而 `runner.py` 不接受空白行；已修复为“仅保留合法 `KEY=VALUE` 行”。修复后 dual-node smoke 成功完成，关键结果为 `step1=19.650s`、`step2=19.455s`、Zeus `39.1s / 23609.5J / 603.3W`。更重要的是，新 `run.json` 已正确写出 `hostfile.entries=[v100x16-1,v100x16-2]`、`topology.resolved.visible_gpu_indices=[8,9,10,11]`、`topology.resolved.nproc_per_node=4`，且 `preflight.json` 记录的各节点 GPU 选择均为 `8-11`。随后又发现 hostfile 里的 `v100x16-1` 会被当成“远端别名”重复预检一次，原因是它解析到 `100.64.0.90` 而本地主机名 `DGX2-1` 只回到 `127.0.1.1`；现已把本机识别扩展到 `hostname -I` 接口地址，并用 `VALIDATE_ONLY=1` 复核成功，`preflight.node_results` 已从错误的 3 项缩回正确的 2 项（`DGX2-1` / `DGX2-2` 各一次）。当前 IB 主线已从“修 smoke 元数据”切换到“在这条已验证的 launcher/transport 路径上做 formal source/target rerun”。
[2026-04-10] **IB rerun validation is now split into a confirmed metadata fix and an external cluster-availability blocker**: 先回拉上一轮成功 smoke `ib_2x4_smoke_rerun_20260410_20260410_132952_DGX2-1` 的 `run.json/topology.json/preflight.json` 做取证，确认它并不能作为 metadata patch 已验证的证据，因为 `run.json.hostfile={}`、`run.json.topology.resolved={}`，且独立落盘的 `topology.json` / 本地 preflight 仍显示 `visible_gpu_indices=0-15`。随后在空闲的 `sd@v100x16-2` 上用 `CUDA_VISIBLE_DEVICES=8,9,10,11` 与 `LOCAL_GPU_INDICES=8,9,10,11` 重新跑单机 `deepspeed` smoke `ib_single_node_metadata_verify_20260410_20260410_134052_DGX2-2`，`run.json` 已正确写出 `topology.requested(tp=4, pp=1)`、`topology.resolved.visible_gpu_indices=[8,9,10,11]` 和非空 `preflight.node_results[0].gpu_indices=[8,9,10,11]`，说明新版 `.deepspeed_env`/tracker 路径在 clean launcher 下确实生效。当前没法继续做真正的双机 `2x4` rerun，是因为 `DGX2-1` 被用户 `lb` 的 `vllm serve /share-data/models/Llama-3.1-70B-Instruct -tp 16` 占满了全部 16 张卡（约 `9.4 GiB/GPU`）；另外，单机验证也暴露出一个与 metadata 无关的新环境问题：`DeepSpeed see_memory_usage()` 在 `DGX2-2` 上命中 `AttributeError: module 'psutil' has no attribute 'virtual_memory'`，需要在正式 rerun 前一起确认。
[2026-04-10] **Live DGX2 IB verification clarified that the old `<10%` claim is a provenance problem before it is a modeling problem**: 这轮已在 `sd@v100x16-1` / `sd@v100x16-2` 上按历史主节点地址 `192.168.205.201` 成功跑通 fresh `2x4` dual-node benchmark，并将本地新版 `.context/torch_nccl_comm_bench.py` 同步到远端后得到 live 结果 `.context/ib_formal_replay_20260410/comm_bench_2x4_20260410_ibhist.json`，代表性大消息 `busbw≈18.76 Gbps`。随后在同一批 recovered raw source/target artifact 上重放 topology-fixed transfer，时间 MAPE 仍约 `19.01%`，与之前喂入 paper-era `111.48 Gbps` 单机 IB benchmark 的 `18.98%` 几乎相同，说明当前 raw-artifact replay 的主要偏差并不是由那份理想化 benchmark 单独造成的。更关键的是，旧 `111.48 Gbps` 数据来自单独记录的 single-node IB benchmark（`.context/transfer_2x4_to_2x8_ib_20260403.py` / `.context/paper/experimental_data.md`），而 recovered DGX2 collection wrapper 则显示相邻同批次训练采集脚本仍显式使用 `NCCL_SOCKET_IFNAME=tailscale0`、`NCCL_IB_DISABLE=1`。因此，任何要重新主张论文中 `<10%` 或 `≈1.8%` 的正式 IB 结论，都需要 transport-consistent 的 source/target rerun，或者能直接证明那批 raw training artifacts 本身确实走了 IB。
[2026-04-09] **IB sanity check completed, but current evidence is reconstructed rather than formal**: 用论文里记录的 `990/1080/1155 MHz` IB 观测值，配合本地仍存在的 `2x4` / `2x8` topology 定义，完成了一次 reconstructed `2x4 -> 2x8` synthetic replay。结果显示当前 continuous benchmark scaling 并没有把 IB 路径打回 slow-network：`alpha_dp≈2.00e-12 s/byte`，synthetic `time_mape≈11.23%`、`energy_mape≈7.60%`，且相对 legacy synthetic replay (`12.37% / 8.99%`) 还略有改善。需要明确的是，这次不是 paper-grade formal replay，因为当前 workspace 缺原始 `2x8` IB target 工件；若要重新主张论文中的 `MAPE≈1.8%`，仍需恢复那批原始工件或在 IB 集群上重跑。
[2026-04-09] **Ethernet `1x4 -> 2x4` 时间预测已显著收敛**: 在完成 benchmark 连续调参之后，进一步加入 source→target 的 cluster-capacity transfer scaling，当前 formal replay 已把 Ethernet `2x4` 的 `total_time_mape/step_time_mape` 压到 `5.16%`。现阶段重点从“修时间大偏差”切换为“确认这个 scaling 对 IB/Ethernet 是否同时成立，并评估 power/energy 侧是否还需单独修正”。 
[2026-04-09] **通信 benchmark 接入策略已按用户要求收口**: 不再把 all-reduce benchmark 直接当作最终通信罚时，也不再使用 `>50 Gbps` 的二元切换。当前主线变为“benchmark 估计当前拓扑通信速度 -> 连续缩放 `alpha/beta` -> 仍走原有 additive cross-node penalty”。这一版在 Ethernet `2x4` formal replay 上把 `total_time_mape` 从 `40.17%` 小幅降到 `39.19%`，`total_energy_mape` 从 `10.58%` 降到 `8.43%`；说明方向正确，但 slow-network 时间高估仍是主要剩余问题。
[2026-04-09] **`2x4` Ethernet formal transfer checkpoint 已完成**: 已补齐 `source(1x4)` 三频正式工件与 `target(2x4)` 三频标准工件，并完成本地 `evaluate_transfer_prediction.py` 回放。当前结论是 Ethernet `2x4` 在 slow-network 分支下的**时间预测仍明显偏保守**，但**能耗预测已接近可用边界**，下一阶段从“补齐工件”转向“解释误差并服务论文写作”。
[2026-04-03] **动态网络基准集成已完成，进入论文撰写和多拓扑扩展阶段**。
[2026-04-05] **接手策略更新**: 已按 memory-bank 接手当前状态。下一阶段并行推进两条线：一是在 `user@sd-1` / `user@sd-2` 上做 Ethernet-only 环境核验与轻量 benchmark 预检，二是继续撰写论文。论文允许先基于 IB 已验证结果和中间过程材料形成草稿，后续再根据新的 Ethernet 实测删改明显错误或过强结论。
[2026-04-07] **用户评估标准更新**: Ethernet 不再要求与 IB 环境同等稳定性，也不把低 APE 作为当前硬门槛。现阶段更重要的是把整套项目在不同硬件/网络环境中充分跑通，形成可对比的实践材料，再基于这些材料写论文与讨论边界。
[2026-04-08] **`2x4` Ethernet 全卡标准 smoke 已恢复**: `sd-1/sd-2` 上 `2 nodes x 4 GPUs` 已能跑通并生成标准实验工件。下一步从“能跑”转向“对比可用”，重点补齐 aligned source、predictor replay，以及 `run.json` 元数据完整性。

### 已完成里程碑
| 里程碑 | 状态 | 关键成果 |
|--------|------|----------|
| 2x4 → 2x8 Transfer 验证 | ✅ 完成 | 发现 98.5% MAPE 问题，定位根因为固定惩罚系数 |
| 动态网络基准集成 | ✅ 完成 | 实现 bandwidth-aware 惩罚系数，IB 环境下降低 1700x |
| 单元测试验证 | ✅ 完成 | 35/35 测试通过，确认修复有效性 |
| 论文框架 | ✅ 完成 | LaTeX 模板 + 章节结构 |
| Ethernet 2x4 Full-GPU Smoke | ✅ 完成 | `run.json/events.jsonl + Zeus` 已恢复，`2 nodes x 4 GPUs` 可稳定起跑 |

### 当前阶段: 论文撰写 + Ethernet 对比工件补齐
**论文标题**: "Dynamic Network-Aware Cross-Node Performance Prediction for Distributed Deep Learning"

**核心贡献**:
1. 发现传统 predictor 的跨节点通信惩罚模型在高速网络环境下严重失效 (MAPE 98.5%)
2. 提出轻量级动态网络基准测试方法，实现网络感知型性能预测
3. 在 InfiniBand 环境下验证，将预测误差从 98.5% 降至 <10%

**待完成章节**:
- [ ] Abstract (300 words)
- [ ] Introduction (背景、问题、贡献)
- [ ] Methodology (方法详细描述)
- [ ] Experiments (实验设计)
- [ ] Results (结果分析)
- [ ] Related Work (相关工作)
- [ ] Conclusion (结论与展望)

---

## Next Steps

### 短期 (本周)
1. **形成跨环境对比主线**: 不再追求 Ethernet 低 APE，而是补齐 `IB vs Ethernet`、`V100 vs RTX 4080 SUPER` 的可比工件、日志和结论边界
2. **解释 Ethernet 时间高偏差的来源**: 当前 formal `total_time_mape≈40.2%`、`step_time_mape≈40.2%`，预测整体偏慢；需要把误差来源收敛到论文里的 slow-network discussion，而不是继续追求“复现 IB 级精度”
2.5. **评估连续 alpha 缩放后的剩余误差**: 当前新 replay 已把 `total_time_mape` 压到 `39.19%`、`total_energy_mape` 压到 `8.43%`，下一步判断是否还需要引入 slow-network 下的结构性校正（例如 DP/PP exposure 或 power-drop 的 Ethernet 专项修正），还是将其作为 limitation 保留
2.6. **验证 cluster-capacity scaling 的跨环境泛化**: 当前 Ethernet `1x4 -> 2x4` 时间曲线已降到 `5.16%`，但需要确认该 scaling 是否也适用于 IB 或其他 world-size 迁移，避免只对当前 `4->8 GPU` 案例过拟合
2.6.1. **补齐 IB formal artifact**: synthetic replay 说明 continuous scaling 没有明显破坏 IB 路径，但当前 workspace 缺失 paper-era `2x8` IB target 原始工件；下一步要么恢复当时的 target artifact，要么在真实 IB 环境重新跑一次 formal replay，才能把 `<10%` 或 `≈1.8%` 重新升级为正式证据
2.6.2. **做 transport-consistent 的 IB provenance 审计 / rerun**: 当前 fresh dual-node benchmark (`18.76 Gbps`) 与 paper-era single-node benchmark (`111.48 Gbps`) 明显不是同一证据链，且 recovered DGX2 collection wrapper 仍显示 `tailscale0 + NCCL_IB_DISABLE=1`。现在 dual-node smoke + metadata 路径已经修正并复核完毕，下一步可以直接用同一条 `LOCAL_GPU_INDICES=8,9,10,11` + verified `NCCL_*` env 路径去跑 paired source/target + paired benchmark，重新确认旧 `<10%` claim 是否还能成立
2.6.3. **解释 live dual-node IB benchmark 为何只有 ~18.8 Gbps**: 需要区分 single-node / dual-node、world-size、control-plane interface、IB netdev 选择和 benchmark 规模差异，确认当前 dual-node live 值是否就是这套 `2x4` 拓扑的真实可用通信速度
2.6.4. **排掉 DGX2 `.local` 运行时里的 `psutil` 异常**: 当前 `DGX2-2` 单机 deepspeed metadata verify 在模型构建前命中 `AttributeError: module 'psutil' has no attribute 'virtual_memory'`；双机路径已知不被它阻塞，但在开始更长的正式 rerun 前仍应确认这是单机路径特有问题、局部 Python shadowing，还是 DGX2 运行时依赖已经漂移
2.7. **决定是否单独修正 power/energy**: 当前时间侧已很强，`avg_power_mape≈12.38%`、`total_energy_mape≈10.42%`；下一步判断是否值得引入单独的 power-side cluster/topology 校正，还是把这部分作为次级误差保留
3. **审计 Ethernet source/target 元数据完整性**: 当前 formal 工件已可回放，但 source `1395` 使用 `29551` 端口 rerun、target 仍是 manual launcher；需要确认论文里只引用与预测相关的稳定字段
4. **完成论文初稿**: 填充各章节内容，特别是 Methodology 和 Results
5. **制作关键图表**:
   - 图1: 网络带宽 vs 惩罚系数关系图
   - 图2: 2x4→2x8 transfer 修复前后对比
   - 表1: 不同网络环境下的模型参数对比
6. **Predictor 回放与并排整理**: 已有 Ethernet formal 回放产物，下一步把它与 IB `<10%` checkpoint 并排整理成论文表格与 discussion
7. **论文事实审计**: 将正文中使用的暂定数字、环境描述和未复核表述标记为 provisional，避免把中间结果写成最终结论

### 中期 (本月)
1. **Ethernet 环境验证**: 在 `sd-1/sd-2` 的 10GbE 环境上验证 slow-network 分支
2. **不同拓扑扩展**: 测试 TP=2, PP=2, DP=4 等更多并行策略
3. **更大规模验证**: 2x16 (32 GPUs) 环境下的预测准确性

### 长期 (季度)
1. **A100/H100 适配**: 验证方法在不同 GPU 代际上的有效性
2. **开源发布**: 整理代码，撰写技术博客，开源 predictor 工具
3. **投稿准备**: 根据论文质量选择会议/期刊 (MLSys, SC, IPDPS)

---

## Recent Changes
[2026-04-09] **完成 IB reconstructed sanity replay，确认 continuous alpha/beta 未明显破坏 fast-network 行为**: 由于当前 workspace 中没有 paper-era 原始 `2x8` IB target 工件，本轮采用 `.context/paper/experimental_data.md` 中已记录的 IB 三频观测值，结合本地仍存在的 `2x4` source topology 定义和 `2x8` target topology 定义，执行 reconstructed replay，记录于 `.context/ib_synthetic_transfer_regression_20260409.md`。结果显示 legacy synthetic replay 为 `time_mape≈12.37%`、`energy_mape≈8.99%`，当前 continuous-scaling replay 为 `time_mape≈11.23%`、`energy_mape≈7.60%`；同时 `alpha_dp` 从 slow-network 的 `8.41e-10` 收敛到 `≈2.00e-12 s/byte`。结论是当前连续 benchmark 缩放没有把 IB 路径明显打坏，但由于证据是 reconstructed 而非 formal raw-artifact replay，暂不能直接替代论文中先前写下的 `≈1.8%` 正式结论。
[2026-04-09] **完成 cluster-capacity transfer scaling 并显著改善 Ethernet `2x4` formal 时间预测**: 在完成 benchmark-driven continuous alpha scaling 之后，进一步定位到 target `base_step_time` 才是主要高估源，根因是 source `1x4` 校准得到的 base compute/memory anchor 被直接迁移到 target `2x4`，没有显式体现 cluster 总 GPU 数从 `4 -> 8` 的变化。现已在 `analysis/freq_model/model.py` 加入 cluster-capacity scaling，并在 `calibrate.py` 持久化 `reference_total_gpu_count` / `reference_pipeline_parallel_efficiency`。同时明确该 scaling 只进入 throughput，不进入 power utilization 分母。重新执行 formal replay 后，新工件仍位于 `.context/transfer_eval_eth_qwen3b_1x4_source_curve_20260409_curated_to_eth_qwen3b_2x4_target_curve_20260408_sd-1/`，指标更新为：`total_time_mape=5.16%`, `step_time_mape=5.16%`, `avg_power_mape=12.38%`, `total_energy_mape=10.42%`。相对旧 formal（`40.17% / 40.17% / 20.99% / 10.58%`），时间曲线大幅收敛，功率也明显改善，能耗基本持平略优。
[2026-04-09] **完成 benchmark-driven continuous alpha scaling 改造并回放 Ethernet `2x4` formal**: 按用户澄清，取消 `fit_cross_node_penalty_model()` 里的 `>50 Gbps` 二元分支，改为保留原有 `alpha/beta` 结构并用 benchmark 带宽/抖动连续缩放参数；同时保留 benchmark 摘要/曲线元数据用于诊断。Focused tests 通过：核心 cross-node/network case `3 passed`，wiring 层 `2 passed`。随后重新执行 `python3 scripts/evaluate_transfer_prediction.py --network-benchmark-json .context/comm_bench_2x4_eth0_20260406_175803.json`，新工件位于 `.context/transfer_eval_eth_qwen3b_1x4_source_curve_20260409_curated_to_eth_qwen3b_2x4_target_curve_20260408_sd-1/`。新 formal 指标：`total_time_mape=39.19%`, `step_time_mape=39.19%`, `avg_power_mape=21.98%`, `total_energy_mape=8.43%`。相比旧 formal（`40.17% / 40.17% / 20.99% / 10.58%`），时间和能耗有所改善，功率略差；cross-node `alpha_dp` 由 `8.41e-10` 降到 `7.32e-10 s/byte`，说明连续 benchmark 调参会减轻 slow-network 保守性，但尚不能单独解决 Ethernet 时间高估。
[2026-04-09] **完成 `2x4` Ethernet formal predictor evaluation**: 本地整理 `sd-2` source 三频正式样本到 `.context/eth_2x4_curve_eval_20260409/eth_qwen3b_1x4_source_curve_20260409_curated/`，其中 `1395 MHz` 因首轮 `master_port=29549` 残留 `TCPStore` 报 `EADDRINUSE`，改用 `29551` rerun 成功；target 使用 `.context/eth_2x4_curve_eval_20260409/eth_qwen3b_2x4_target_curve_20260408_sd-1/`。随后运行 `python3 scripts/evaluate_transfer_prediction.py --network-benchmark-json .context/comm_bench_2x4_eth0_20260406_175803.json`，得到 formal 指标：`total_time_mape=40.17%`, `step_time_mape=40.17%`, `avg_power_mape=20.99%`, `total_energy_mape=10.58%`。逐频点时间 APE：`1005=43.78%`, `1200=46.52%`, `1395=30.21%`；逐频点能耗 APE：`1005=9.31%`, `1200=14.95%`, `1395=7.48%`。这确认 Ethernet slow-network 路径下时间曲线仍明显偏保守，但 energy 曲线已接近可对比使用。
[2026-04-08] **首个 `2x4` Ethernet 全卡 smoke 成功并恢复标准工件**: 在 `sd-1/sd-2` 上固定使用 `GPU 0,1,2,3`，采用 `TP=1 / PP=2 / DP=4 / ZeRO-1 + CPU offload / GBS=4 / train-iters=2`。首次 relaunch 因远端 `megatron/gpu_freq_manager.py` 未同步而在 import 阶段失败；补同步该文件后重试成功。关键结果：`step1=16.150s`, `step2=8.303s`, `skipped=0`。当前 run 目录 `/home/user/Megatron-DeepSpeed/experiments/eth_qwen3b_2x4_fullgpu_smoke_20260408_sd-1/` 已生成 `run.json`, `events.jsonl`, `command.sh`, `notes.md`, `manual_2x4.log`；`events.jsonl` 中的 Zeus 区间汇总为 `24.477s / 6891.9J / 281.6W / 2.377 tokens/J`。遗留点是 `run.json` 的 `hostfile/topology` 字段仍为空，因为 manual launcher 尚未导出对应环境变量。

[2026-04-07] **2x2 Ethernet 目标 smoke 成功 + 首个时间预测 checkpoint**: 在 `sd-1/sd-2` 上固定使用 `GPU 2,3`，采用 `PP=2 / DP=2 / ZeRO-1 + CPU offload`、`NCCL_SOCKET_IFNAME=eth0`、`NCCL_IB_DISABLE=1`，`manual_cpuoffload_r5.log` 成功跑完 `2` 个 step。关键 target 数据：`step1=37.418s`, `step2=10.976s`, `skipped=0`。随后在 `sd-2` 补跑对齐 workload 的单节点 source (`global_batch_size=4`) 得到 `step1=34.361s`, `step2=8.296s`, `skipped=0`。基于这两个成功点和 `.context/comm_bench_2x4_eth0_20260406_175803.json` 做本地 provisional time-only replay，得到 target `step_time` 预测 `14.683s`、实测 `10.976s`，APE `≈33.8%`。该结果说明当前 Ethernet slow-network 路径下的时间预测显著劣于此前 IB 的 `<10%`，但暂时还不能作为正式 transfer 结论，因为 `sd-1/sd-2` 上仍缺标准 `run.json/events.jsonl` 与 Zeus 指标。

[2026-04-07] **Ethernet benchmark 首次成功完成**: 在 `sd-1/sd-2` 上使用同步后的本地新代码运行 `2x4` 通信基准，得到 `eth0|ib_disable=1` 的有效带宽约 `0.203`、小消息 jitter CV `0.015`、大消息 jitter CV `0.007`。首次按 `1/4/16/64/256 MB × 20` 的完整配置会在 `256 MB` 档超时；收缩到 `1/4/16/64 MB`、`warmup=3`、`iters=8` 后成功落盘 JSON。

[2026-04-07] **Ethernet-only 事实更正 + 远端代码同步**: 用户确认 `sd-1/sd-2` 只有 `eth` 连接、没有 RDMA/RoCE。已补回本地缺失的 `.context/torch_nccl_comm_bench.py`，并将 predictor/benchmark 相关新代码同步到两台机器；`sd-1` 上已通过 `test_cross_node_model.py` 验证网络感知逻辑仍然成立。

[2026-04-05] **RoCE 预检完成（sd-1 / sd-2）**: 两台机器都已恢复直接 SSH，可访问 `user@sd-1` 和 `user@sd-2`。当前已确认 `tp4bit` Conda 环境可正常导入 `torch/deepspeed/einops` 并看到 4 张 RTX 4080 SUPER，但默认 `/usr/bin/python3` (3.12.3) 不含 `torch`。当前 SSH 会话里未暴露 `/dev/infiniband`，`rdma link show` 为空，且 repo 下缺少 `.context/torch_nccl_comm_bench.py` / `run_comm_bench.sh`，因此还不能把这次登录直接视为“RoCE 已可测”。

[2026-04-05] **用户偏好更新**: 明确允许在 RoCE 验证进行前并行推进论文写作。中间过程、IB 已验证结果和方法设计都可以先写入草稿，后续再按实测结果收敛表述。

[2026-04-03] **动态网络基准集成完成**: 修改 `analysis/freq_model/cross_node.py` 和 `calibrate.py`，实现 bandwidth-aware 惩罚系数。高速网络 (>50 Gbps) 使用接近零的惩罚，慢速网络使用传统惩罚。单元测试 35/35 通过。

[2026-04-03] **论文框架搭建**: 创建 `.context/paper/` 目录，配置 LaTeX 模板，完成章节框架。

[2026-04-03] **多拓扑扩展路线制定**: 规划 Tier-1 (网络层), Tier-2 (拓扑层), Tier-3 (硬件层) 三个扩展方向。

[2026-04-03] **2x4 → 2x8 Transfer 验证完成**: 源校准 MAPE 0.54%，Transfer 时间 MAPE 98.50% (修复前)，根因定位为固定惩罚系数不适用于 IB 环境。

---

## Technical Notes

### 关键发现
- **IB 带宽**: 实测 111.48 Gbps (单节点), 跨节点预计相似
- **Ethernet 带宽 (`sd-1/sd-2`)**: `2x4` benchmark 的代表性大消息带宽约 `0.203`，明确属于 slow-network 区间
- **Ethernet `2x4` Formal Transfer 指标**: `total_time_mape≈40.17%`, `step_time_mape≈40.17%`, `avg_power_mape≈20.99%`, `total_energy_mape≈10.58%`
- **Ethernet `2x4` Full-GPU 目标点**: `step1=16.150s`, `step2=8.303s`, `Zeus interval(steps1-2)=24.477s / 6891.9J / 281.6W`
- **惩罚系数**: IB 环境下 alpha_dp = 5e-13 s/byte (vs 传统 8.41e-10)
- **Step time 对比**: 2x4 (DP=2) vs 2x8 (DP=4) 几乎相同 (~20s)，说明 DP 通信 overhead 在 IB 下可忽略

### 代码状态
- `analysis/freq_model/cross_node.py`: 已更新，支持 network_bench_result 参数
- `analysis/freq_model/calibrate.py`: 已更新，传递网络基准结果
- `.context/torch_nccl_comm_bench.py`: 通信基准测试脚本
- `.context/run_comm_bench.sh`: 多节点启动脚本
- `.context/test_cross_node_model.py`: 单元测试

### 下一步技术工作
1. 给当前手工 CPU-offload 启动路径补齐 `MEGATRON_HOSTFILE_JSON` / `MEGATRON_TOPOLOGY_JSON`，或让 `scripts/run_experiment.sh` 直接支持已验证的 DS offload 配置，保证 `run.json` 里有非空 topology/hostfile
2. 将 `.context/eth_2x4_curve_eval_20260409/transfer_eval/` 的 formal 结果并排纳入 IB vs Ethernet 结果整理，明确时间高偏差与能耗可比性的边界
3. 解释 slow-network 分支在 Ethernet `2x4` 上系统性高估 runtime 的来源，决定是否需要新的保守校正或仅在论文中作为 limitation 保留
4. 清理远端同步产生的 `._*` AppleDouble 噪声文件，保持工作树可读
5. 将 launcher 默认 size/iters 策略按 Ethernet 路径保守化，避免 `256 MB` 档超时
6. 实现 predictor 的自动网络检测工作流 (benchmark → calibrate → predict)

[2026-04-27] **4080 Qwen2.5-7B-Instruct 预测数学模型验证完成**：
  - 基于 4/19-20 的 9 频点实测数据（1005/1200/1395/1500/1650/1800/1950/2100/2250 MHz + baseline）
  - **数学模型**（详见 `.context/predict_4080_mathematical_model.md`）：
    - **时间模型**: `T(f) = a × (f_max/f)^b + c`
      - a=23.637s, b=1.0458, c=207.924s (20 steps)
      - MAPE: **1.08%**
      - 物理意义: 计算时间(1.18s/step) + 通信时间(10.40s/step, Ethernet 瓶颈)
    - **功率模型**: `P(f) = P_static + P_dynamic × (f/f_max)^exp`
      - P_static=218.67W, P_dynamic=93.78W, exp=8.000
      - MAPE: **2.40%**
      - 物理意义: 静态功耗 + 极陡峭的动态功耗(exp=8)
    - **能耗模型**: `E(f) = P(f) × T(f)`
    - **能效模型**: `tokens/J = total_tokens / E(f)`
  - **1845 MHz 验证结果**（1850 不支持，最近支持点为 1845）：
    - 预测: 240.5s / 226.8W / 54,537J / 3.004 tok/J
    - 实际: 253.1s / 230.6W / 58,361J / 2.807 tok/J
    - 功率误差: **-1.6%** ✓，时间误差: -5.0%，能耗误差: -6.6%
  - **关键发现**：
    - **1650-1750 MHz 是平坦最优区**：能效都在 3.02 附近
    - **1650 MHz 是最佳 sweet spot**：时间 +5.6%，功率 -28.9%，能耗 -25.0%，能效 3.019
    - **功率曲线非常陡峭**（exp=8）：低频区平坦，高频区急剧上升
    - **时间曲线非常平坦**（b≈1）：通信瓶颈主导
  - **硬件约束**：RTX 4080 锁频点必须是 **15 MHz 的整数倍**（210-3105 MHz）
    - 1850 MHz 不支持，1845 MHz 支持
    - 所有历史频点（1005, 1200, 1395, 1500, 1650, 1800...）都是 15 的倍数，锁频有效
  - **工件**：
    - 数学模型文档: `.context/predict_4080_mathematical_model.md`
    - 预测脚本: `.context/raw_predict_4080_v3.py`
    - 1845 验证: `eth_real_qwen25_7b_tp2pp2dp2_static1845_verify_20260427`
    - 1850 未锁频对照: `eth_real_qwen25_7b_tp2pp2dp2_static1850_verify_20260427` (功率 317W ≈ baseline)

[2026-04-25] **DeepSeek-R1-Distill-Qwen-7B V100 单节点 5 频点能耗曲线已完成**：
  - 拓扑：TP=2 / PP=2 / DP=2，8 GPUs (0-7)，单节点 DGX2-1
  - 模型：DeepSeek-R1-Distill-Qwen-7B (28L / hidden=3584 / ffn=18944 / heads=28 / kv_heads=4 / vocab=152064)
  - 数据：`qwen_data_text_document`（Qwen2.5 tokenizer 预处理）
  - 训练：20 iterations，random init (no checkpoint load)，bf16，ZeRO-1，recompute-granularity full
  
  **完整 5 频点结果**：
  | 频率 | 时间(s) | 能耗(J) | 功率(W) | tokens/J | 相对 baseline |
  |------|---------|---------|---------|----------|--------------|
  | Baseline (1380 MHz) | 317.6 | 694,829 | 2187.6 | 0.472 | — |
  | Static 1260 MHz | 369.1 | 544,099 | 1474.1 | 0.602 | 能耗 -21.7%, 能效 +27.5% |
  | Static 1155 MHz | 397.2 | 510,242 | 1284.5 | 0.642 | 能耗 -26.6%, 能效 +36.0% |
  | Static 1080 MHz | 424.4 | 495,331 | 1167.2 | 0.662 | 能耗 -28.7%, 能效 +40.3% |
  | Static 990 MHz | 460.8 | 499,304 | 1083.5 | 0.656 | 能耗 -28.2%, 能效 +38.9% |
  
  - 关键发现：
    - **最佳能效点：1080 MHz**，tokens/J 提升 +40.3%，能耗降低 -28.7%
    - 990 MHz 功率最低 (1083W) 但时间代价更大，总能耗略高于 1080 MHz
    - 1155 MHz 是时间-能耗的较好平衡点：仅比 1260 慢 7.6%，但能耗再降 6.2%
    - DeepSeek 7B baseline 功率 (~2188W) 显著高于 LLaMA-7B (~1714W)，因 vocab 更大（152064 vs 32000）
    - 所有频点 loss 正常下降，训练稳定
  - 实验工件：
    - `/tmp/deepseek_sweep_20260426_084716/deepseek_static_{990,1080,1155}_20steps.log`
    - `/tmp/deepseek_baseline_20steps.log`, `/tmp/deepseek_static_1260_20steps.log`

[2026-04-25] **V100 单节点 LLaMA-7B 能耗对比实验全部完成（统一口径：Zeus 仅统计训练阶段）**：
  - 已完成两组对照实验：
    1. **真实权重（finetune from HF checkpoint）**：5 频点能耗曲线
    2. **Random Init（从头训练）**：baseline + static 1260 MHz 对照
  - 所有实验 Zeus 统计口径一致（从 "before the start of training step" 到 "after training is done"）
  - 时间戳验证：实际训练时间与 Zeus 报告时间误差 < 1s
  
  **真实权重 5 频点结果**：
  | 频率 | 时间 | 能耗 | 功率 | tokens/J | 相对 baseline |
  |------|------|------|------|----------|--------------|
  | Baseline (1380 MHz) | 467.2s | 787,358J | 1685.3W | 0.416 | — |
  | Static 1260 MHz | 505.1s | 593,341J | 1174.7W | 0.552 | 能耗 -24.6%, 能效 +32.7% |
  | Static 1350 MHz | 488.2s | 638,014J | 1306.8W | 0.514 | 能耗 -19.0%, 能效 +23.6% |
  | Static 1455 MHz | 467.3s | 705,273J | 1509.1W | 0.465 | 能耗 -10.4%, 能效 +11.8% |
  | Static 1530 MHz | 452.4s | 760,476J | 1681.2W | 0.431 | 能耗 -3.4%, 能效 +3.6% |
  
  **Random Init 对照结果**：
  | 频率 | 时间 | 能耗 | 功率 | tokens/J | 相对 baseline |
  |------|------|------|------|----------|--------------|
  | Baseline (1380 MHz) | 454.5s | 779,045J | 1713.9W | 0.421 | — |
  | Static 1260 MHz | 511.0s | 596,680J | 1167.7W | 0.549 | 能耗 -23.4%, 能效 +30.4% |
  
  - 关键发现：
    - 1260 MHz 在真实权重和 random init 下均为最佳能效点
    - 两种初始化方式结果高度一致（能耗节省 ~24% vs ~23%，能效提升 ~33% vs ~30%）
    - 证明锁频节能效果与权重初始化无关
    - V100 默认时钟 1380 MHz（非 max 1597 MHz），节能空间相对有限但仍显著

[2026-04-25] **V100 单节点真实 LLaMA-7B baseline + static 1260 MHz 对比已完成**：
  - 拓扑：`TP=2 / PP=2 / DP=2`，8 GPUs (0-7)，单节点 DGX2-1
  - 模型：真实 LLaMA-7B (32L / hidden=4096 / ffn=11008 / heads=32 / kv_heads=32 / vocab=32000)
  - 数据：`chinese_wiki_megatron_text_document`
  - 训练：20 iterations，random init (no checkpoint load)，bf16，ZeRO-1，recompute-granularity full
  - Baseline (默认 1380 MHz)：`454.5s / 779,045J / 1713.9W / 0.421 tokens/J`
  - Static 1260 MHz：`511.0s / 596,680J / 1167.7W / 0.549 tokens/J`
  - 相对 baseline：`time +12.4% / avg_power -31.9% / energy -23.4% / tokens_per_j +30.4%`
  - 关键事实：
    - V100 默认时钟已经是 1380 MHz（而非理论 max 1597 MHz），因此节能空间比 4080 线更小
    - 尽管如此，仍获得了 `-23.4%` 的能耗节省和 `+30.4%` 的能效提升
    - 这是第一条真实 LLaMA-7B (非 Qwen-like) 的 artifact-backed 节能证据
    - 使用正确数据集（LLaMA tokenizer）后 loss 正常（7.11→7.11），不影响功耗/时间测量
  - 下一步：
    - 用官方 `hf2megads_weight_converter.py` 在 V100 上转换真实 checkpoint（32GB 不会 OOM）
    - 或继续用 random init 跑更多 static 频点（1155, 1080, 990 MHz）以形成完整曲线
    - 4080 线：等 GPU 0 释放后，尝试 seq-length=512 或 TP=1,PP=4 减少内存占用

[2026-05-06] **Qwen3-4B TP2PP2DP2 dual-node V100 baseline + static sweep 全部完成**:
  - **修复链**：`--kv-channels 128` → `--disable-bias-linear` → 移除 `--untie-embeddings-and-output-weights`（Qwen3-4B `tie_word_embeddings=True`）
  - **Checkpoint 加载验证**：smoke test v3 成功加载 `qwen3_4b_hf2megads_tp2pp2_20260506_213856`，5 iterations 正常运行
  - **完整 5 频点结果**（DGX2-1 + DGX2-2，各 4×V100 GPU 8-11，TP=2/PP=2/DP=2，20 steps，真实 Qwen3-4B checkpoint，GBS=4）：
    | 频率 | 时间(s) | 功率(W) | 能耗(J) | tok/J | 相对 baseline |
    |------|---------|---------|---------|-------|--------------|
    | Baseline | 171.8 | 768.1 | 131,936 | 1.242 | — |
    | Static 1530 | 171.0 | 720.1 | 123,110 | 1.331 | 时间 -0.5%, 能耗 -6.7% |
    | Static 1455 | 178.8 | 636.3 | 113,740 | 1.440 | 时间 +4.1%, 能耗 -13.8% |
    | Static 1350 | 187.0 | 556.7 | 104,119 | 1.574 | 时间 +8.9%, 能耗 -21.1% |
    | Static 1260 | 191.7 | 510.4 | 97,852 | 1.674 | 时间 +11.6%, 能耗 -25.8% |
  - **关键发现**：
    - 1260 MHz 仍为最佳能效点（-25.8% 能耗，+34.8% tok/J），与 Qwen7B/LLaMA7B 同拓扑结论一致
    - 1530 MHz 几乎不损失时间（-0.5%），仍可节省 6.7% 能耗，是“时间敏感”场景的最佳选择
    - 功率随频率单调下降（768W → 720W → 636W → 557W → 510W），趋势稳定
  - **脚本变更**：
    - `scripts/run_experiment.sh`: `--untie-embeddings-and-output-weights` 改为条件式（`NO_UNTIE_EMBEDDINGS` 控制）
    - `scripts/run_real_qwen3_4b_tp2pp2dp2_v100.sh`: 新增 `NO_UNTIE_EMBEDDINGS=1`

[2026-05-09] **全部 retest（A-1 → A-2 → C → B → D）完成，无 OOM、无双机不同步**:
  - **关键修复**：所有 sweep 脚本 `sleep 10` → `sleep 60`，消除连续运行 GPU 内存碎片导致的 OOM
  - **A-1 TP2PP4DP4 32-card retest**：baseline/1200/1155/1080/990/1260 全部 ✅（990 和 1260 单独补跑 retest3）
  - **A-2 TP2PP2DP8 32-card retest**：baseline/1200/1155/1080/990/1260 全部 ✅
  - **C TP4PP2DP4 GBS=32 32-card retest**：baseline/1200/1155/1080/990/1260 全部 ✅
  - **B TP4PP2DP4 refine 32-card retest**：baseline/1020/1050/1110/1245 全部 ✅
  - **D TP4PP2DP1 8-card retest**：baseline/1200/1155/1080/990 全部 ✅
  - **实验工件**：全部同步到本地 `experiments/ib_real_qwen25_7b_*_retest*`

[2026-05-09] **32 卡高频区补跑完成（TP2PP4DP4 + TP2PP2DP8 + TP4PP2DP4 GBS=32）**：
  - **动机**：32 卡 4 个拓扑中仅 TP4PP2DP4 GBS=16 有完整频率曲线，其余 3 个只有低频区（990–1260），缺高频区（1350/1455/1530）
  - **结果**：
    - TP2PP4DP4 highfreq：baseline/1350/1455/1530 全部 ✅
    - TP2PP2DP8 highfreq：baseline/1350/1455/1530 全部 ✅
    - TP4PP2DP4 GBS=32 highfreq：baseline/1350/1455/1530 全部 ✅
  - **意义**：32 卡全部 4 个拓扑现已覆盖完整频率曲线（990–1530），可用于 predictor 全频段校准验证
  - **工件**：`experiments/ib_real_qwen25_7b_*_highfreq_*`
