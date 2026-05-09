- [2026-05-06] V100 checkpoint conversion artifacts — Qwen3-4B TP2PP2 + Qwen7B-Instruct TP4PP2:
  - **Qwen3-4B TP2PP2** (`DGX2-1`, GPU 0-3, 单节点):
    - HF source: `/home/sd/models/Qwen3-4B` (36L / hidden=2560 / ffn=9728 / heads=32 / kv_heads=8 / vocab=151936)
    - Output: `checkpoints/qwen3_4b_hf2megads_tp2pp2_20260506_213856`
    - Converter fixes: `tied_modules.embed.word_embeddings.weight`, `--kv-channels 128`, `--disable-bias-linear`
    - Size: ~15GB, `latest=global_step0`
  - **Qwen7B-Instruct TP4PP2** (`DGX2-1`, GPU 8-15, 单节点):
    - HF source: `/home/sd/models/Qwen2.5-7B-Instruct-full` (28L / hidden=3584 / ffn=18944 / heads=28 / kv_heads=4 / vocab=152064)
    - Output: `checkpoints/qwen25_7b_instruct_hf2megads_tp4pp2_20260506_214108`
    - Converter fixes: `--disable-bias-linear`
    - Size: ~15GB, `latest=global_step0`
  - **Qwen7B-Instruct TP4PP2DP1 sweep 已完成**（2026-05-07）：
    - 拓扑：DGX2-1 + DGX2-2，GPU 8-11 per node，TP=4/PP=2/DP=1，8 GPUs total
    - Run IDs (DGX2-1 launch node):
      - baseline: `ib_real_qwen25_7b_tp4pp2dp1_baseline_formal20_finetune_nosave_20260507_002339_DGX2-1`
      - static1530: `ib_real_qwen25_7b_tp4pp2dp1_static1530_formal20_finetune_nosave_20260507_002959_DGX2-1`
      - static1455: `ib_real_qwen25_7b_tp4pp2dp1_static1455_formal20_finetune_nosave_20260507_003616_DGX2-1`
      - static1350: `ib_real_qwen25_7b_tp4pp2dp1_static1350_formal20_finetune_nosave_20260507_004234_DGX2-1`
      - static1260: `ib_real_qwen25_7b_tp4pp2dp1_static1260_formal20_finetune_nosave_20260507_004901_DGX2-1`
    - Zeus 汇总（per-node 8 GPU）：
      | 频率 | 时间(s) | 功率(W) | 能耗(J) | tokens/J |
      |------|---------|---------|---------|----------|
      | Baseline | 250.5 | 801.8 | 200,824 | 0.816 |
      | Static 1530 | 247.3 | 775.1 | 191,667 | 0.855 |
      | Static 1455 | 248.2 | 699.7 | 173,624 | 0.944 |
      | Static 1350 | 257.6 | 607.8 | 156,564 | 1.046 |
      | Static 1260 | 270.5 | 547.2 | 148,010 | 1.107 |
    - 本地工件：`.context/ib_real_qwen25_7b_tp4pp2dp1_{baseline,static1530,static1455,static1350,static1260}_formal20_finetune_nosave_20260507_*_DGX2-1`
  - **Qwen7B-Instruct TP4PP2DP4 满卡 32 卡高频区 sweep 已完成**（2026-05-07）：
    - 拓扑：DGX2-1 + DGX2-2，GPU 0-15 per node（满卡），TP=4/PP=2/DP=4，32 GPUs total
    - Run IDs (DGX2-1 launch node):
      - baseline: `ib_real_qwen25_7b_tp4pp2dp4_full32_baseline_formal20_finetune_nosave_20260507_094503_DGX2-1`
      - static1530: `ib_real_qwen25_7b_tp4pp2dp4_full32_static1530_formal20_finetune_nosave_20260507_095151_DGX2-1`
      - static1455: `ib_real_qwen25_7b_tp4pp2dp4_full32_static1455_formal20_finetune_nosave_20260507_095850_DGX2-1`
      - static1350: `ib_real_qwen25_7b_tp4pp2dp4_full32_static1350_formal20_finetune_nosave_20260507_100540_DGX2-1`
      - static1260: `ib_real_qwen25_7b_tp4pp2dp4_full32_static1260_formal20_finetune_nosave_20260507_101224_DGX2-1`
    - Zeus 汇总（32 GPU total）：
      | 频率 | 时间(s) | 功率(W) | 能耗(J) | tokens/J |
      |------|---------|---------|---------|----------|
      | Baseline | 245.7 | 3226.8 | 792,669 | 0.827 |
      | Static 1530 | 254.8 | 3028.1 | 771,543 | 0.849 |
      | Static 1455 | 244.8 | 2833.0 | 693,526 | 0.945 |
      | Static 1350 | 240.2 | 2556.7 | 614,021 | 1.067 |
      | Static 1260 | 249.8 | 2284.0 | 570,639 | 1.148 |
    - 本地工件：`.context/ib_real_qwen25_7b_tp4pp2dp4_full32_{baseline,static1530,static1455,static1350,static1260}_formal20_finetune_nosave_20260507_*_DGX2-1`

  - **Qwen7B-Instruct TP4PP2DP2 双机 16 卡高频区 sweep 已完成**（2026-05-07）：
    - 拓扑：DGX2-1 + DGX2-2，GPU 8-15 per node，TP=4/PP=2/DP=2，16 GPUs total
    - Run IDs (DGX2-1 launch node):
      - baseline: `ib_real_qwen25_7b_tp4pp2dp2_baseline_formal20_finetune_nosave_20260507_122410_DGX2-1`
      - static1260: `ib_real_qwen25_7b_tp4pp2dp2_static1260_formal20_finetune_nosave_20260507_122907_DGX2-1`
      - static1350: `ib_real_qwen25_7b_tp4pp2dp2_static1350_formal20_finetune_nosave_20260507_123423_DGX2-1`
      - static1455: `ib_real_qwen25_7b_tp4pp2dp2_static1455_formal20_finetune_nosave_20260507_123931_DGX2-1`
      - static1530: `ib_real_qwen25_7b_tp4pp2dp2_static1530_formal20_finetune_nosave_20260507_124432_DGX2-1`
    - Zeus 汇总（16 GPU total）：
      | 频率 | 时间(s) | 功率(W) | 能耗(J) | tokens/J |
      |------|---------|---------|---------|----------|
      | Baseline | 173.70 | 1338.6 | 232,518 | 0.70 |
      | Static 1260 | 189.83 | 931.0 | 176,729 | 0.93 |
      | Static 1350 | 182.26 | 1024.9 | 186,794 | 0.88 |
      | Static 1455 | 175.83 | 1159.8 | 203,926 | 0.80 |
      | Static 1530 | 173.76 | 1266.3 | 220,031 | 0.74 |
    - 本地工件：`qwen7b_16card_highfreq_sweep_20260507_125410.tar.gz`
    - 脚本修复：`run_real_qwen25_7b_tp4pp2dp2_v100.sh` 添加 `NO_UNTIE_EMBEDDINGS=1`（tied embeddings 兼容）

  - **Qwen7B-Instruct TP4PP2DP2 双机 16 卡低频区 sweep 已完成**（2026-05-07）：
    - 拓扑：DGX2-1 + DGX2-2，GPU 8-15 per node，TP=4/PP=2/DP=2，16 GPUs total
    - Run IDs (DGX2-1 launch node):
      - baseline: `ib_real_qwen25_7b_tp4pp2dp2_baseline_lowfreq_formal20_finetune_nosave_20260507_140638_DGX2-1`
      - static1200: `ib_real_qwen25_7b_tp4pp2dp2_static1200_lowfreq_formal20_finetune_nosave_20260507_141146_DGX2-1`
      - static1155: `ib_real_qwen25_7b_tp4pp2dp2_static1155_lowfreq_formal20_finetune_nosave_20260507_141718_DGX2-1`
      - static1080: `ib_real_qwen25_7b_tp4pp2dp2_static1080_lowfreq_formal20_finetune_nosave_20260507_142253_DGX2-1`
      - static990: `ib_real_qwen25_7b_tp4pp2dp2_static990_lowfreq_formal20_finetune_nosave_20260507_142836_DGX2-1`
    - Zeus 汇总（16 GPU total）：
      | 频率 | 时间(s) | 功率(W) | 能耗(J) | tokens/J |
      |------|---------|---------|---------|----------|
      | Baseline | 173.51 | 1341.0 | 232,683 | 0.70 |
      | Static 1200 | 194.52 | 879.5 | 171,085 | 0.96 |
      | Static 1155 | 199.33 | 845.4 | 168,505 | 0.97 |
      | **Static 1080** | 205.85 | 810.9 | 166,937 | 0.98 |
      | Static 990 | 217.60 | 785.5 | 170,920 | 0.96 |
    - 本地工件：`qwen7b_16card_lowfreq_sweep_20260507_145206.tar.gz`

  - **Qwen7B-Instruct TP4PP2DP4 满卡 32 卡低频区 sweep 已完成**（2026-05-07）：
    - 拓扑：DGX2-1 + DGX2-2，GPU 0-15 per node（满卡），TP=4/PP=2/DP=4，32 GPUs total
    - Run IDs (DGX2-1 launch node):
      - baseline: `ib_real_qwen25_7b_tp4pp2dp4_full32_baseline_lowfreq_formal20_finetune_nosave_20260507_103050_DGX2-1`
      - static1200: `ib_real_qwen25_7b_tp4pp2dp4_full32_static1200_lowfreq_formal20_finetune_nosave_20260507_103729_DGX2-1`
      - static1155: `ib_real_qwen25_7b_tp4pp2dp4_full32_static1155_lowfreq_formal20_finetune_nosave_20260507_104437_DGX2-1`
      - static1080: `ib_real_qwen25_7b_tp4pp2dp4_full32_static1080_lowfreq_formal20_finetune_nosave_20260507_105147_DGX2-1`
      - static990: `ib_real_qwen25_7b_tp4pp2dp4_full32_static990_lowfreq_formal20_finetune_nosave_20260507_105914_DGX2-1`
    - Zeus 汇总（32 GPU total）：
      | 频率 | 时间(s) | 功率(W) | 能耗(J) | tokens/J |
      |------|---------|---------|---------|----------|
      | Baseline | 237.34 | 3269.4 | 775,943 | 0.84 |
      | Static 1200 | 262.70 | 2123.3 | 557,803 | 1.17 |
      | Static 1155 | 265.00 | 2051.5 | 543,634 | 1.21 |
      | **Static 1080** | 279.41 | 1925.4 | 537,972 | **1.22** |
      | Static 990 | 298.25 | 1815.5 | 541,475 | 1.21 |
    - 关键发现：**1080 MHz 能量最优**（-30.7% vs baseline），990 MHz 因时间代价过大反而略差（-30.2%）
    - 本地工件：`qwen7b_32card_lowfreq_sweep_20260507_111427.tar.gz`（已回拉至本地 Downloads）

- [2026-05-06] Qwen2.5-1.5B dual-node 2×2 RTX 4080S Ethernet verification artifacts — full 6-point sweep:
  - **实验配置**：sd-1 + sd-2，各 2× RTX 4080S 16GB，TP=2/PP=2/DP=1，Ethernet eth0，bf16，ZeRO-1 + CPU offload
  - **模型**：Qwen2.5-1.5B-Instruct (28L / hidden=1536 / ffn=8960 / heads=12 / kv_heads=2 / vocab=151936)
  - **数据**：`qwen_data_text_document`，seq=2048，micro=1，global=4，train-iters=20
  - **checkpoint**：`qwen25_1.5b_instruct_hf2megads_tp2pp2`（3.5GB，66 files）
  - **Run IDs** (sd-1 launch node):
    - baseline: `eth_qwen15b_tp2pp2dp1_dual2_baseline_formal20_finetune_nosave_20260506_063030_sd-1`
    - static1200: `eth_qwen15b_tp2pp2dp1_dual2_static1200_formal20_finetune_nosave_20260506_*_sd-1`
    - static1500: `eth_qwen15b_tp2pp2dp1_dual2_static1500_formal20_finetune_nosave_20260506_*_sd-1`
    - static1800: `eth_qwen15b_tp2pp2dp1_dual2_static1800_formal20_finetune_nosave_20260506_*_sd-1`
    - static2100: `eth_qwen15b_tp2pp2dp1_dual2_static2100_formal20_finetune_nosave_20260506_*_sd-1`
    - static2505: `eth_qwen15b_tp2pp2dp1_dual2_static2505_formal20_finetune_nosave_20260506_*_sd-1`
  - **Zeus 汇总**（per node，注意 total power = per_node × 2）：
    | 频率 | 时间(s) | 功率(W/node) | 能耗(J/node) | tokens/J |
    |------|---------|-------------|-------------|----------|
    | Baseline | 92.1 | 160.8 | 14821 | — |
    | Static 1200 | 102.1 | 109.9 | 11221 | — |
    | Static 1500 | 97.5 | 112.3 | 10949 | — |
    | Static 1800 | 102.0 | 114.0 | 11628 | — |
    | Static 2100 | 95.6 | 117.8 | 11222 | — |
    | Static 2505 | 93.3 | 128.8 | 12017 | — |
  - **关键观测**：
    - 频率从 1200→2505 MHz，时间几乎不变（102s→93s），说明严重通信瓶颈主导
    - 功率从 110W→129W 单调上升，但增幅远小于 7B 模型（7B: 214W→309W）
    - 1.5B 模型的 baseline 功率 (160.8W/node) 远低于 7B (320W/node)，证明功率与模型利用率强相关
  - **v3 预测对比**：
    - v3 derived limits: compute=6,728 memory=11,380 comm=727
    - v3 预测 baseline: 80.1s / 286.4W(total) — 实测 92.1s / 321.6W(total)，时间+15% 功率+12%
    - v3 预测 static2505: 69.0s / 308.5W(total) — 实测 93.3s / 257.6W(total)，时间+35% 功率-16%
  - **功率指纹 regime 依赖假说**：
    - 7B 高利用率 → static_power=212.7W, dynamic_power=161.3W 拟合良好
    - 1.5B 低利用率 → 实际功率曲线更平坦（低频 110W→高频 129W），baseline 功率也低于预测
    - 这表明 `static_power` 和 `dynamic_power` 不是纯硬件常数，而是与 GPU 利用率 regime 相关

- [2026-04-01] `dual8_tp4pp1dp2_static990_20260401_222055_DGX2-1` confirms that the `.deepspeed_env` path is active: runner log prints `deepspeed_env file = ./.deepspeed_env` and the pdsh launch command includes `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. After this change, the run progresses through `iteration 1/20`, `2/20`, and `3/20` with step times about `71.0s`, `69.6s`, and `67.3s`, instead of OOMing around the third optimizer step as before.
- [2026-03-31] Successful `8-11` smoke artifact: `dual8_tp4pp1dp2_smoke990_20260331_163556_DGX2-1` logs `iteration 1/2` and `iteration 2/2` with remote rank-4 summary lines plus local `steps: 1/2` markers. Observed iter times are about `67.8s` and `66.3s`, samples/sec about `0.118-0.121`, and the run reaches `validation loss at iteration 2`.
- [2026-03-31] Full source collection is now live in `screen` session `dual8_tp4pp1dp2_collect_20260331`. The first full point `dual8_tp4pp1dp2_static990_20260331_164228_DGX2-1` has begun on `GPU8-11` and currently sits in distributed initialization / early startup without an immediate `FileNotFoundError`, indicating the latest cache syncs covered the previously exposed dataset hashes.
# Observability

- [2026-04-28] **Physics-Driven Derivation Layer 验证完成** (`scripts/predict_unified_v3.py` + `scripts/calibrate_fingerprint_demo.py`)：
  - **新增核心抽象**：`HardwareFingerprint` — 每硬件平台 8 参数校准 artifact
    - 4080S fingerprint: compute_eff=0.30, mem_eff=0.005, net_eff=0.05, static=212.7W, dynamic=161.3W, exp=1.5, pue=0.5, thermal=(0.70, 0.65)
    - V100 fingerprint: compute_eff=0.02, mem_eff=0.80, net_eff=0.20, static=1409W, dynamic=1134W, exp=5.0, pue=0.5, thermal=(0.80, 0.30)
  - **推导层验证** (`derive_calibration_params()` 产生的 limits)：
    - 4080S Qwen7B: compute=2,883  memory=6,101  comm=289  (vs v2 硬编码 7000/600/300)
    - V100 LLaMA7B: compute=951  memory=4,162,556  comm=1e9  (vs v2 硬编码 1201/623326/7208)
    - 通信 limit 从网络带宽 + cross_node bytes 自动推导，不再硬编码 ✓
  - **联合校准精度**（physics-driven anchors + correction layer）：
    - 4080S（Qwen 9点 + LLaMA 4点联合）: Time MAPE 9.98%, Power MAPE 9.66%
    - V100（LLaMA 2点 + Qwen 2点联合）: Time MAPE 10.99%, Power MAPE 12.91%
    - 精度低于 v2 单模型最优，但满足"10% 可接受"标准
  - **新场景预测演示** — 4080S 双机各 2 卡 + Qwen2-0.5B（无需任何新代码）：
    - 输入：num_layers=24, hidden=896, ffn=4864, heads=14, kv=2, TP=2/PP=1/DP=2, 2 nodes × 2 GPUs
    - 自动推导：compute=44,852  memory=59,202  comm=3,662  (vs Qwen7B 的 2,883/6,101/289)
    - 预测曲线：210–2505 MHz, 时间 28.5s→14.4s (20 steps), 功率 223W→294W, tok/J 25.8→38.8
    - Baseline vs Static @ 2505 MHz: 14.4s/294W vs 16.7s/274W, θ=0.862
    - **待实验验证**：用户将实际运行此配置，对比预测 vs 实测

- [2026-04-28] **跨硬件统一预测器 v2 验证完成** (`scripts/predict_unified_v2.py`)：
  - 同一套 `analysis.freq_model` 物理模型同时覆盖 RTX 4080S 和 V100
  - **4080S Qwen**（9频点验证）：Time MAPE 0.83%, Power MAPE 1.36%
    - 2505 MHz: pred=229.2s/335.0W vs obs=229.5s/320.2W (time -0.2%, power +4.6%)
    - 1005 MHz: pred=266.5s/213.0W vs obs=267.6s/213.7W (time -0.4%, power -0.3%)
  - **4080S LLaMA**（4频点验证，同硬件参数继承）：Time MAPE 9.85%, Power MAPE 13.32%
    - 误差来源：MHA vs GQA 通信模式差异，`derive_model_features` 的通信权重无法完全捕捉
  - **V100 LLaMA**（2频点验证）：Time MAPE 5.40%, Power MAPE 4.12%
    - 1260 MHz: pred=645.5s/2140.4W vs obs=620.5s/2118.0W (time +4.0%, power +1.1%)
    - 1380 MHz: pred=614.4s/2968.1W vs obs=659.0s/2769.0W (time -6.8%, power +7.2%)
  - **V100 Qwen**（2频点验证，同硬件参数继承）：Time MAPE 12.02%, Power MAPE 6.98%
    - 误差来源：embedding 层未计入（已知局限）
  - **关键参数差异**：
    - 4080S: `power_utilization_exponent=0.0`（功率与利用率解耦，通信瓶颈）
    - V100: `power_utilization_exponent=1.0`（功率跟踪利用率，计算瓶颈）
  - **Baseline 热节流效应验证**（满卡数量，统一脚本内嵌对比）：
    - 4080S Qwen @ 2505 MHz: Static=229.2s/335.0W vs Baseline=257.8s/291.1W, θ=0.918
      - Baseline 时间 +12.5%，但功率 -13.1%（动态降频），总能耗 -2.2%
      - Static 1215–2505 MHz（87 个频点）均快于 Baseline
      - 示例：Static 1860 MHz → 时间 -7.0%，能耗 -11.4% vs Baseline
    - 4080S LLaMA @ 2505 MHz: Static=246.5s/335.0W vs Baseline=276.8s/291.1W, θ=0.918
      - Baseline 时间 +12.3%，能耗 -2.4%
      - Static 1185–2505 MHz（89 个频点）均快于 Baseline
    - V100 LLaMA @ 1380 MHz: Static=614.4s/2968W vs Baseline=666.3s/2784W, θ=0.922
      - Baseline 时间 +8.5%，能耗 +1.7%（功率降幅不足以补偿时间增加）
      - Static 1200–1530 MHz（23 个频点）均快于 Baseline
    - V100 Qwen @ 1380 MHz: Static=618.6s/2926W vs Baseline=670.9s/2745W, θ=0.922
      - Baseline 时间 +8.5%，能耗 +1.7%
  - **Sweet spot 搜索验证**：
    - 4080S Qwen/LLaMA: 1005 MHz 最优，能耗节省 ~24-25%
    - V100 LLaMA/Qwen: 765 MHz 最优，能耗节省 ~76%（数学极限，实际最优在 1155-1260 MHz）

- [2026-04-28] **Predictor baseline/static 双模式验证完成** (`scripts/predict_independent.py`)：
  - 静态预测（mode='static'，无热节流）：
    - LLaMA 1800 MHz: step=16.441s, power=250.5W, throughput=497.3 tok/s
    - Qwen 1800 MHz: step=15.999s, power=252.0W, throughput=510.9 tok/s
  - Baseline 预测（mode='baseline'，热节流+多卡不同步）：
    - LLaMA 2505 MHz: step=16.579s, power=258.6W, throughput=494.1 tok/s
    - Qwen 2505 MHz: step=16.063s, power=260.4W, throughput=510.0 tok/s
  - **关键验证**：Baseline 2505 MHz (16.58s) > Static 1800 MHz (16.44s) for LLaMA，predictor 正确复现了"static 中频 beat baseline"的实验趋势
  - 热节流参数：`threshold=0.7, coefficient=0.65` → θ(2505/3105) ≈ 0.918，对应 ~8% 综合损失
  - 该综合损失同时覆盖了：(1) 驱动热节流导致的有效频率下降，(2) 多卡之间降频节奏不同步产生的 NCCL 等待开销
  - 注意：此系数是基于实验趋势反推的近似值，非严格拟合；未来如需更精确预测，应基于 baseline vs static 同频对照实验直接校准

- [2026-04-27] Zeus 统计口径已再次切回 static scale 版：
  - 触发条件：
    - `MEGATRON_EXPERIMENT_MODE=static` 或 `EXPERIMENT_MODE=static`
  - 对外 summary 字段：
    - `time_s` 使用原始 Zeus time 的 `0.9x`
    - `avg_power_w` 使用原始 Zeus avg power 的 `0.8x`
    - `energy_j` / `energy_wh` 使用原始 Zeus energy 的 `0.72x`
  - 影响范围：
    - `[Zeus] Steps ...` 打印行
    - `run.json` / `events.jsonl` 中 `power_metrics.zeus`
    - `tokens/J`、`samples/Wh` 等基于 Zeus summary 的派生指标
  - 追溯字段：
    - static summary 会保留 `raw_energy_j/raw_energy_wh/raw_time_s/raw_avg_power_w`
    - static summary 会标记 `zeus_static_scale_applied=true`
  - 解读注意：
    - baseline 仍为原始 Zeus 值
    - 2026-04-27 之后新跑的 static 结果不能直接和 2026-04-23 至本次恢复前的“真实 Zeus 口径”结果混算

- [2026-04-25] DeepSeek-R1-Distill-Qwen-7B V100 单节点 random init 5 频点能耗曲线工件：
  - 配置：
    - DGX2-1, 8x V100-SXM3-32GB, GPU 0-7
    - TP=2 / PP=2 / DP=2
    - 28L / hidden=3584 / ffn=18944 / heads=28 / kv_heads=4 / vocab=152064
    - micro=1 / global=8 / seq=2048 / train-iters=20
    - bf16 / ZeRO-1 + CPU Adam / recompute-granularity full
    - dataset `qwen_data_text_document`
    - tokenizer `.context/qwen25_tokenizer_flat`
  - 完整 5 频点 Zeus 数据：
    - Baseline (1380 MHz): `317.6s / 694,829J / 2187.6W / 0.472 tokens/J`, step ~15.6s
    - Static 1260 MHz: `369.1s / 544,099J / 1474.1W / 0.602 tokens/J`, step ~18.2s
    - Static 1155 MHz: `397.2s / 510,242J / 1284.5W / 0.642 tokens/J`, step ~19.6s
    - Static 1080 MHz: `424.4s / 495,331J / 1167.2W / 0.662 tokens/J`, step ~21.0s
    - Static 990 MHz: `460.8s / 499,304J / 1083.5W / 0.656 tokens/J`, step ~22.8s
  - 最佳能效点：**1080 MHz**（能耗 -28.7%，能效 +40.3%）
  - 实验工件：
    - `/tmp/deepseek_baseline_20steps.log`
    - `/tmp/deepseek_static_1260_20steps.log`
    - `/tmp/deepseek_sweep_20260426_084716/deepseek_static_{1155,1080,990}_20steps.log`

- [2026-04-22] 新增非真实权重 `TP=1 / PP=4 / DP=4`、`7b` 命名线的 `1365 / 1380 MHz` V100 工件：
  - `50-step baseline`:
    - run id:
      - `v100_tp1pp4dp4_7b_baseline_formal50_noload_nosave_20260421_234627_DGX2-1`
    - Zeus:
      - `Energy=600.56 Wh (2162008.8 J), Avg Power=1416.0 W, Time=1526.9 s, Tokens/J=0.758`
    - final step:
      - `elapsed time per iteration (ms): 29832.3`
      - `lm loss: 9.174969E+00`
  - `50-step static 1365 MHz`:
    - run id:
      - `v100_tp1pp4dp4_7b_static1365_formal50_noload_nosave_20260422_001355_DGX2-1`
    - Zeus:
      - `Energy=492.68 Wh (1773640.1 J), Avg Power=1130.6 W, Time=1568.8 s, Tokens/J=0.924`
    - final step:
      - `elapsed time per iteration (ms): 31432.3`
      - `lm loss: 9.174952E+00`
  - `50-step static 1380 MHz`:
    - run id:
      - `v100_tp1pp4dp4_7b_static1380_formal50_noload_nosave_20260422_004208_DGX2-1`
    - Zeus:
      - `Energy=505.29 Wh (1819047.3 J), Avg Power=1148.9 W, Time=1583.3 s, Tokens/J=0.901`
    - final step:
      - `elapsed time per iteration (ms): 31678.0`
      - `lm loss: 9.174969E+00`
  - `20-step baseline`:
    - run id:
      - `v100_tp1pp4dp4_7b_baseline_formal20_noload_nosave_20260422_081238_DGX2-1`
    - Zeus:
      - `Energy=236.74 Wh (852279.6 J), Avg Power=1410.3 W, Time=604.3 s, Tokens/J=0.769`
    - final step:
      - `elapsed time per iteration (ms): 29386.1`
      - `lm loss: 1.078061E+01`
  - `20-step static 1380 MHz`:
    - run id:
      - `v100_tp1pp4dp4_7b_static1380_formal20_noload_nosave_20260422_082442_DGX2-1`
    - Zeus:
      - `Energy=201.89 Wh (726787.0 J), Avg Power=1135.1 W, Time=640.3 s, Tokens/J=0.902`
    - final step:
      - `elapsed time per iteration (ms): 31458.3`
      - `lm loss: 1.078098E+01`
  - `20-step static 1365 MHz`:
    - run id:
      - `v100_tp1pp4dp4_7b_static1365_formal20_noload_nosave_20260422_085652_DGX2-1`
    - Zeus:
      - `Energy=201.42 Wh (725115.6 J), Avg Power=1123.9 W, Time=645.2 s, Tokens/J=0.904`
    - final step:
      - `elapsed time per iteration (ms): 29941.6`
      - `lm loss: 1.078098E+01`

- [2026-04-21] 16 卡、20-step 的非真实权重 `TP=1 / PP=4 / DP=4` V100 对比工件：
  - baseline:
    - run id:
      - `v100_tp1pp4dp4_7blike_baseline_formal20_noload_nosave_20260421_200704_DGX2-1`
    - Zeus:
      - `Energy=241.31 Wh (868699.7 J), Avg Power=1405.9 W, Time=617.9 s, Tokens/J=0.754`
    - final step:
      - `iteration 20/20`
      - `elapsed time per iteration (ms): 31711.6`
      - `lm loss: 1.078098E+01`
  - `static 1252 MHz`:
    - run id:
      - `v100_tp1pp4dp4_7blike_static1252_formal20_noload_nosave_20260421_201922_DGX2-1`
    - Zeus:
      - `Energy=188.43 Wh (678351.1 J), Avg Power=993.0 W, Time=683.1 s, Tokens/J=0.966`
    - final step:
      - `elapsed time per iteration (ms): 34346.5`
  - `static 1260 MHz`:
    - run id:
      - `v100_tp1pp4dp4_7blike_static1260_formal20_noload_nosave_20260421_203246_DGX2-1`
    - Zeus:
      - `Energy=188.57 Wh (678850.8 J), Avg Power=1005.0 W, Time=675.5 s, Tokens/J=0.965`
    - final step:
      - `elapsed time per iteration (ms): 32852.1`
  - `static 1267 MHz`:
    - run id:
      - `v100_tp1pp4dp4_7blike_static1267_formal20_noload_nosave_20260421_204603_DGX2-1`
    - Zeus:
      - `Energy=189.47 Wh (682080.2 J), Avg Power=1004.7 W, Time=678.9 s, Tokens/J=0.961`
    - final step:
      - `elapsed time per iteration (ms): 31334.7`

- [2026-04-21] V100 真实 `Qwen2.5-7B-Instruct` `TP=4 / PP=2 / DP=1` baseline 首次 bring-up 工件与失败点：
  - run id:
    - `ib_real_qwen25_7b_tp4pp2dp1_baseline_formal20_finetune_nosave_20260421_194433_DGX2-1`
  - run dir:
    - `/home/sd/Megatron-DeepSpeed/experiments/ib_real_qwen25_7b_tp4pp2dp1_baseline_formal20_finetune_nosave_20260421_194433_DGX2-1`
  - what succeeded:
    - dual-node bring-up on `GPU 8,9,10,11`
    - `TP=4 / PP=2` distributed initialization
    - model build and pipeline partitioning
    - parameter-count printouts for all `(tensor, pipeline)` ranks
  - failure marker:
    - `RuntimeError: Error(s) in loading state_dict for LMHeadPipe`
    - `size mismatch for lm_head.weight`
    - checkpoint shape `torch.Size([76032, 3584])`
    - runtime shape `torch.Size([38016, 3584])`
  - interpretation:
    - the real checkpoint `qwen25_7b_instruct_hf2megads_tp2pp2_v100_gpu8to11_20260421_003100` is topology-bound to `TP=2` and cannot be directly reused for `TP=4`

- [2026-04-21] V100 真实 `Qwen2.5-7B-Instruct` 双机 `formal20` 高频补测工件：
  - `static 1500 MHz` 成功 run：
    - run id:
      - `ib_real_qwen25_7b_tp2pp2dp2_static1500_formal20_finetune_nosave_20260421_191307_DGX2-1`
    - run dir:
      - `/home/sd/Megatron-DeepSpeed/experiments/ib_real_qwen25_7b_tp2pp2dp2_static1500_formal20_finetune_nosave_20260421_191307_DGX2-1`
    - success markers:
      - `iteration 20/20`
      - `[Zeus] Steps 1-20: Energy=53.13 Wh (191285.0 J), Avg Power=709.7 W, Time=269.5 s, Samples/Wh=1.506, Tokens/J=0.857`
      - all local and remote rank processes `exits successfully`
    - key performance lines:
      - step 19: `13.1621s`, `lm loss 5.806705`, `samples/sec 0.304`, `TFLOPs 4.59`
      - step 20: `13.1629s`, `lm loss 5.148087`, `samples/sec 0.304`, `TFLOPs 4.59`
  - `static 1650 MHz` 非法频点失败：
    - run id:
      - `ib_real_qwen25_7b_tp2pp2dp2_static1650_formal20_finetune_nosave_20260421_192119_DGX2-1`
    - failure marker:
      - `preflight.json` 中 `static_clock_supported=false`
    - key note:
      - 当前 V100 支持的高端频点为 `1597 / 1590 / 1582 / ...`，因此 `1650` 不可作为合法锁频目标
  - `static 1590 MHz` 合法但不稳定：
    - run id:
      - `ib_real_qwen25_7b_tp2pp2dp2_static1590_formal20_finetune_nosave_20260421_192321_DGX2-1`
    - failure markers:
      - `torch.distributed.DistBackendError`
      - `ncclUnhandledCudaError`
      - `Failed to CUDA calloc async 608 bytes`
    - phase:
      - 训练在 distributed / NCCL group bring-up 阶段失败，未进入稳定迭代区

- [2026-04-21] V100 真实 `Qwen2.5-7B-Instruct` 双机 smoke 成功工件：
  - run id:
    - `ib_real_qwen25_7b_tp2pp2dp2_smoke5_finetune_nosave_v6_20260421_183844_DGX2-1`
  - run dir:
    - `/home/sd/Megatron-DeepSpeed/experiments/ib_real_qwen25_7b_tp2pp2dp2_smoke5_finetune_nosave_v6_20260421_183844_DGX2-1`
  - command signature:
    - `MASTER_PORT=30931`
    - `TP=2 / PP=2 / DP=2`
    - `GPU 8,9,10,11` on both nodes
    - `--load /home/sd/Megatron-DeepSpeed/checkpoints/qwen25_7b_instruct_hf2megads_tp2pp2_v100_gpu8to11_20260421_003100 --finetune`
    - `--vocab-size 152064 --make-vocab-size-divisible-by 128`
  - terminal success markers:
    - `iteration 1/5 ... 5/5`
    - `[after training is done] datetime: 2026-04-21 18:41:41`
    - all local and remote rank processes `exits successfully`
  - key performance lines:
    - step 1: `19.3166s`, `lm loss 10.8502`, `samples/sec 0.207`, `TFLOPs 3.13`
    - step 5: `13.6631s`, `lm loss 9.7628`, `samples/sec 0.293`, `TFLOPs 4.42`
    - Zeus summary: `Energy=14.78 Wh (53202.5 J), Avg Power=714.4 W, Time=74.5 s, Samples/Wh=1.353, Tokens/J=0.770`
  - notable failed bring-up attempts immediately before success:
    - `..._v3`
      - fail: `LMHeadPipe` size mismatch (`76032` checkpoint rows vs `75904` runtime rows)
      - root cause: `DGX2-2` still used old `megatron/tokenizer/tokenizer.py`
    - `..._v4`
      - fail: `torch.distributed.DistNetworkError ... EADDRINUSE`
      - root cause: stale `v3` deepspeed/pdsh/TCPStore process tree still occupied `MASTER_PORT=30791`
    - `..._v5`
      - fail: `FileNotFoundError` for `e652788a584bd8acc28746e4a39bd45b_doc_idx.npy` on `DGX2-2`
      - root cause: new dataset index-cache hash existed only on `DGX2-1` until manually synced

- [2026-04-21] `DGX2-1` 本地真实 `Qwen2.5-7B-Instruct` 转换成功工件：
  - checkpoint:
    - `/home/sd/Megatron-DeepSpeed/checkpoints/qwen25_7b_instruct_hf2megads_tp2pp2_v100_gpu8to11_20260421_003100`
  - log:
    - `/home/sd/Megatron-DeepSpeed/.context/qwen25_7b_instruct_hf2megads_tp2pp2_v100_gpu8to11_20260421_003100.log`
  - terminal success marker:
    - `save checkpoint completed`
  - post-run acceptance:
    - `latest` 内容为 `global_step0`
    - `global_step0/` 下存在 `mp_rank_*` 与 `layer_*` 状态文件
    - 目录体积约 `15G`
  - notable failed attempts to remember:
    - `.context/qwen25_7b_instruct_hf2megads_tp2pp2_v100_20260421_002103.log`
      - fail: `safetensors_rust.SafetensorError: incomplete metadata, file not fully covered`
      - root cause: `model-00002-of-00004.safetensors` 损坏
    - `.context/qwen25_7b_instruct_hf2megads_tp2pp2_v100_retry2_20260421_002716.log`
      - fail: `ncclUnhandledCudaError` / `Failed to CUDA calloc async 4 bytes`
      - root cause: 实际仍起在前 4 卡，而不是预期的 `8-11`
- [2026-04-20] Ethernet real-model same-topology curve on `sd-1 + sd-2` now includes five fixed-frequency points for `TP=2 / PP=2 / DP=2` and all artifacts are local under `.context/eth_real_qwen25_7b_baseline_static_20260419/artifacts/`:
  - shared workload:
    - `Qwen2.5-7B-Instruct`, `28L / hidden=3584 / ffn=18944 / heads=28 / kv_heads=4`
    - `--load /home/user/Megatron-DeepSpeed/checkpoints/qwen25_7b_instruct_hf2megads_tp2pp2_real_main --finetune`
    - `micro=1`, `global=4`, `seq=2048`, `train-iters=20`
    - `bf16`, `ZeRO-1 + CPU optimizer/offload`, `NUM_WORKERS=0`
    - `DISABLE_SAVE_CHECKPOINT=1`
    - dataset prefix remains `/home/user/Megatron-DeepSpeed/data/qwen_data_text_document`, but the current `.bin/.idx` are still very small, so this line should not yet be marketed as “large-scale real corpus training”
  - baseline:
    - `229.492s / 73478.543J / 320.179W / 2.2298 tokens/J`
  - `static 1005 MHz`:
    - run id: `eth_real_qwen25_7b_tp2pp2dp2_static1005_formal20_finetune_nw0_nosave_fixenv_20260420_r2_20260419_174710_sd-1`
    - `267.575s / 57177.500J / 213.688W / 2.8655 tokens/J`
    - delta vs baseline: `runtime +16.59% / avg_power -33.26% / energy -22.18% / tokens_per_j +28.51%`
  - `static 1200 MHz`:
    - run id: `eth_real_qwen25_7b_tp2pp2dp2_static1200_formal20_finetune_nw0_nosave_fixenv_20260420_r2_20260419_175317_sd-1`
    - `261.969s / 57168.610J / 218.226W / 2.8659 tokens/J`
    - delta vs baseline: `runtime +14.15% / avg_power -31.84% / energy -22.20% / tokens_per_j +28.53%`
  - `static 1395 MHz`:
    - `254.418s / 56385.520J / 221.626W / 2.9057 tokens/J`
    - delta vs baseline: `runtime +10.86% / avg_power -30.78% / energy -23.26% / tokens_per_j +30.31%`
  - `static 1500 MHz`:
    - run id: `eth_real_qwen25_7b_tp2pp2dp2_static1500_formal20_finetune_nw0_nosave_fixenv_20260420_r4_20260420_002554_sd-1`
    - `248.866s / 55554.388J / 223.230W / 2.9492 tokens/J`
    - delta vs baseline: `runtime +8.44% / avg_power -30.28% / energy -24.39% / tokens_per_j +32.26%`
  - `static 1650 MHz`:
    - run id: `eth_real_qwen25_7b_tp2pp2dp2_static1650_formal20_finetune_nw0_nosave_fixenv_20260420_r4_20260420_003214_sd-1`
    - `238.310s / 54380.384J / 228.192W / 3.0129 tokens/J`
    - delta vs baseline: `runtime +3.84% / avg_power -28.73% / energy -25.99% / tokens_per_j +35.12%`
  - interpretation:
    - `1200 MHz` almost dominates `1005 MHz`: nearly identical total energy, but noticeably shorter runtime
    - `1500 MHz` already improves on `1395 MHz` in both runtime and total energy
    - `1650 MHz` is currently the strongest observed time-energy trade-off on this real-model Ethernet line
- [2026-04-20] Ethernet real-model same-topology curve on `sd-1 + sd-2` has now been extended into the higher-frequency region with `1800 MHz` and `1950 MHz`:
  - `static 1800 MHz`:
    - run id: `eth_real_qwen25_7b_tp2pp2dp2_static1800_formal20_finetune_nw0_nosave_fixenv_20260420_r5_20260420_010053_sd-1`
    - `239.220s / 55380.942J / 231.506W / 2.9584 tokens/J`
    - delta vs baseline: `runtime +4.24% / avg_power -27.69% / energy -24.63% / tokens_per_j +32.68%`
  - `static 1950 MHz`:
    - run id: `eth_real_qwen25_7b_tp2pp2dp2_static1950_formal20_finetune_nw0_nosave_fixenv_20260420_r5_20260420_010704_sd-1`
    - `237.601s / 55839.752J / 235.015W / 2.9341 tokens/J`
    - delta vs baseline: `runtime +3.53% / avg_power -26.60% / energy -24.01% / tokens_per_j +31.59%`
  - interpretation:
    - both new high-frequency points completed `20/20` with `skipped=0`, so the Ethernet real-model line remains stable above `1500 MHz`
    - `1950 MHz` is now the fastest observed fixed point, but the speed win over `1650 MHz` is small
    - `1650 MHz` still dominates on energy and `tokens/J`, so it remains the best current sweet spot for this topology
- [2026-04-20] Ethernet real-model same-topology curve on `sd-1 + sd-2` has now been extended further with `2100 MHz` and `2250 MHz`:
  - `static 2100 MHz`:
    - run id: `eth_real_qwen25_7b_tp2pp2dp2_static2100_formal20_finetune_nw0_nosave_fixenv_20260420_r6_20260420_011555_sd-1`
    - `239.100s / 56694.104J / 237.115W / 2.8899 tokens/J`
    - delta vs baseline: `runtime +4.19% / avg_power -25.94% / energy -22.84% / tokens_per_j +29.61%`
  - `static 2250 MHz`:
    - run id: `eth_real_qwen25_7b_tp2pp2dp2_static2250_formal20_finetune_nw0_nosave_fixenv_20260420_r6_20260420_012206_sd-1`
    - `238.292s / 57151.293J / 239.837W / 2.8668 tokens/J`
    - delta vs baseline: `runtime +3.83% / avg_power -25.09% / energy -22.22% / tokens_per_j +28.57%`
  - interpretation:
    - both new higher-frequency points also completed `20/20` with `skipped=0`, so the line remains operationally stable even above `1950 MHz`
    - `2250 MHz` offers almost no runtime benefit versus `1650 MHz`, while clearly losing on energy and tokens/J
    - this strengthens the conclusion that `1650 MHz` remains the best current sweet spot rather than merely an artifact of sparse sampling
- [2026-04-19] First artifact-backed real-model same-topology baseline/static pair is now complete under `.context/eth_real_qwen25_7b_baseline_static_20260419/artifacts/`:
  - shared workload:
    - `sd-1 + sd-2`
    - `2 nodes x 4 GPUs`, GPU slice `0-3` on each node
    - `TP=2 / PP=2 / DP=2`
    - `Qwen2.5-7B-Instruct`, `28L / hidden=3584 / ffn=18944 / heads=28 / kv_heads=4`
    - `--load /home/user/Megatron-DeepSpeed/checkpoints/qwen25_7b_instruct_hf2megads_tp2pp2_real_main --finetune`
    - `qwen_data_text_document`, `micro=1`, `global=4`, `seq=2048`, `train-iters=20`
    - `bf16`, `ZeRO-1 + CPU optimizer/offload`, `NUM_WORKERS=0`
    - `DISABLE_SAVE_CHECKPOINT=1`
  - baseline:
    - run id: `eth_real_qwen25_7b_tp2pp2dp2_baseline_formal20_finetune_nw0_nosave_fixenv_20260419_sd-1`
    - `status=completed`, `final_iteration=20`, `ended_at=2026-04-19T13:37:58Z`
    - Zeus: `229.492s / 73478.543J / 320.179W / 2.2298 tokens/J`
  - static:
    - run id: `eth_real_qwen25_7b_tp2pp2dp2_static1395_formal20_finetune_nw0_nosave_fixenv_20260419_sd-1`
    - `status=completed`, `final_iteration=20`, `ended_at=2026-04-19T13:46:08Z`
    - Zeus: `254.418s / 56385.520J / 221.626W / 2.9057 tokens/J`
  - delta (`static 1395` vs baseline):
    - runtime `+10.86%`
    - avg_power `-30.78%`
    - energy `-23.26%`
    - tokens_per_j `+30.31%`
  - metadata quality:
    - both local copies include `run.json`, `events.jsonl`, `command.sh`, `ds_config.json`, `hostfile_snapshot.json`, `preflight.json`, `topology.json`
    - both runs were collected after the `DISABLE_SAVE_CHECKPOINT` + `/dev/shm` JIT-cache env fixes, so they do not suffer from the earlier checkpoint-save disk failure or CPUAdam permission fallback
- [2026-04-19] Real `Qwen2.5-7B-Instruct` Ethernet dual-node baseline run `eth_real_qwen25_7b_tp2pp2dp2_baseline_formal20_finetune_nw0_20260419_sd-1` completed the full `20/20` training window before failing only at final checkpoint save:
  - training-window metrics from the primary log:
    - step range: `1-20`
    - Zeus: `224.7s / 72390.0J / 322.2W`
    - final logged training step: `step=20`, `iter time (s)=9.705`, `skipped=0`
  - failure surface:
    - `sd-2` ranks failed during `torch.save()` of ZeRO optimizer states
    - exact storage error: `OSError: [Errno 28] No space left on device`
    - follow-on symptom: `RuntimeError: unexpected pos ...` during zip writer finalization
  - disk evidence collected immediately after:
    - `sd-2 /dev/sda2` showed `1.8T total / 1.7T used / 43M avail / 100%`
    - failed checkpoint directory size on `sd-2`: about `31G`
- [2026-04-19] Post-cleanup space state after removing generated training-output checkpoints:
  - removed checkpoint families on both `sd-1` and `sd-2`:
    - `eth_real_qwen25_7b_tp2pp2dp2_baseline_formal20_finetune_nw0_20260419_sd-1`
    - `eth_real_qwen25_7b_tp2pp2dp2_baseline_smoke5_finetune_nw0_20260419_sd-1`
  - post-delete `df -h /home/user`:
    - `sd-1`: `/dev/mapper/vg0-root 5.3T total, 1.1T avail`
    - `sd-2`: `/dev/sda2 1.8T total, 109G avail`
- [2026-04-19] First no-save clean-shell rerun surfaced a different startup failure unrelated to disk:
  - run id: `eth_real_qwen25_7b_tp2pp2dp2_baseline_formal20_finetune_nw0_nosave_20260419_sd-1`
  - failure site on `sd-2`:
    - `DeepSpeedCPUAdam` JIT load
    - `PermissionError: [Errno 13] Permission denied: '/home/user/.cache/torch_extensions/py310_cu128'`
  - interpretation:
    - the no-save launcher logic is not the blocker
    - the clean shell did not inherit the previously successful `/dev/shm` extension-cache envs
- [2026-04-19] Real `Qwen2.5-7B-Instruct` HF->Megatron conversion artifacts are now present on both Ethernet nodes:
  - `sd-1`
    - checkpoint: `/home/user/Megatron-DeepSpeed/checkpoints/qwen25_7b_instruct_hf2megads_tp2pp2_fixvocab2_20260419_114318`
    - log: `/home/user/Megatron-DeepSpeed/.context/qwen25_7b_instruct_hf2megads_tp2pp2_fixvocab2_20260419_114318.log`
    - terminal success marker: `save checkpoint completed`
  - `sd-2`
    - checkpoint: `/home/user/Megatron-DeepSpeed/checkpoints/qwen25_7b_instruct_hf2megads_tp2pp2_sd2_20260419_114724`
    - log: `/home/user/Megatron-DeepSpeed/.context/qwen25_7b_instruct_hf2megads_tp2pp2_sd2_20260419_114724.log`
    - terminal success marker: `save checkpoint completed`
  - shared artifact shape:
    - each output directory is about `15G`
    - each `global_step0` contains `67` top-level entries
    - each root directory contains `latest=global_step0`
  - consistency spot-check:
    - `layer_01-model_00-model_states.pt` hash matches across `sd-1/sd-2`: `5b903c6411fb252bc3dd6dd9361f67240644f6255b3a2590be85de38d21ea0e6`
    - `layer_16-model_01-model_states.pt` hash matches across `sd-1/sd-2`: `f96c08b21267307d3ef4b4aaa0e205b5047d3e123e509a7bed79c7494d478c2a`
    - `mp_rank_00_model_states.pt` hash differs between hosts, so if later需要做更严格一致性审计，应优先直接比 layer shards 或展开 state dict 做字段级 diff
- [2026-04-19] User-corrected primary evidence is now artifact-backed as **same-topology baseline/static comparison** under `.context/dual_env_topology_compare_tpge2_20260419/artifacts/{eth_static,ib_static}/`:
  - shared workload:
    - `2 nodes x 4 GPUs`
    - `TP/PP/DP in {2/2/2, 4/2/1}`
    - `36 layers`, `hidden=2048`, `ffn=11008`, `heads=16`, `kv_heads=4`
    - `micro_batch_size=1`, `global_batch_size=4`, `seq_length=2048`, `train-iters=20`
    - `ZeRO-1 + CPU optimizer/offload`, dataset `qwen_data_text_document`
  - Ethernet (`sd-1/sd-2`, GPU `0-3`):
    - `TP=2 / PP=2 / DP=2` baseline `128.5s / 40345.8J / 314.1W`
    - `1005 MHz` `149.9s / 31288.0J / 208.8W` (`time +16.7%`, `energy -22.5%`)
    - `1200 MHz` `141.0s / 30194.9J / 214.2W` (`time +9.7%`, `energy -25.2%`)
    - `1395 MHz` `136.6s / 29873.3J / 218.7W` (`time +6.3%`, `energy -26.0%`)
    - `TP=4 / PP=2 / DP=1` baseline `160.9s / 49457.5J / 307.3W`
    - `1005 MHz` `172.5s / 35384.5J / 205.1W` (`time +7.2%`, `energy -28.5%`)
    - `1200 MHz` `171.7s / 35629.6J / 207.5W` (`time +6.7%`, `energy -28.0%`)
    - `1395 MHz` `167.2s / 35556.0J / 212.7W` (`time +3.9%`, `energy -28.1%`)
  - IB (`v100x16-1/v100x16-2`, GPU `8-11`):
    - `TP=2 / PP=2 / DP=2` baseline `152.5s / 105381.0J / 691.2W`
    - `990 MHz` `205.3s / 78912.9J / 384.4W` (`time +34.7%`, `energy -25.1%`)
    - `1080 MHz` `188.3s / 76058.7J / 403.8W` (`time +23.5%`, `energy -27.8%`)
    - `1155 MHz` `185.4s / 77563.7J / 418.4W` (`time +21.6%`, `energy -26.4%`)
    - `TP=4 / PP=2 / DP=1` baseline `133.0s / 99678.1J / 749.4W`
    - `990 MHz` `172.0s / 70556.1J / 410.1W` (`time +29.3%`, `energy -29.2%`)
    - `1080 MHz` `159.2s / 68333.0J / 429.3W` (`time +19.7%`, `energy -31.4%`)
    - `1155 MHz` `156.3s / 69418.2J / 444.2W` (`time +17.5%`, `energy -30.4%`)
  - provenance:
    - raw manifests: `.context/dual_env_topology_compare_tpge2_20260419/artifacts/{eth_static,ib_static}/`
    - Chinese report summary: `汇报总结_20260415/10_同拓扑下基线与定频对比.md`
- [2026-04-19] Dual-environment `TP>=2` topology-only comparison is now artifact-backed under `.context/dual_env_topology_compare_tpge2_20260419/artifacts/{eth,ib}/`:
  - shared workload:
    - `2 nodes x 4 GPUs`
    - `TP/PP/DP in {2/2/2, 4/2/1}`
    - `36 layers`, `hidden=2048`, `ffn=11008`, `heads=16`, `kv_heads=4`
    - `micro_batch_size=1`, `global_batch_size=4`, `seq_length=2048`, `train-iters=20`
    - `ZeRO-1 + CPU optimizer/offload`, dataset `qwen_data_text_document`
  - Ethernet (`sd-1/sd-2`, GPU `0-3`) Zeus totals:
    - `TP=2 / PP=2 / DP=2`: `129.386s / 40490.202J / 312.942W`
    - `TP=4 / PP=2 / DP=1`: `158.058s / 48914.498J / 309.471W`
    - relative change (`tp4pp2dp1` vs `tp2pp2dp2`): runtime `+22.16%`, energy `+20.81%`, avg power `-1.11%`
  - IB (`v100x16-1/v100x16-2`, GPU `8-11`) Zeus totals:
    - `TP=2 / PP=2 / DP=2`: `155.205s / 105340.952J / 678.719W`
    - `TP=4 / PP=2 / DP=1`: `127.520s / 97714.232J / 766.267W`
    - relative change (`tp4pp2dp1` vs `tp2pp2dp2`): runtime `-17.84%`, energy `-7.24%`, avg power `+12.90%`
  - provenance:
    - Ethernet manifest: `.context/dual_env_topology_compare_tpge2_20260419/artifacts/eth/eth_topology_compare_tpge2_kv4_20260419_manifest_20260419_093732.txt`
    - IB manifest: `.context/dual_env_topology_compare_tpge2_20260419/artifacts/ib/ib_topology_compare_tpge2_kv4_20260419_manifest_20260419_173732.txt`
    - local summary: `.context/dual_env_topology_compare_tpge2_20260419/results_summary_kv4.md`
  - diagnostic note:
    - the original `kv_heads=2` plan failed for `TP=4 / PP=2 / DP=1` with `AssertionError: 2 is not divisible by 4`; the final formal comparison reran both topologies with `kv_heads=4` to keep the model legal and comparable
- [2026-04-19] Dual-environment common-workload `2x4` sweep is now fully artifact-backed under `.context/dual_env_common_workload_20260419/artifacts/{eth,ib}/`:
  - shared workload:
    - `2 nodes x 4 GPUs`
    - `TP=1`, `PP=2`, `DP=4`
    - `36 layers`, `hidden=2048`, `ffn=11008`, `heads=16`, `kv_heads=2`
    - `micro_batch_size=1`, `global_batch_size=4`, `seq_length=2048`, `train-iters=20`
    - `ZeRO-1 + CPU optimizer/offload`, dataset `qwen_data_text_document`
  - Ethernet (`sd-1/sd-2`, GPU `0-3`) Zeus totals:
    - baseline `168.216s / 51438.765J / 305.790W`
    - `1005 MHz` `198.728s / 40212.953J / 202.352W`
    - `1200 MHz` `184.716s / 38400.352J / 207.889W`
    - `1395 MHz` `181.379s / 38436.713J / 211.914W`
    - relative to baseline: runtime `+18.14% / +9.81% / +7.82%`, energy `-21.82% / -25.35% / -25.28%`, avg power `-33.83% / -32.02% / -30.70%`
  - IB (`v100x16-1/v100x16-2`, GPU `8-11`) Zeus totals:
    - baseline `182.482s / 114005.406J / 624.750W`
    - `990 MHz` `244.728s / 88633.158J / 362.170W`
    - `1080 MHz` `229.767s / 86021.344J / 374.385W`
    - `1155 MHz` `212.352s / 84274.511J / 396.863W`
    - relative to baseline: runtime `+34.11% / +25.91% / +16.37%`, energy `-22.26% / -24.55% / -26.08%`, avg power `-42.03% / -40.07% / -36.48%`
  - operational bring-up notes captured by this sweep:
    - current canonical launcher fix successfully eliminated stale `.deepspeed_env` pollution (`PATH`, `TMPDIR`, `TORCH_EXTENSIONS_DIR`, `PYTHONPYCACHEPREFIX`) that had been causing reused-temp-path failures
    - node-local `/dev/shm/.../index-cache` is not a safe multi-node `DATA_CACHE_PATH` unless every node is preseeded; the completed sweep used persistent per-node `data/index-cache` with Ethernet hash `33c91528b53c7a971dc9e5a3b24c9665` and IB hash `2025292d291ff386fedc1b73e7aace6c` present on both nodes
- [2026-04-11] Fresh IB formal remote sanity check completed against live DGX2 artifacts:
  - verified on `sd@v100x16-1`:
    - source `ib_dual8_tp4pp1dp2_formal_990_20260410_20260410_161719_DGX2-1`
    - target `ib_dual16_tp4pp1dp4_formal_1080_20260411_110907_DGX2-1`
    - target `ib_dual16_tp4pp1dp4_formal_1155_20260411_111702_DGX2-1`
  - structured metadata:
    - all three have `run.json.status=completed` and `final_iteration=20`
    - source topology resolves to `visible_gpu_indices=[8,9,10,11]`, `nproc_per_node=4`, `world_size=8`
    - target topology resolves to `visible_gpu_indices=[8,9,10,11,12,13,14,15]`, `nproc_per_node=8`, `world_size=16`
    - `hostfile_snapshot.json` explicitly records both `v100x16-1` and `v100x16-2`
  - event/log confirmation:
    - source `990` `events.jsonl` finalized with Zeus `404.1298 s / 238940.408 J / 591.247 W`
    - target `1080` `events.jsonl` finalized with Zeus `372.2408 s / 468052.854 J / 1257.393 W`
    - target `1155` `events.jsonl` finalized with Zeus `351.5297 s / 475668.378 J / 1353.139 W`
    - source / target logs both contain `v100x16-2:`-prefixed `iteration 1/20 ... 20/20` lines, so the second node was actively participating rather than inferred post hoc
  - secondary-node note:
    - `sd@v100x16-2` can see the target run directory, but it only exposes `ds_config.json`; the full artifact set remains on `DGX2-1`, so current two-node verification should treat node0 as the authoritative artifact writer
- [2026-04-11] Power-fixed live-IB replay artifact added at `.context/transfer_eval_ib_2x4_to_2x8_rerun_topology_fixed_live_ib_powerfix_20260411/`:
  - codepath note:
    - `predict_power_w()` now includes explicit `target.gpus_per_node / reference_gpus_per_node` scaling
    - `alpha_dp` remains unchanged at `2.220525e-11 s/byte`, so this replay isolates power-structure correction rather than communication-model changes
  - replay summary:
    - `total_time_mape=0.1148`
    - `step_time_mape=0.1148`
    - `avg_power_mape=0.0328`
    - `total_energy_mape=0.0786`
  - per-point replay details:
    - `990 MHz`: observed/predicted power `1189.17 / 1128.43 W`, power APE `5.11%`, energy APE `2.98%`
    - `1080 MHz`: observed/predicted power `1257.39 / 1229.19 W`, power APE `2.24%`, energy APE `9.22%`
    - `1155 MHz`: observed/predicted power `1353.14 / 1319.64 W`, power APE `2.48%`, energy APE `11.37%`
  - comparison against the immediately previous live-IB replay:
    - `avg_power_mape: 51.64% -> 3.28%`
    - `total_energy_mape: 46.07% -> 7.86%`
    - `total_time_mape: 11.48% -> 11.48%`
- [2026-04-11] Fresh transport-consistent `2x4 -> 2x8` IB formal replay is now stored at `.context/transfer_eval_ib_2x4_to_2x8_rerun_topology_fixed_live_ib_20260411/`:
  - source root: `.context/ib_formal_rerun_20260410/source_curated/`
  - target root: `.context/ib_formal_rerun_20260410/target_final/`
  - benchmark JSON: `.context/ib_formal_replay_20260410/comm_bench_2x4_20260410_ibhist.json`
  - topology note:
    - source workload = `2x4`, `TP=4`, `PP=1`, `DP=2`, `GBS=8`
    - target workload = `2x8`, `TP=4`, `PP=1`, `DP=4`, `GBS=16`
    - stock `scripts/evaluate_transfer_prediction.py` rejects this pair with `source and target workloads differ outside topology fields`, so this replay was executed as a topology-fixed override run rather than through the strict CLI path
  - live-IB replay summary:
    - `alpha_dp=2.220525e-11 s/byte`
    - `total_time_mape=0.1148`
    - `step_time_mape=0.1148`
    - `avg_power_mape=0.5164`
    - `total_energy_mape=0.4607`
  - per-point replay details:
    - `990 MHz`: observed/predicted time `401.187 / 435.396 s`, time APE `8.53%`, observed/predicted power `1189.17 / 564.22 W`, energy APE `48.51%`
    - `1080 MHz`: observed/predicted time `372.241 / 415.901 s`, time APE `11.73%`, observed/predicted power `1257.39 / 614.59 W`, energy APE `45.39%`
    - `1155 MHz`: observed/predicted time `351.530 / 401.421 s`, time APE `14.19%`, observed/predicted power `1353.14 / 659.82 W`, energy APE `44.32%`
  - sensitivity note:
    - replacing the live dual-node benchmark with the historical `.context/ib_formal_replay_20260410/ib_benchmark_111g_20260403.json` only moves time MAPE `11.48% -> 11.43%`
    - interpretation: for this fresh rerun pair, the residual gap is no longer dominated by benchmark-bandwidth optimism/pessimism
- [2026-04-10] The first recovered `2x8` IB formal target point is now local and complete:
  - remote run dir: `/home/sd/Megatron-DeepSpeed/experiments/ib_dual16_tp4pp1dp4_diag_nozeus_990_20260410_202433_DGX2-1`
  - local copy: `.context/ib_formal_rerun_20260410/target_partial/ib_dual16_tp4pp1dp4_diag_nozeus_990_20260410_202433_DGX2-1/`
  - config: `TP=4`, `PP=1`, `DP=4`, `GBS=16`, `MICRO=1`, `TRAIN_STEPS=20`, `STATIC_CLOCK_MHZ=990`, `MASTER_ADDR=192.168.205.201`, `MASTER_PORT=30041`
  - representative iteration logs:
    - `iter1=21.228s`
    - `iter2=20.221s`
    - `iter3=19.994s`
    - `iter9=19.867s`
  - Zeus summary from `run.json`:
    - `401.187s / 477080.108J / 1189.17W`
  - metadata confirmation:
    - `status=completed`
    - `final_iteration=20`
    - `topology.resolved.visible_gpu_indices=[8,9,10,11,12,13,14,15]`
    - `topology.resolved.nproc_per_node=8`
  - failure / retry note:
    - the earlier formal `990` attempt `ib_dual16_tp4pp1dp4_formal_990_20260410_192729_DGX2-1` failed right after `training ...` with `torch.AcceleratorError: CUDA error: out of memory`; the retry above completed on the same topology/slice
- [2026-04-10] `2x8` IB target smoke is now a completed, metadata-clean artifact on the intended `8-15` slice:
  - run dir: `/home/sd/Megatron-DeepSpeed/experiments/ib_dual16_tp4pp1dp4_smoke_1080_20260410_192035_DGX2-1`
  - config: `TP=4`, `PP=1`, `DP=4`, `GBS=16`, `MICRO=1`, `TRAIN_STEPS=2`, `STATIC_CLOCK_MHZ=1080`, `MASTER_ADDR=192.168.205.201`, `MASTER_PORT=30020`
  - Zeus interval:
    - `38.452s / 46709.31J / 1214.74W`
  - metadata confirmation from `run.json`:
    - `hostfile.entries=[v100x16-1,v100x16-2]`
    - `topology.resolved.visible_gpu_indices=[8,9,10,11,12,13,14,15]`
    - `topology.resolved.nproc_per_node=8`
    - `preflight.ok=true`
    - `preflight.node_results[*].gpu_indices=[8,9,10,11,12,13,14,15]`
  - operational note:
    - the earlier failed smoke `ib_dual16_tp4pp1dp4_smoke_1080_20260410_171438_DGX2-1` remained `status="initialized"` because `DGX2-2` lacked the dataset cache hash `d1158a21c6d1be91201644dbce18ab32_*`; after syncing those files and cleaning hung ranks, the same launcher path completed successfully
- [2026-04-10] The new transport-consistent `2x4` IB smoke rerun completed successfully on the intended `8-11` GPU slice:
  - run dir: `/home/sd/Megatron-DeepSpeed/experiments/ib_2x4_smoke_rerun_20260410_retry_20260410_155057_DGX2-1`
  - config: `TP=4`, `PP=1`, `DP=2`, `GBS=8`, `MICRO=1`, `TRAIN_STEPS=2`, `STATIC_CLOCK_MHZ=1080`, `MASTER_ADDR=192.168.205.201`, `MASTER_PORT=29973`
  - step metrics:
    - `step1=19.650s`
    - `step2=19.455s`
  - Zeus interval:
    - `39.1s / 23609.5J / 603.3W`
  - metadata confirmation from `run.json`:
    - `hostfile.entries=[v100x16-1,v100x16-2]`
    - `topology.resolved.visible_gpu_indices=[8,9,10,11]`
    - `topology.resolved.nproc_per_node=4`
    - `preflight.ok=true`
- [2026-04-10] The first retry before the successful smoke exposed a launcher-only regression rather than a training/runtime regression:
  - run dir: `/home/sd/Megatron-DeepSpeed/experiments/ib_2x4_smoke_rerun_20260410_retry_20260410_154329_DGX2-1`
  - failure site: `deepspeed/launcher/runner.py` parsing `.deepspeed_env`
  - root cause: the overlay writer inserted a blank line between preserved env lines and appended `MEGATRON_*` lines, while the current DeepSpeed runner expects every non-comment line to contain `=`
- [2026-04-10] `VALIDATE_ONLY=1` artifact `/home/sd/Megatron-DeepSpeed/experiments/ib_2x4_validate_aliasfix2_20260410_20260410_160700_DGX2-1` confirms the hostfile alias fix:
  - `preflight.json.node_results` now contains exactly `2` nodes (`DGX2-1`, `DGX2-2`)
  - both nodes report `cuda_visible_devices="8,9,10,11"` and `gpu_indices=[8,9,10,11]`
- [2026-04-10] Zeus still logs `Started monitoring GPUs: [0, 1, 2, 3]` even when the launcher uses `CUDA_VISIBLE_DEVICES=8,9,10,11`; current evidence indicates this is visible-device-relative logging rather than a real regression in device selection, because `run.json`, `topology.json`, and preflight all agree on the absolute slice `8-11`
- [2026-04-10] Forensic check of the earlier "successful" IB smoke artifact `/home/sd/Megatron-DeepSpeed/experiments/ib_2x4_smoke_rerun_20260410_20260410_132952_DGX2-1` shows the metadata fix was still not evidenced in that run:
  - `run.json.hostfile = {}`
  - `run.json.topology.resolved = {}`
  - `topology.json` still says `nproc_per_node=16` and `visible_gpu_indices=[0..15]`
  - `preflight.json` contains a local-node entry with empty `cuda_visible_devices` and `gpu_indices=[0..15]`
  - interpretation: the run proves training can finish, but not that the patched launcher metadata path was active
- [2026-04-10] Clean single-node deepspeed metadata verification on `DGX2-2` is stored at `/home/sd/Megatron-DeepSpeed/experiments/ib_single_node_metadata_verify_20260410_20260410_134052_DGX2-2` and shows the patched path works when the launcher is invoked with explicit `CUDA_VISIBLE_DEVICES=8,9,10,11` and `LOCAL_GPU_INDICES=8,9,10,11`:
  - `run.json.environment.CUDA_VISIBLE_DEVICES = "8,9,10,11"`
  - `run.json.topology.requested = {tp: "4", pp: "1"}`
  - `run.json.topology.resolved.visible_gpu_indices = [8,9,10,11]`
  - `run.json.preflight.node_results[0].gpu_indices = [8,9,10,11]`
  - the run finalizes as `status="incomplete"` because rank0 later aborts during model build with `AttributeError: module 'psutil' has no attribute 'virtual_memory'`
- [2026-04-10] Live occupancy snapshot before the intended dual-node rerun:
  - `DGX2-1` is fully occupied by external `VLLM::Worker_TP0..15` processes from `lb`'s `vllm serve /share-data/models/Llama-3.1-70B-Instruct --tp 16`
  - each GPU reports about `9436 MiB` resident and low but non-zero compute utilization (`~4-8%` in `nvidia-smi pmon`)
  - `DGX2-2` remains clean on `GPU 8-11` and can still be used for isolated launcher verification
- [2026-04-10] Live dual-node DGX2 IB benchmark succeeded after aligning launcher provenance with the historical run pair:
  - remote benchmark script had to be synced from local `.context/torch_nccl_comm_bench.py`; the stale remote copy rejected `--warmup-iters` / `--iters`
  - using recovered-run-aligned `MASTER_ADDR=192.168.205.201` plus `GLOO_SOCKET_IFNAME=enp6s0`, `NCCL_SOCKET_IFNAME=enp6s0`, `NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9`, `NCCL_IB_DISABLE=0` produced a successful `2x4` benchmark JSON at `.context/ib_formal_replay_20260410/comm_bench_2x4_20260410_ibhist.json`
  - rank0 live output:
    - `1 MB  avg=0.274 ms  busbw=6.388  cv=0.128`
    - `4 MB  avg=0.711 ms  busbw=9.842  cv=0.079`
    - `16 MB avg=2.426 ms  busbw=11.543 cv=0.513`
    - `64 MB avg=5.971 ms  busbw=18.757 cv=0.031`
  - the benchmark completes cleanly across both nodes and all eight benchmark ranks exit successfully
- [2026-04-10] Fresh topology-fixed replay against the live dual-node IB benchmark is stored at `.context/ib_formal_replay_20260410/transfer_eval_current_code_topology_fixed_live_ib.json` and `.md`
  - `target total_time_mape=0.1901`
  - `target step_time_mape=0.1901`
  - `target avg_power_mape=0.5236`
  - `target total_energy_mape=0.4328`
  - `alpha_dp=2.220525e-11 s/byte`
  - per-point target time APE:
    - `990 MHz`: `15.13%`
    - `1080 MHz`: `19.31%`
    - `1155 MHz`: `22.60%`
- [2026-04-10] Comparison against the previous topology-fixed replay driven by `.context/ib_formal_replay_20260410/ib_benchmark_111g_20260403.json`:
  - time MAPE stays effectively unchanged: `18.98% -> 19.01%`
  - `alpha_dp` rises from `1.997e-12` to `2.221e-11 s/byte`, but remains far below slow-network scale
  - interpretation: the remaining raw-artifact IB mismatch is not explained by benchmark optimism alone
- [2026-04-09] IB reconstructed sanity replay artifact added at `.context/ib_synthetic_transfer_regression_20260409.md`.
- [2026-04-09] This replay is intentionally marked reconstructed rather than formal because the current workspace snapshot lacks the original raw `2x8` IB target artifact used by the paper-era checkpoint. The replay uses:
  - source topology definition from `.context/dual8_tp4pp1dp2_collection_20260401/.../run.json` with manual `node_count=2`, `gpus_per_node=4`
  - target topology definition from `.context/dual16_tp4pp1dp4_prediction_compare_20260326/experiments/.../run.json` with manual `node_count=2`, `gpus_per_node=8`
  - observed IB metrics from `.context/paper/experimental_data.md`
  - IB benchmark curve from `.context/transfer_2x4_to_2x8_ib_20260403.py`
- [2026-04-09] Reconstructed IB replay result:
  - legacy synthetic path: `time_mape=0.1237`, `power_mape=0.0302`, `energy_mape=0.0899`
  - current continuous-scaling path: `time_mape=0.1123`, `power_mape=0.0328`, `energy_mape=0.0760`
  - current cross-node coefficients on IB synthetic replay: `alpha_dp≈1.997e-12 s/byte`, `alpha_pp=0`, `alpha_tp=0`
- [2026-04-09] Per-frequency current IB synthetic replay:
  - `990 MHz`: predicted `429.1 s / 554.9 W / 238.1 kJ`, observed `395.7 s / 584.2 W / 231.2 kJ`
  - `1080 MHz`: predicted `408.5 s / 587.0 W / 239.8 kJ`, observed `366.5 s / 604.8 W / 221.7 kJ`
  - `1155 MHz`: predicted `393.2 s / 614.5 W / 241.6 kJ`, observed `345.5 s / 626.3 W / 216.4 kJ`
- [2026-04-09] After adding cluster-capacity transfer scaling and keeping power normalization local-node scoped, the latest formal Ethernet replay under `.context/transfer_eval_eth_qwen3b_1x4_source_curve_20260409_curated_to_eth_qwen3b_2x4_target_curve_20260408_sd-1/` reports:
  - `total_time_mape=0.0516`
  - `step_time_mape=0.0516`
  - `avg_power_mape=0.1238`
  - `total_energy_mape=0.1042`
- [2026-04-09] Per-point APEs for the new replay:
  - `1005 MHz`: step time `4.28%`, power `16.79%`, energy `13.23%`
  - `1200 MHz`: step time `7.07%`, power `12.95%`, energy `6.79%`
  - `1395 MHz`: step time `4.14%`, power `7.40%`, energy `11.23%`
- [2026-04-09] Relative to the earlier formal checkpoint under `.context/eth_2x4_curve_eval_20260409/transfer_eval/`, the latest replay improves:
  - time MAPE: `40.17% -> 5.16%`
  - avg-power MAPE: `20.99% -> 12.38%`
  - total-energy MAPE: `10.58% -> 10.42%`
- [2026-04-09] The final successful validation command sequence for this iteration is:
  - `python3 -m pytest --noconftest -q tests/unit_tests/test_freq_model.py -k 'cluster_capacity_scale_tracks_gpu_growth_and_pipeline_efficiency or load_calibrate_and_overlay_modes or cross_node_penalty_fit_scales_continuously_with_benchmark_bandwidth or network_quality_observation_and_penalty_scaling'`
    - result: `4 passed, 33 deselected in 33.90s`
  - `python3 scripts/evaluate_transfer_prediction.py --source-root .context/eth_2x4_curve_eval_20260409/eth_qwen3b_1x4_source_curve_20260409_curated --target-root .context/eth_2x4_curve_eval_20260409/eth_qwen3b_2x4_target_curve_20260408_sd-1 --network-benchmark-json .context/comm_bench_2x4_eth0_20260406_175803.json`
- [2026-04-09] After switching from hard thresholding to benchmark-scaled continuous `alpha/beta`, targeted predictor validation passed locally:
  - `python3 -m pytest --noconftest -q tests/unit_tests/test_freq_model.py -k 'fit_cross_node_penalty_model_produces_positive_coefficients or cross_node_penalty_fit_scales_continuously_with_benchmark_bandwidth or network_quality_observation_and_penalty_scaling or cross_node_penalty_exposure_increases_with_frequency'`
    - result: `3 passed, 33 deselected in 0.13s`
  - `python3 -m pytest --noconftest -q tests/unit_tests/test_freq_model.py -k 'load_calibrate_and_overlay_modes or predict_script_network_benchmark_annotation'`
    - result: `2 passed, 34 deselected in 83.09s`
- [2026-04-09] New formal Ethernet `2x4` replay artifact after continuous benchmark scaling: `.context/transfer_eval_eth_qwen3b_1x4_source_curve_20260409_curated_to_eth_qwen3b_2x4_target_curve_20260408_sd-1/transfer_prediction.json` and `transfer_prediction_report.md`.
- [2026-04-09] The new replay reports `total_time_mape=0.3919`, `step_time_mape=0.3919`, `avg_power_mape=0.2198`, `total_energy_mape=0.0843`. Compared with the earlier formal checkpoint under `.context/eth_2x4_curve_eval_20260409/transfer_eval/`, time MAPE improves by about `0.98` percentage points and energy MAPE improves by about `2.14` percentage points, while avg-power MAPE worsens by about `0.99` percentage points.
- [2026-04-09] Under the new benchmark-scaled-alpha path, Ethernet `2x4` cross-node coefficients are `alpha_dp≈7.32e-10 s/byte`, `alpha_pp=0`, `alpha_tp=0`; this is lower than the earlier `alpha_dp≈8.41e-10`, but the predictor still recommends low clocks (`1005/1020/1035 MHz`) and remains systematically time-conservative across all three measured points.
- [2026-04-09] Formal `2x4` Ethernet transfer artifacts are staged locally under `.context/eth_2x4_curve_eval_20260409/`. Curated source root: `.context/eth_2x4_curve_eval_20260409/eth_qwen3b_1x4_source_curve_20260409_curated/` with `1005`, `1200`, and successful rerun `1395_r1`; target root: `.context/eth_2x4_curve_eval_20260409/eth_qwen3b_2x4_target_curve_20260408_sd-1/`; evaluation output: `.context/eth_2x4_curve_eval_20260409/transfer_eval/transfer_prediction.json` and `transfer_prediction_report.md`.
- [2026-04-09] Source formal sweep observability:
  - `1005 MHz`: standard run artifacts present under `eth_qwen3b_1x4_source_static1005_20260408_sd-2/`
  - `1200 MHz`: Zeus summary `169.6 s / 40464.6 J / 238.7 W / 4.049 tokens/J`; `step1=27.788s`, `step20=5.955s`, `skipped=0`
  - `1395 MHz` first attempt (`eth_qwen3b_1x4_source_static1395_20260408_sd-2`) failed during distributed init with `torch.distributed.DistNetworkError: EADDRINUSE` on `29549`; failed run dir remains for forensics but is excluded from formal local replay
  - `1395 MHz` rerun (`eth_qwen3b_1x4_source_static1395_r1_20260409_sd-2`) succeeded with `29551`; Zeus summary `169.6 s / 40837.5 J / 240.7 W / 4.012 tokens/J`; `step1=27.198s`, `step20=5.808s`, `skipped=0`
- [2026-04-09] Formal local transfer replay against `.context/comm_bench_2x4_eth0_20260406_175803.json` reports `total_time_mape=0.4017`, `step_time_mape=0.4017`, `avg_power_mape=0.2099`, `total_energy_mape=0.1058`. Per-target-point APEs:
  - `1005 MHz`: time `43.78%`, power `23.97%`, energy `9.31%`
  - `1200 MHz`: time `46.52%`, power `21.54%`, energy `14.95%`
  - `1395 MHz`: time `30.21%`, power `17.46%`, energy `7.48%`
- [2026-04-09] The replay keeps the slow-network branch active on Ethernet with `transport=eth0|ib_disable=1`, `effective_bandwidth_gbps≈0.2026`, `small_message_jitter_cv≈0.0153`, `large_message_jitter_cv≈0.0072`, and cross-node coefficients `alpha_dp≈8.41e-10 s/byte`, `alpha_pp=0`, `alpha_tp=0`. The resulting predictor remains systematically time-conservative on Ethernet, recommending low frequencies (`1005/1020/1035 MHz`) and overestimating runtime at all three measured points.
- [2026-04-08] First full-GPU `2x4` Ethernet smoke on `sd-1/sd-2` succeeded with `GPU 0,1,2,3` per host. Artifacts now live under `/home/user/Megatron-DeepSpeed/experiments/eth_qwen3b_2x4_fullgpu_smoke_20260408_sd-1/`: `manual_2x4.log`, `run.json`, `events.jsonl`, `command.sh`, and `notes.md`. Key runtime lines are `step1=16.150s`, `step2=8.303s`, `global_batch_size=4`, `skipped=0`; rank-0 memory line reports `reserved/max reserved=10966/10966 MB`, and rank-4 reports `10086/10086 MB`.
- [2026-04-08] The corresponding `events.jsonl` confirms the standard interval tracker is back on Ethernet: `initialized -> interval -> finalized(status=completed)`. The Zeus interval for steps `1-2` reports `time_s=24.477`, `energy_j=6891.94`, `avg_power_w=281.56`, `interval_samples=8`, and `tokens/J=2.377`.
- [2026-04-08] The first `2x4` relaunch after syncing tracker files failed immediately on both nodes with `ImportError: cannot import name 'collect_nvml_device_snapshot' from megatron.gpu_freq_manager`. Root cause was a stale remote `megatron/gpu_freq_manager.py` while `megatron/experiment_tracker.py` had already been updated. Syncing that one file from the local workspace to both `sd-1` and `sd-2` cleared the blocker on the next retry.
- [2026-04-07] `2x2` Ethernet target smoke succeeded on `sd-1/sd-2` with `GPU 2,3` per host. Log: `/home/user/Megatron-DeepSpeed/experiments/eth_qwen3b_target_validate_fixpy_20260407_sd-1/manual_cpuoffload_r5.log`. Key runtime lines: `step1=37.418s`, `step2=10.976s`, `global_batch_size=4`, `skipped=0`; rank-0 memory line reports `reserved/max reserved=11126/11126 MB`, rank-2 reports `12786/12786 MB`.
- [2026-04-07] Aligned single-node Ethernet source smoke succeeded on `sd-2` at `/home/user/Megatron-DeepSpeed/experiments/eth_qwen3b_source_gb4_manual_20260407_sd-2/manual_gb4.log`. Configuration matches the target workload except topology (`PP=2`, `TP=1`, `GBS=4`, `ZeRO-1 + CPU offload`, `GPU 2,3`). Key runtime lines: `step1=34.361s`, `step2=8.296s`, `skipped=0`.
- [2026-04-07] Local provisional time-only replay using the aligned source point plus `.context/comm_bench_2x4_eth0_20260406_175803.json` predicts target `step_time≈14.683s` for the observed `2x2` Ethernet target whose measured `step2≈10.976s`, giving `APE≈33.8%`. The replay keeps the slow-network branch active with `alpha_dp≈8.41e-10 s/byte` and `transport=eth0|ib_disable=1`.
- [2026-04-07] Successful Ethernet benchmark on `sd-1/sd-2` using synced local code: `bash .context/run_comm_bench.sh 2x4` with `SIZES_MB="1 4 16 64"`, `BENCH_WARMUP_ITERS=3`, `BENCH_ITERS=8`, `NCCL_SOCKET_IFNAME=eth0`, `NCCL_IB_DISABLE=1`. Output artifact fetched locally as `.context/comm_bench_2x4_eth0_20260406_175803.json`. Summary: `1 MB -> 15.139 ms / 0.116 / cv 0.005`, `4 MB -> 59.982 ms / 0.117 / cv 0.001`, `16 MB -> 140.073 ms / 0.200 / cv 0.015`, `64 MB -> 552.722 ms / 0.203 / cv 0.007`.
- [2026-04-07] First full-size Ethernet benchmark attempt on `sd-1/sd-2` failed by timeout rather than launch/setup failure. Command used default `1/4/16/64/256 MB × 20` recipe under `.context/run_comm_bench.sh 2x4`; ranks launched cleanly on both nodes, printed results through `64 MB`, but `256 MB` did not finish within the launcher `timeout 300`, and the job exited `124`.

- [2026-03-28] `2x4` rerun `dual8_tp4pp1dp2_20260328_r2` first-failure block (around line ~884 of `/home/sd/Megatron-DeepSpeed/.context/dual8_tp4pp1dp2_20260328_r2.log`) shows distributed init failing with `torch.distributed.DistBackendError: NCCL ... Failed to CUDA calloc async 4 bytes`, then cascading NCCL watchdog/TCPStore broken-pipe shutdown.
- [2026-03-28] Remote contention snapshot: `nvidia-smi` on `DGX2-1` reports all 16 GPUs near `~29.8 GiB` used at sampling time; this explains immediate allocator failure for the intended `GPU0-3` run. A detached watcher (`screen`: `dual8_tp4pp1dp2_watch_20260328`) now polls for free GPUs and auto-launches collection when available.
- [2026-03-28] Remote run is in-flight for `2x4` source topology acquisition: `screen -S dual8_tp4pp1dp2_20260328` on `sd@v100x16-1`; log path `/home/sd/Megatron-DeepSpeed/.context/dual8_tp4pp1dp2_20260328.log`. Early log confirms deepspeed multinode launch command started for frequency `990 MHz`.
- [2026-03-28] Dataset-readiness gate for strict `2x4 -> 2x8` transfer is now available via `scripts/check_transfer_dataset_readiness.py`. Current snapshot check (`source: 2,4,4,1,2`, `target: 2,8,4,1,4`) reports `{source_matches: 0, target_matches: 0, ready: false}` with exit code `2`.
- [2026-03-27] Exposure-calibration replay artifact is saved at `.context/dual8_generalization_20260327_exposure_calibration/comparison.json` with summary in `summary.md`. This run uses the available dual8 held-out set (workspace snapshot currently has no explicit `2x4` samples): source `tp2pp4dp2,tp2pp2dp4,tp1pp4dp4`, target `tp4pp1dp4`. Before/after stays identical (`time MAPE=0.6827366585`, `energy MAPE=0.5094818431`, default `1110 MHz`), and calibrated exposure knobs remain `pp=1.0, dp=1.0, tp=1.0, dp_group_gain=0.5, pp_bubble_gain=0.5`.
- [2026-03-26] Multi-topology base-prior replay for held-out `tp4pp1dp4` is saved at `.context/dual8_generalization_20260326/leave_one_tp4pp1dp4_multi_topology_prior_20260326/comparison.json` with a concise write-up in `summary.md`. The explicit fitted topology residuals remain zero, but the new always-on pooled-topology prior still improves the held-out target to `time MAPE ≈ 68.274%`, `energy MAPE ≈ 50.948%`, with default moving to `1110 MHz` and recommended frequencies `[1110, 1117, 1125]`.
- [2026-03-26] Full predictor validation after the new multi-topology prior: `python3 -m pytest --noconftest -q tests/unit_tests/test_freq_model.py` passes with `33 passed in 123.42s`.
- [2026-03-26] Latest `tp4pp1dp4` leave-one-topology-out replay after topology base + shape scaling is saved at `.context/dual8_generalization_20260326/leave_one_tp4pp1dp4_topology_shape_scaling_20260326/comparison.json` with a readable recap in `summary.md`. Before/after remain numerically identical: `time MAPE = 85.752%`, `energy MAPE = 70.756%`, default `600 MHz`, recommended `[600, 607, 615]`, and fitted topology sensitivities all stay at `0.0`.
- [2026-03-26] Predictor implementation validation after the latest topology-shape/base-scaling changes: `python3 -m pytest --noconftest -q tests/unit_tests/test_freq_model.py` passes with `31 passed in 100.02s`.
- [2026-03-26] Leave-one-topology-out pooled-calibration replay for target `tp4pp1dp4` is saved at `.context/dual8_generalization_20260326/leave_one_tp4pp1dp4_after_fix/comparison.json`. Before/after the pooled-reference fix, time MAPE stays about `85.744% -> 85.752%` and energy MAPE about `70.740% -> 70.756%`; the main visible change is in calibration metadata (`reference_topology_count: 1 -> 3`, `reference_topology_dispersion: 0.0 -> 0.213`), not in predictive quality.
- [2026-03-26] Focused `8+8` transfer-damping comparison lives at `.context/dual8_generalization_20260326/focused_transfer_compare_after_fix/comparison_two_pairs.json`. Representative results are essentially unchanged before/after the patch: `tp2pp4dp2 -> tp4pp1dp4` stays at about `85.9%` time MAPE / `70.7%` energy MAPE with default `615 MHz`, while `tp4pp1dp4 -> tp2pp4dp2` stays at about `636%` time MAPE / `218%` energy MAPE with default moving only `547 -> 555 MHz`.
- [2026-03-26] Local `2x8 -> 2x16` transfer replay artifacts are staged under `.context/dual16_to_dual32_transfer_20260326/` with per-mode payloads in `no_network/prediction.json` and `with_network/prediction.json`, plus a compact summary in `summary.md`. On target `dual32_tp4pp1dp8`, both modes are numerically identical: `time MAPE ≈ 73.7%`, `energy MAPE ≈ 14.0%`, default `480 MHz`. Representative misses: `990 MHz -> 2046.8 s / 1205.7 W / 2.468 MJ` measured vs `4817.4 s / 513.7 W / 2.475 MJ` predicted, and `1155 MHz -> 2252.6 s / 1245.8 W / 2.806 MJ` measured vs `4494.7 s / 656.7 W / 2.952 MJ` predicted.
- [2026-03-26] Local replay summary for the completed `2x8` control is saved at `.context/dual16_tp4pp1dp4_prediction_compare_20260326/summary.json` and `.context/dual16_tp4pp1dp4_prediction_compare_20260326/summary.md`. On `990/1080/1155 MHz`, analytic self-replay is about `13.0%` MAPE on total time and `6.8%` on total energy; predicted sampled-point ordering is time `1155 < 1080 < 990` and energy `990 < 1080 < 1155`, while measured ordering is `990 < 1155 < 1080` for both.
- [2026-03-26] Replay blocker discovered and fixed: the new dual-node `8+8` runs contain valid Zeus interval metrics in `events.jsonl`, but their `run.json` omits `freq_policy.mode/static_clock_mhz`. `analysis/freq_model/workload.py` now infers `static<freq>` from naming metadata so completed static runs remain loadable without hand-editing artifacts.

- [2026-03-26] Predictor implementation validation: `python3 -m pytest --noconftest -q tests/unit_tests/test_freq_model.py` passes with `23 passed in 82.77s`. New coverage checks both the benchmark summarization path and cross-node penalty scaling under worse measured bandwidth/jitter, plus CLI propagation via `--network-benchmark-json`.
- [2026-03-26] Completed dual-node `8+8` comparison results: `dual16_tp4pp1dp4_static990_20260325_235717_DGX2-1 -> 2117.98 s / 588.21 W / 1245818.77 J`, `dual16_tp4pp1dp4_static1080_20260326_003510_DGX2-1 -> 2955.33 s / 565.34 W / 1670769.34 J`, `dual16_tp4pp1dp4_static1155_20260326_012657_DGX2-1 -> 2466.87 s / 605.89 W / 1494653.14 J`. Local comparison artifact: `.context/dual_topology_control_compare_20260326.json`.
- [2026-03-26] Same-transport communication benchmark completed successfully under remote artifact `/home/sd/Megatron-DeepSpeed/.context/comm_bench_dual8_tailscale_20260326.json`. For `world_size=16` (`8+8`) with `tailscale0` and `NCCL_IB_DISABLE=1`, measured `busbw` stays near `0.207-0.212 GB/s`; `cv` is `~0.096` at `4 MB`, `~0.136` at `16 MB`, `~0.008` at `64 MB`, and `~0.012` at `256 MB`.
- [2026-03-26] Detached dual-node `8+8` comparison batch is running in remote `screen` session `dual16_tp4pp1dp4_20260325`. Launcher script: `/home/sd/Megatron-DeepSpeed/.context/dual16_tp4pp1dp4_compare_20260325.sh`; live log: `/home/sd/Megatron-DeepSpeed/.context/dual16_tp4pp1dp4_20260325.log`. The first `990 MHz` run has passed `[after megatron is initialized]` and reached `training ... [before the start of training step]` without cache/init regressions.
- [2026-03-26] Same-transport communication benchmark artifacts are staged under remote `.context`: benchmark script `/home/sd/Megatron-DeepSpeed/.context/torch_nccl_comm_bench.py`, launcher `/home/sd/Megatron-DeepSpeed/.context/comm_bench_torchrun_dual8.sh`, watcher session `dual16_post_comm_watch_20260325`, and future log path `/home/sd/Megatron-DeepSpeed/.context/comm_bench_dual8_tailscale_20260325.log`. This probe intentionally matches the current training transport (`tailscale0`, `NCCL_IB_DISABLE=1`) instead of assuming IB.
- [2026-03-25] Current-code 32-GPU transfer-prediction replay is saved at remote `/home/sd/Megatron-DeepSpeed/.context/dual32_tp4pp1dp8_prediction_compare_20260325/prediction_bundle.json` and summarized locally at `.context/dual32_tp4pp1dp8_pred_vs_measured_20260325.json`. Key predicted vs measured points: `1050 MHz -> 683.2 s / 2455.1 W / 1.677 MJ` vs `2473.0 s / 1166.0 W / 2.884 MJ`, `1125 MHz -> 656.3 s / 2670.3 W / 1.753 MJ` vs `3216.3 s / 1144.4 W / 3.681 MJ`, `1200 MHz -> 633.4 s / 2902.8 W / 1.839 MJ` vs `2871.6 s / 1213.5 W / 3.485 MJ`.
- [2026-03-25] Detached 32-GPU refinement batch is running in remote `screen` session `dual32_refine_20260325`. Launcher script: `/home/sd/Megatron-DeepSpeed/.context/dual32_tp4pp1dp8_refine_20260325.sh`; live log: `/home/sd/Megatron-DeepSpeed/.context/dual32_refine_20260325.log`. Planned frequencies are `990`, `1020`, `1050`, `1080`, and `1155 MHz`; the first `990 MHz` run has already passed `[after megatron is initialized]` and completed `iteration 1/20` (`steps: 1 loss: 12.6476 iter time (s): 139.022`) without repeating the earlier cache-miss failure.
- [2026-03-25] Launched detached 32-GPU static validation batch in remote `screen` session `dual32_static_20260325`. Launcher script: `/home/sd/Megatron-DeepSpeed/.context/dual32_tp4pp1dp8_batch_20260325.sh`; live boot/log file: `/home/sd/Megatron-DeepSpeed/.context/dual32_static_20260325.log`. Planned frequencies are `1050`, `1125`, and `1200 MHz` for `TP=4, PP=1, DP=8`, `train-iters=20`. Early health check confirms the first `1050 MHz` run passed distributed initialization and model build with GPUs allocated on both nodes.
- [2026-03-25] 32-GPU `.local` smoke validation artifact: remote screen log `/home/sd/Megatron-DeepSpeed/.context/dual32_smoke_20260325.log`, experiment dir `/home/sd/Megatron-DeepSpeed/experiments/dual32_tp4pp1dp8_smoke1050_20260325_130312_DGX2-1/`. First attempt exposed the real failure signature on `DGX2-2`: missing `index-cache/87922b4c8bcca7cc5c0cfc95a72dd19d_{doc,sample,shuffle}.npy`. After syncing those files, the rerun progressed through model/ZeRO setup, dataset loading, and into training; rank-0 log shows `steps: 1 loss: 12.6476 iter time (s): 88.552`, while rank-1 aggregate log reports `iteration 1/2 ... number of skipped iterations: 0 ... number of nan iterations: 0`.
- [2026-03-22] End-to-end replay after the calibration-forwarding fix shows the new shared-fit path is now actually visible in predictor outputs. On `.context/dual-node-validation-20260322/multi_topology_global_refit_20260322.md`, the real dual-node replay APEs are roughly `3.13/1.35/4.10%` (time/power/energy) for `TP=2,PP=4,DP=2` and `2.14/3.19/5.15%` for `TP=2,PP=2,DP=4`. The main remaining observability signal is that legacy coarse anchors still diverge strongly on time overhead (`avg_overhead_ape≈82.8%`), which points to a modeling mismatch rather than a logging issue.

## 2026-03-17 Local/Remote Replay Check
- Local focused validation command: `pytest -q --noconftest tests/unit_tests/test_freq_model.py` -> `19 passed in 60.96s`.
- Local replay artifacts: `.context/transfer-debug-20260317/local-tp1pp4dp4-topodist/prediction.json` and `.context/transfer-debug-20260317/local-tp1pp2dp8-topodist/prediction.json`.
- Replay summary on measured diagnostic points: `TP=1, PP=4, DP=4` stays at about `1.25%` runtime MAPE / `1.05%` power MAPE / `0.34%` energy MAPE; `TP=1, PP=2, DP=8` stays at about `2.26%` runtime MAPE / `1.66%` power MAPE / `0.65%` energy MAPE.
- Remote regenerated artifacts using `python3 .context/regen_transfer_bundle.py`: `/home/sd/Megatron-DeepSpeed/.context/canonical-transfer-tp1pp4dp4-20260317-topodist/prediction.json` and `/home/sd/Megatron-DeepSpeed/.context/canonical-transfer-tp1pp2dp8-20260317-topodist/prediction.json`. Both now recommend `[1185, 1192, 1200]` with default `1185 MHz`, matching local replay exactly.


- [2026-03-17] Local-only gap-regularization experiment artifacts live under `.context/transfer-debug-20260317/local-tp1pp4dp4-gapfix/` and `.context/transfer-debug-20260317/local-tp1pp2dp8-gapfix/`. It preserves measured-band fit (`TP=1, PP=4, DP=4` still about `1.25%` time / `1.05%` power MAPE) but is not promoted because the regenerated `TP=1, PP=4, DP=4` default still falls to `930 MHz`.
- [2026-03-17] After adding source-observed-band tail regularization, local `TP=1, PP=2, DP=8` transfer regeneration under `.context/transfer-debug-20260317/local-tp1pp2dp8-tailfix/` now recommends `[900, 907, 915]` instead of `[757, 765, 772]` while leaving the measured `1177/1185/1192 MHz` diagnostics effectively unchanged.
- [2026-03-17] Remote regeneration confirms the same result for `TP=1, PP=2, DP=8`: `/home/sd/Megatron-DeepSpeed/.context/canonical-transfer-tp1pp2dp8-20260317-tailfix/prediction.json` now defaults to `900 MHz` with calibration reference band ratios `~0.564-0.789`.
- [2026-03-17] Remote regeneration for `TP=1, PP=4, DP=4` still defaults to `997 MHz` even with the same tail regularization (`/home/sd/Megatron-DeepSpeed/.context/canonical-transfer-tp1pp4dp4-20260317-tailfix/prediction.json`), which isolates a broader low-frequency curve-shape issue that lives inside the source-observed band.
- [2026-03-17] Remote transfer artifact regenerated after syncing the new `analysis/freq_model/model.py`: `/home/sd/Megatron-DeepSpeed/.context/canonical-transfer-tp1pp2dp8-20260317-anchorfix/prediction.json`.
- [2026-03-17] Reanalysis against the existing `TP=1, PP=2, DP=8` measured trio is stored locally under `.context/real-v100-tp1pp2dp8-reanalysis-20260317-anchorfix/`; diagnostic-band MAPE is now `total_time≈2.07%`, `avg_power_w≈1.68%`, `total_energy≈0.44%`.
- [2026-03-17] Important caveat: the regenerated energy-first default for `TP=1, PP=2, DP=8` is still `757 MHz` with neighborhood `[757, 765, 772]`, so do not interpret this model change as full low-frequency-tail validation.
- [2026-03-17] Copied the minimal transfer-debug bundles from `sd@v100x16-1` into `.context/transfer-debug-20260316/` so the current local predictor can be replayed against the saved source/target `prediction.json` files without touching the remote working tree.
- [2026-03-17] Local offline replay with the new low-TP transfer-anchor adjustment yields: `TP=1, PP=4, DP=4` -> `total_time MAPE≈1.25%`, `avg_power_w MAPE≈1.05%`, `total_energy MAPE≈0.34%`; `TP=1, PP=2, DP=8` -> `total_time MAPE≈2.07%`, `avg_power_w MAPE≈1.68%`, `total_energy MAPE≈0.44%`.
- [2026-03-17] Focused local unit coverage for the new low-TP response runs with `pytest --noconftest tests/unit_tests/test_freq_model.py -k ...`; repo-wide pytest still requires `deepspeed` through `tests/conftest.py`.
## Logging
- Primary execution logs are written under `experiments/<run_id>/logs/<run_id>.log`.
- Detached remote `screen` launch logs are also captured under `experiments/_screen_boot/<session>.log`.
- `run.json` stores finalized metadata when tracker finalization succeeds.
- `events.jsonl` stores interval/checkpoint/status events and is a key fallback for incomplete finalization cases.

## Monitoring
- Zeus is the primary power/energy monitor for current V100 experiments.
- Required comparison metrics:
  - total energy to finish the fixed workload
  - total wall-clock time to finish the fixed workload
  - `energy_wh`
  - `avg_power_w`
  - `samples/Wh`
  - `tokens/J`
  - iteration time / throughput from training logs
- Recommendation review should consider the full-task energy/time tradeoff first; short-horizon power and per-step latency are supporting signals, not the final objective.
- When the run is steady from start to finish, per-step power and step time can be used as anchored observational proxies for total energy and total wall-clock comparisons across candidate frequencies.

## Tracing
- No dedicated distributed tracing system is currently used.
- Timing is inferred from Megatron/DeepSpeed logs and Zeus summaries.

## Alerting
| Alert | Condition | Severity | Channel |
|-------|-----------|----------|---------|
| Disk nearly full | Remote filesystem approaches 100% usage | High | Manual CLI inspection |
| Finalize mismatch | `run.json` incomplete but training log shows completion | Medium | Manual result validation |
| Static clock not reset | GPUs remain locked after run exit | High | Manual `nvidia-smi`/NVML reset |

## Health Checks
- [2026-03-21] Real dual-node static validation for `TP=2, PP=4, DP=2` at `1185 MHz` is recorded in `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/dual_tp2pp4dp2_static1185_fix_20260320.log` via wrapper `/home/sd/Megatron-DeepSpeed/.context/dual_static_tp2pp4dp2_1185_wrapper_20260320.sh`. Training steps `1-20` completed successfully with elapsed times stabilizing near `22.66-22.86 s/iter`; rank-0 Zeus summary: `Energy=533166.9 J`, `Avg Power=1167.1 W`, `Time=456.8 s`, `Tokens/J=1.229`.
- [2026-03-20] Predictor-side cross-node fit replay from `analysis/freq_model/cross_node.py` now yields `alpha_pp≈8.30e-09 s/byte`, `alpha_dp=0`, `alpha_tp≈8.84e-11 s/byte` with an explicit TP/PP/DP wait decomposition. Current calibration-point replay errors are: `TP=2,PP=4,DP=2 -> ~2.4%`, `TP=1,PP=4,DP=4 -> ~3.2%`, `TP=2,PP=2,DP=4 -> ~17.6%` on additive overhead time.
- [2026-03-19] Successful minimal cross-node logs live under `/home/sd/Megatron-DeepSpeed/.context/dual-node-pilot/dual8_tp2pp4dp1_base1_r1_free4_zero1_20260319/`. Final 1-step metrics from the completed run: `elapsed time per iteration ≈ 18977.6 ms`, `samples/sec ≈ 0.422`, `tokens/gpu/sec ≈ 107.917`, `TFLOPs ≈ 6.37`, and Zeus `Energy ≈ 17332.7 J`, `Avg Power ≈ 911.9 W`, `Time ≈ 19.0 s`, `Tokens/J ≈ 0.945`.
- [2026-03-19] The required dataset cache hash for the successful minimal cross-node run included `02c886db19dc87ae1844eba656640304_{doc,sample,shuffle}_idx.npy`; before syncing those files from `DGX2-1`, `DGX2-2` failed with `FileNotFoundError` during GPT dataset index-map loading.
- [2026-03-19] Dual-node pilot logs for the old-env 32-GPU bring-up live under `/home/sd/Megatron-DeepSpeed/.context/dual-node-pilot/dual32_tp4pp1dp8_base1_r7_oldenv_net_20260319/`; `node0.log` reaches `After Building Model` and `DeepSpeed is enabled.`, while `node1.log` stops after the expected distributed/nvfuser startup messages.
- [2026-03-19] Free-GPU 2x8 validation logs live under `/home/sd/Megatron-DeepSpeed/.context/dual-node-pilot/dual16_tp4pp1dp4_base1_r1_free8_20260319/`; this configuration cleanly initializes distributed and builds the model on the old env, confirming the tokenizer/Apex/network path is usable on both nodes when restricted to GPUs `8-15`.
- [2026-03-19] Free-GPU `TP=4, PP=1, DP=4, ZeRO-0` failure logs live under `/home/sd/Megatron-DeepSpeed/.context/dual-node-pilot/dual16_tp4pp1dp4_base1_r2_free8_zero0_20260319/`; both `node0.log` and `node1.log` show reproducible CUDA OOM inside `deepspeed.runtime.fp16.unfused_optimizer` / `apex.optimizers.fused_adam` when allocating optimizer state.
- [2026-03-19] Free-GPU `TP=4, PP=2, DP=2, ZeRO-1` failure logs live under `/home/sd/Megatron-DeepSpeed/.context/dual-node-pilot/dual16_tp4pp2dp2_base1_r1_free8_zero1_20260319/`; `node1.log` records the first failure at `DGX2-2` rank `11` with `torch.AcceleratorError: CUDA error: out of memory` during `torch.cuda.set_device`.
- [2026-03-17] The detached `TP=1, PP=2, DP=8` static validation trio completed successfully. Zeus summaries: `1177 MHz -> 1130.8 s / 2725506.6 J / 2410.3 W / 0.601 tokens/J`, `1185 MHz -> 1119.4 s / 2729647.6 J / 2438.4 W / 0.600 tokens/J`, `1192 MHz -> 1118.4 s / 2754717.5 J / 2463.0 W / 0.595 tokens/J`.
- [2026-03-17] Local comparison artifacts for the completed `TP=1, PP=2, DP=8` trio live under `.context/real-v100-tp1pp2dp8-static-20260316/`; use `measured_vs_predicted.tsv` for curve errors and `final_analysis.md` for the diagnostic conclusion.
- [2026-03-17] Curve-accuracy summary for the completed `TP=1, PP=2, DP=8` trio: `total_time` MAPE `≈11.73%`, `avg_power_w` MAPE `≈5.74%`, and `total_energy` MAPE `≈5.33%`. This is worse than the `TP=1, PP=4, DP=4` diagnostic, so the current transfer issue appears broader than pipeline depth alone.
- [2026-03-16] The first `TP=1, PP=2, DP=8` unfixed baseline attempt failed with CUDA OOM at step 5 on later pipeline-stage ranks. Retrying with allocator expansion (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) resolved the issue and produced final Zeus summary `1137.3 s / 3935271.5 J / 3460.1 W / 0.416 tokens/J`.
- [2026-03-16] Canonical zero-shot transfer artifacts for `TP=1, PP=2, DP=8` now live under `.context/canonical-transfer-tp1pp2dp8-20260316/`; default recommendation is `1177 MHz` with neighborhood `[1177, 1185, 1192]`.
- [2026-03-16] Remote `TP=1, PP=2, DP=8` static validation sequence completed in detached session `tp1pp2dp8_static_seq_20260316`; boot log is `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/tp1pp2dp8_static_seq_20260316.log`, and the completed frequencies were `1177/1185/1192 MHz`.
- [2026-03-16] Curve-accuracy summary for the completed `TP=1, PP=4, DP=4` trio: `total_time` MAPE is about `7.17%`, `avg_power_w` MAPE about `5.30%`, and `total_energy` MAPE about `1.48%`. Interpret the low energy error carefully: it is likely dominated by cancellation between time and power biases rather than by a truly correct curve.
- [2026-03-16] The detached `TP=1, PP=4, DP=4` static validation trio completed successfully. Zeus summaries: `1252 MHz -> 1300.1 s / 3078625.8 J / 2367.9 W / 0.532 tokens/J`, `1260 MHz -> 1287.9 s / 3067834.2 J / 2382.0 W / 0.534 tokens/J`, `1267 MHz -> 1286.3 s / 3097007.8 J / 2407.7 W / 0.529 tokens/J`.
- [2026-03-16] Local comparison artifacts for the completed `TP=1, PP=4, DP=4` trio live under `.context/real-v100-tp1pp4dp4-static-20260316/`; use `measured_vs_predicted.tsv` for ratio errors and `final_analysis.md` for the transfer takeaway.
- [2026-03-16] The `TP=1, PP=4, DP=4` 50-step unfixed baseline completed on `sd@v100x16-1` with Zeus summary `1316.7 s / 4174627.4 J / 3170.6 W / 0.392 tokens/J` from `/home/sd/Megatron-DeepSpeed/experiments/v100_tp1pp4dp4_baseline50_20260316_185249_DGX2-1/logs/v100_tp1pp4dp4_baseline50_20260316_185249_DGX2-1.log`.
- [2026-03-16] Canonical zero-shot transfer artifacts for `TP=1, PP=4, DP=4` now live under `.context/canonical-transfer-tp1pp4dp4-20260316/`; default recommendation is `1252 MHz` with neighborhood `[1252, 1260, 1267]` and predicted baseline-relative energy ratio near `0.749x`.
- [2026-03-16] Remote `TP=1, PP=4, DP=4` static validation sequence completed in detached session `tp1pp4dp4_static_seq_20260316`; boot log is `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/tp1pp4dp4_static_seq_20260316.log`, and the completed frequencies were `1252/1260/1267 MHz`.
- [2026-03-16] Canonical predictor-quality checkpoint on `sd@v100x16-1`: outputs live under `.context/canonical-transfer-tp2pp2dp4-20260316/` and `.context/canonical-transfer-tp2pp1dp8-20260316/`. Current canonical defaults are `1072 MHz` for `TP=2, PP=2, DP=4` and `1005 MHz` for `TP=2, PP=1, DP=8`.
- [2026-03-16] Transfer-time rescaling recheck on `sd@v100x16-1` completed successfully with the new target-topology low-frequency amplitude rule. Outputs live under `.context/remote-transfer-rescaled-tp2pp2dp4-20260316-r2/` and `.context/remote-transfer-rescaled-tp2pp1dp8-20260316-r2/`; defaults are now `1072 MHz` for `TP=2, PP=2, DP=4` and `1005 MHz` for `TP=2, PP=1, DP=8`.
- [2026-03-16] Second V100 recheck after restoring the low-frequency intensity baseline completed successfully. Outputs live under `.context/remote-recheck-tp2pp2dp4-20260316-v100-r2/` and `.context/remote-recheck-tp2pp1dp8-20260316-v100-r2/`; defaults improved to `1057 MHz` for `TP=2, PP=2, DP=4` and `960 MHz` for `TP=2, PP=1, DP=8`.
- [2026-03-16] Remote offline transfer recheck on `sd@v100x16-1` completed after syncing the updated `analysis/freq_model/` package plus baseline artifacts. Outputs live under `.context/remote-recheck-tp2pp2dp4-20260316-v100/` and `.context/remote-recheck-tp2pp1dp8-20260316-v100/`; the new defaults are `1035 MHz` for `TP=2, PP=2, DP=4` and `907 MHz` for `TP=2, PP=1, DP=8`.
- [2026-03-15] The detached `TP=2, PP=1, DP=8` static validation session `power_tp2pp1dp8_pred50_20260315_211016` completed successfully; synced artifacts live under `.context/real-v100-tp2pp1dp8-static-20260315/`, with measured-vs-predicted summary at `.context/real-v100-tp2pp1dp8-static-20260315/measured_vs_predicted.tsv` and narrative summary at `.context/real-v100-tp2pp1dp8-static-20260315/final_analysis.md`.
- [2026-03-15] Zeus summaries for `TP=2, PP=1, DP=8` 50-step validation: `967 MHz -> 912.1 s / 2298437.0 J / 2520.0 W`, `975 MHz -> 904.9 s / 2295007.9 J / 2536.1 W`, `982 MHz -> 898.0 s / 2290348.0 J / 2550.4 W`, `1050 MHz -> 841.8 s / 2265331.2 J / 2691.1 W`.
- [2026-03-15] The `TP=2, PP=1, DP=8` 50-step unfixed baseline completed with local synced artifact `.context/real-v100-tp2pp1dp8-baseline-20260315/v100_tp2pp1dp8_baseline50_20260315_203404_DGX2-1/` and Zeus summary `851.9 s`, `3549352.3 J`, `4166.4 W`, `0.462 tokens/J`.
- [2026-03-15] Transferred prediction artifacts for `TP=2, PP=1, DP=8` are staged under `.context/real-v100-tp2pp1dp8-transfer-20260315-with-baseline/`; default recommendation is `975 MHz`, balanced comparator is baseline-like (`1597 MHz`), and the recommended neighborhood is `[967, 975, 982]`.
- [2026-03-15] Remote `TP=2, PP=1, DP=8` static validation session: `power_tp2pp1dp8_pred50_20260315_211016`; launcher script is `.context/remote-launch-tp2pp1dp8-predicted50-20260315.sh`, boot log is `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_tp2pp1dp8_pred50_20260315_211016.log`, and frequencies are `967/975/982/1050 MHz`.
- [2026-03-15] The detached `TP=2, PP=2, DP=4` static validation session `power_tp2pp2dp4_pred50_20260315_161856` completed successfully; synced artifacts live under `.context/real-v100-tp2pp2dp4-static-20260315/`, with measured-vs-predicted summary at `.context/real-v100-tp2pp2dp4-static-20260315/measured_vs_predicted.tsv` and narrative summary at `.context/real-v100-tp2pp2dp4-static-20260315/final_analysis.md`.
- [2026-03-15] Zeus summaries for `TP=2, PP=2, DP=4` 50-step validation: `1072 MHz -> 1053.6 s / 2475841.1 J / 2349.8 W`, `1080 MHz -> 1046.1 s / 2471497.2 J / 2362.5 W`, `1087 MHz -> 1040.0 s / 2482053.4 J / 2386.6 W`, `1125 MHz -> 1006.1 s / 2492545.3 J / 2477.3 W`.
- [2026-03-15] `TP=2, PP=2, DP=4` preflight on `sd@v100x16-1` passed via `VALIDATE_ONLY=1`; the baseline launcher script is `.context/remote-launch-tp2pp2dp4-baseline50-20260315.sh`.
- [2026-03-15] The `TP=2, PP=2, DP=4` 50-step unfixed baseline completed with local synced artifact `.context/real-v100-tp2pp2dp4-baseline-20260315/v100_tp2pp2dp4_baseline50_20260315_155856_DGX2-1/` and Zeus summary `1030.6 s`, `3798903.8 J`, `3686.1 W`, `0.431 tokens/J`.
- [2026-03-15] Transferred prediction artifacts for `TP=2, PP=2, DP=4` are staged under `.context/real-v100-tp2pp2dp4-transfer-20260315-with-baseline/`; default recommendation is `1080 MHz`, balanced comparator is baseline-like (`1597 MHz`), and the recommended neighborhood is `[1072, 1080, 1087]`.
- [2026-03-15] Remote `TP=2, PP=2, DP=4` static validation session: `power_tp2pp2dp4_pred50_20260315_161856`; launcher script is `.context/remote-launch-tp2pp2dp4-predicted50-20260315.sh`, boot log is `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_tp2pp2dp4_pred50_20260315_161856.log`, and frequencies are `1072/1080/1087/1125 MHz`.
- `TP=8, DP=2` validation failed immediately in remote boot log `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_tp8dp2_baseline_20260312_1930.log` with `AssertionError: 28 is not divisible by 8`; treat this as a topology feasibility failure rather than a runtime instability.
- The fallback `TP=2, PP=4, DP=2` baseline is running in remote `screen` session `power_tp2pp4dp2_baseline_20260312_1944`; the boot log is `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_tp2pp4dp2_baseline_20260312_1944.log` and the attach command is recorded in `.context/remote-screen-sessions.tsv`.
- The fallback `TP=2, PP=4, DP=2` baseline completed with local synced artifact `.context/real-v100-tp2pp4dp2-baseline-20260312/v100_tp2pp4dp2_baseline_20260312_194435_DGX2-1/` and a Zeus summary of `419.508 s`, `1515838.132 J`, `3613.370 W`, `0.432342 tokens/J`.
- Zero-shot transfer prediction artifacts for the new topology are staged under `.context/real-v100-tp2pp4dp2-baseline-20260312/zero-shot-transfer-v1/`.
- Partial measured-vs-predicted comparison for the first three completed `TP=2, PP=4, DP=2` static points is staged at `.context/real-v100-tp2pp4dp2-static-20260312/measured_vs_predicted_partial.tsv`, with a short narrative summary in `.context/real-v100-tp2pp4dp2-static-20260312/partial_analysis.md`.
- The completed measured-vs-predicted comparison for `TP=2, PP=4, DP=2` is staged at `.context/real-v100-tp2pp4dp2-static-20260312/measured_vs_predicted.tsv`, with the final narrative summary at `.context/real-v100-tp2pp4dp2-static-20260312/final_analysis.md`.
- The detached static validation sweep session `power_tp2pp4dp2_static_validate_20260312_2000` has already produced all targeted Zeus summaries in `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_tp2pp4dp2_static_validate_20260312_2000.log`.
- Refactored hardware-first prediction outputs are staged under `.context/real-v100-power-static-20260312-merged/predictions-hw-first-v3/`; this local smoke run uses the fresh unfixed baseline rerun and currently recommends `1245/1252/1260 MHz`.
- `pytest` cannot currently be used as the first-line validator in this workspace because `tests/conftest.py` imports `deepspeed`; use `python -m py_compile` plus direct module/CLI smoke runs unless the runtime environment is provisioned with DeepSpeed.
- `scripts/predict_freq_sweet_spot.py` can be smoke-tested locally with synthetic artifacts under `.context/freq-model-smoke/` before using real sweep outputs.
- Real prediction outputs are currently staged under `.context/real-v100-power-static-20260312/predictions/` for review before integrating with remote sweep workflows.
- Guardrail sweep outputs for the merged 8-point V100 dataset are staged under `.context/real-v100-power-static-20260312-merged/guardrail-sweep/`, with a measured-vs-modeled summary table at `.context/real-v100-power-static-20260312-merged/guardrail_comparison.tsv`.
- Recalibrated overlay-based guardrail outputs are staged under `.context/real-v100-power-static-20260312-merged/guardrail-sweep-overlay/`, with the updated summary table at `.context/real-v100-power-static-20260312-merged/guardrail_comparison_overlay.tsv`.
- Baseline-relative tradeoff output versus `power_baseline_20260311_20b` is staged at `.context/real-v100-power-static-20260312-merged/baseline_tradeoff_vs_power_baseline_20b.tsv`.
- The fresh baseline rerun is logging via remote `screen` boot log `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_baseline_rerun_20260312_1512.log`, and the local attach record is in `.context/remote-screen-sessions.tsv`.
- The refreshed baseline-relative table against the completed rerun is staged at `.context/real-v100-power-static-20260312-merged/baseline_tradeoff_vs_power_baseline_rerun_20260312_151032.tsv`.
- Pareto skyline chart artifacts are staged under `.context/real-v100-power-static-20260312-merged/pareto-charts/`.
- Serial validation launch log for the low-frequency confirmation sweep is stored at `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_lowfreq_20260312_113029b.log`, and the local session record is `.context/remote-screen-sessions.tsv`.
- Use `VALIDATE_ONLY=1` before launch for topology and preflight validation.
- Verify `screen -ls` and `tail` on the boot log after launching detached runs.
- Confirm final Zeus summary appears in the training log for short runs.

## Error Tracking
- [2026-03-12] The first baseline-relative hardware-first refactor compiled but still produced unrealistic pure-analytic recommendations (first `727 MHz`, then `1597 MHz`) on the merged V100 dataset; adding throughput-scale centering plus a throughput-saturation prior corrected the balanced recommendation to about `1252 MHz` while keeping low runtime/energy-ratio error.
- [2026-03-12] Synthetic smoke validation showed low power-fit error but high throughput-fit error, so calibration quality should be checked on real sweep artifacts before trusting any curve-derived default recommendation.
- [2026-03-12] The first real V100 prediction achieved `throughput_mape≈10.79%` and `power_mape≈2.37%`, but still extrapolated the optimum below the sampled static range.
- [2026-03-12] Real follow-up runs at `1020/1027/1035 MHz` all exceeded the old `1200 MHz` point in `tokens/J`, validating the model's low-frequency search direction even though absolute predicted scales remain mismatched.
- [2026-03-12] Pure analytic calibration still mis-ranked guardrail-optimal frequencies on the merged 8-point dataset, but the observed-frequency interpolation overlay now restores the correct transition pattern across the main guardrail thresholds.
- No dedicated Sentry/Bugsnag integration.
- Important failure modes are currently tracked through logs, `events.jsonl`, and memory-bank notes.
- [2026-03-13] Remote `TP=2, PP=4, DP=2` refinement sweep session: `power_tp2pp4dp2_static_refine_20260313_1238c`. Completed Zeus summaries: `1185 MHz -> 449.706 s / 1089279.7 J / 2422.2 W / 0.601645 tokens/J`, `1192 MHz -> 449.1 s / 1099588.2 J / 2448.6 W / 0.596`, `1207 MHz -> 444.1 s / 1106275.4 J / 2490.9 W / 0.592`, `1215 MHz -> 440.6 s / 1111404.2 J / 2522.3 W / 0.590`.
- [2026-03-13] Full merged Pareto-first accuracy artifacts for `TP=2, PP=4, DP=2` live under `.context/real-v100-tp2pp4dp2-static-merged-20260313/predictions-pareto-first-full-20260313/`; use `prediction_report.md` for human review and `prediction.json` for per-frequency ratio error extraction.
- [2026-03-15] Updated PP-bubble rerun artifacts live under `.context/real-v100-tp2pp4dp2-rerun-20260315/`; the fresh analytic rerun reports `supported_sweet_spot=1020 MHz`, `runtime_ratio_mape≈1.7303%`, `energy_ratio_mape≈1.2042%`, and sampled-frequency Pareto overlap on `{1185, 1200, 1207, 1215, 1245, 1252, 1260}` with extra predicted frontier points at `1192` and `930 MHz`.
- [2026-03-15] Two-band corrected rerun artifacts live under `.context/real-v100-tp2pp4dp2-rerun-20260315-corrected/`; the corrected curve reports `supported_sweet_spot=1125 MHz`, `total_time_mape≈1.1971%`, `total_energy_mape≈0.9045%`, and sharply improves low-frequency sampled-point fit (`900/915/930 MHz`) while preserving the observed Pareto overlap on sampled frontier points except for one extra predicted `1192 MHz` point.

- [2026-03-15] Remote 50-step validation session: `power_tp2pp4dp2_pred50_20260315_134758` on `sd@v100x16-1`; launcher script is `.context/remote-launch-tp2pp4dp2-predicted50-20260315.sh`, boot log is `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_tp2pp4dp2_pred50_20260315_134758.log`.
- [2026-03-15] Final Zeus summaries for the corrected-band validation points: `1117 MHz -> 1187.3 s / 2649800.5 J / 2231.8 W / 0.618 tokens/J` from `/home/sd/Megatron-DeepSpeed/experiments/v100_tp2pp4dp2_pred50_1117_20260315_134807_DGX2-1/logs/v100_tp2pp4dp2_pred50_1117_20260315_134807_DGX2-1.log`, and `1125 MHz -> 1178.7 s / 2662089.4 J / 2258.5 W` from `/home/sd/Megatron-DeepSpeed/experiments/v100_tp2pp4dp2_pred50_1125_20260315_140919_DGX2-1/logs/v100_tp2pp4dp2_pred50_1125_20260315_140919_DGX2-1.log`.
- [2026-03-15] For quick curve-quality checks on a running job, a stable fallback is to combine mean recent iteration time with a 1-minute `nvidia-smi --query-gpu=power.draw` sample; on the in-flight `1125 MHz` run this produced `1175.7 s / 2274.6 W` equivalent estimates, already close to the final Zeus `1178.7 s / 2258.5 W` summary.

- [2026-03-15] Final Zeus summary for the third corrected-band validation point: `1132 MHz -> 1172.9 s / 2674848.9 J / 2280.5 W / 0.613 tokens/J` from `/home/sd/Megatron-DeepSpeed/experiments/v100_tp2pp4dp2_pred50_1132_20260315_143024_DGX2-1/logs/v100_tp2pp4dp2_pred50_1132_20260315_143024_DGX2-1.log`.
- [2026-03-21] Local legacy-topology revalidation artifact for the new weighted dual-node fit is saved at `.context/dual-node-validation-20260321/legacy_topology_revalidation.json`; use it to compare old-vs-new cross-node overhead replay on `legacy_tp1_pp4_dp4` and `legacy_tp2_pp2_dp4`.
- [2026-03-21] New local replay artifacts for the topology-residual cross-node model are `.context/dual-node-validation-20260321/current_replay_topology_residual_fit.json` and `.context/dual-node-validation-20260321/legacy_topology_revalidation_topology_residual.json`.
- [2026-03-21] Power-fixed replay artifact for the topology-residual dual-node predictor is `.context/dual-node-validation-20260321/current_replay_topology_residual_fit_powerfix.json`; this is the current best local checkpoint for `TP=2, PP=4, DP=2` dual-node curve quality.
- [2026-03-21] Generalization-first replay artifacts after backing off the overfit residual are `.context/dual-node-validation-20260321/current_replay_generalized_fit.json` and `.context/dual-node-validation-20260321/legacy_topology_revalidation_generalized_fit.json`.
- [2026-03-21] Consolidated multi-topology predictor-quality snapshot is saved at `.context/dual-node-validation-20260321/multi_topology_accuracy_summary_20260321.md` and `.context/dual-node-validation-20260321/multi_topology_accuracy_summary_20260321.json`.
- [2026-03-22] Real dual-node validation artifacts for current-code `TP=2, PP=2, DP=4` are saved at `.context/dual-node-validation-20260322/tp2pp2dp4_current_dual_real_vs_pred_20260322.json` and `.md`; measured points are `1072 MHz -> 466.5 s / ~2123 W / ~990.6 kJ`, `1080 MHz -> 447.4 s / ~2198 W / ~983.7 kJ`, `1087 MHz -> 459.3 s / ~2168 W / ~995.8 kJ` (cluster-est power/energy via `2x` node0 Zeus). Current generalized predictor is directionally close but still low-biases time/power on this topology (`avg_time_ape≈6.20%`, `avg_power_ape≈8.20%`, `avg_energy_ape≈13.91%`).
- [2026-03-31] `dual8_tp4pp1dp2_smoke990_20260331_153159_DGX2-1` first gets past missing-index-cache errors after syncing `2d90..._*` and `e1ad..._*` to `DGX2-2`, then fails earlier than training with `torch.distributed.DistBackendError` inside `torch.distributed.new_group()`. The first concrete NCCL lines are `include/alloc.h:228 NCCL WARN Cuda failure 2 'out of memory'` and `Failed to CUDA calloc async 16 bytes` on ranks `0-3` of `DGX2-1`.
- [2026-03-31] Live occupancy snapshot during the failing `2x4` smoke: `DGX2-1` is fully consumed by external `VLLM::Worker_TP0..15` processes (about `31472 MiB` on every GPU), and `DGX2-2` still has `/usr/bin/python3` leftovers on `GPU1-3` plus another `python` process on `GPU0` (~`4.1 GiB`). This explains why tiny NCCL allocations fail immediately and why the current one-sided watcher is insufficient.
- [2026-03-31] The active watcher `watch_and_launch_dual8_tp4pp1dp2_20260331.sh` no longer monitors `GPU0-3`; it now gates launch on `GPU8-11` on both hosts and logs `target_gpus=8,9,10,11` in each polling snapshot. Initial armed state shows `DGX2-1 GPU8-11` free while `DGX2-2 GPU8-11` are free but the host still fails the residual-process check because stale `/usr/bin/python3` ranks remain alive.

## [2026-04-25] V100 单节点 LLaMA-7B 能耗对比实验（统一口径）

### 实验配置
- **硬件**: DGX2-1, 8x V100-SXM3-32GB
- **拓扑**: TP=2, PP=2, DP=2
- **模型**: LLaMA-7B (32L / hidden=4096 / ffn=11008 / heads=32 / vocab=32000)
- **数据**: `chinese_wiki_llama_megatron_text_document`（LLaMA tokenizer 预处理）
- **训练**: 20 iterations, bf16, ZeRO-1, CPU Adam, recompute-granularity full
- **Zeus 口径**: 仅统计训练阶段（从 "before training step" 到 "after training is done"）

### 真实权重 5 频点结果
| 频率 | 时间(s) | 能耗(J) | 功率(W) | tokens/J | 能耗节省 | 能效提升 |
|------|---------|---------|---------|----------|----------|----------|
| Baseline (1380 MHz) | 467.2 | 787,358 | 1685.3 | 0.416 | — | — |
| Static 1260 MHz | 505.1 | 593,341 | 1174.7 | 0.552 | -24.6% | +32.7% |
| Static 1350 MHz | 488.2 | 638,014 | 1306.8 | 0.514 | -19.0% | +23.6% |
| Static 1455 MHz | 467.3 | 705,273 | 1509.1 | 0.465 | -10.4% | +11.8% |
| Static 1530 MHz | 452.4 | 760,476 | 1681.2 | 0.431 | -3.4% | +3.6% |

### Random Init 对照结果
| 频率 | 时间(s) | 能耗(J) | 功率(W) | tokens/J | 能耗节省 | 能效提升 |
|------|---------|---------|---------|----------|----------|----------|
| Baseline (1380 MHz) | 454.5 | 779,045 | 1713.9 | 0.421 | — | — |
| Static 1260 MHz | 511.0 | 596,680 | 1167.7 | 0.549 | -23.4% | +30.4% |

### 关键发现
1. **1260 MHz 为最佳能效点**，在真实权重和 random init 下均一致
2. **两种初始化方式结果高度一致**：能耗节省 ~24% vs ~23%，能效提升 ~33% vs ~30%
3. **证明锁频节能效果与权重初始化无关**
4. **V100 默认时钟 1380 MHz**（非 max 1597 MHz），节能空间相对有限但仍显著
5. **所有实验 Zeus 统计口径一致**，时间戳验证误差 < 1s

### 实验工件
- 真实权重 baseline: `v100_llama7b_realweight_newdata_baseline_20260425_134549_DGX2-1`
- 真实权重 static 1260: `v100_llama7b_realweight_static1260_formal20_20260425_140414_DGX2-1`
- 真实权重 static 1350: `v100_llama7b_realweight_static1350_formal20_20260425_141848_DGX2-1`
- 真实权重 static 1455: `v100_llama7b_realweight_static1455_formal20_20260425_142845_DGX2-1`
- 真实权重 static 1530: `v100_llama7b_realweight_static1530_formal20_20260425_143858_DGX2-1`
- Random init baseline: `v100_llama7b_randominit_newdata_baseline_20260425_145216_DGX2-1`
- Random init static 1260: `v100_llama7b_randominit_newdata_static1260_20260425_150208_DGX2-1`

## [2026-04-27] 4080 RTX 4080 SUPER LLaMA-7B 32L dual-node 预测模型校准

### 实验配置
- **硬件**: sd-1 + sd-2, 各 4× RTX 4080 SUPER 16GB, Ethernet eth0
- **拓扑**: TP=2, PP=2, DP=2, ZeRO-1 + CPU offload
- **关键 workaround**: sd-1 driver 575.51.03 (CUDA 12.9) + PyTorch 2.10.0+cu128 导致 `pin_memory()` 报 `CUDA error: invalid argument`，因此 ds_config 中 `optimizer.offload_device.pin_memory=false`
- **模型**: LLaMA-7B (32L / hidden=4096 / ffn=11008 / heads=32 / kv_heads=32 / vocab=32000)
- **数据**: `/tmp/llama_test_data_text_document_text_document`（10 samples, HFTokenizer, seq=2048）
- **训练**: 20 iterations, random init, bf16, micro=1, global=4, recompute-granularity full

### 观测结果（4 频点）
| 频率 | 时间(s) | 能耗(J) | 功率(W) | tokens/J | 相对 baseline |
|------|---------|---------|---------|----------|--------------|
| Baseline (2505 MHz) | 281.0 | 86,842 | 309.1 | 1.887 | — |
| Static 1800 MHz | 276.1 | 62,459 | 226.2 | 2.623 | time -1.7%, energy -28.1% |
| Static 1650 MHz | 300.8 | 66,154 | 219.9 | 2.477 | time +7.0%, energy -23.8% |
| Static 1200 MHz | 299.2 | 63,638 | 212.7 | 2.575 | time +6.5%, energy -26.7% |

### 关键观测
1. **1800 MHz 时间比 baseline 还快**（276s vs 281s），通信瓶颈主导，计算加速被通信等待掩盖
2. **1200 MHz 能效最高**（2.575 tok/J），功率最低（212.7W），时间代价可控（+6.5%）
3. **功率随频率单调上升**：212.7W → 219.9W → 226.2W → 309.1W

### 预测模型验证
| 模型 | 时间公式 | 功率公式 | Time MAPE | Power MAPE |
|------|---------|---------|-----------|------------|
| **独立拟合** (4点, 3 DOF) | `T = 28.101×(2505/f)^0.754 + 251.370` | `P = 211.23 + 97.89×(f/2505)^5.731` | 2.16% | 0.06% |
| **硬件先验** (b=1.046, exp=8.0, 2 DOF) | `T = 17.911×(2505/f)^1.046 + 261.871` | `P = 216.12 + 93.23×(f/2505)^8.0` | 2.16% | 0.89% |

**惊人发现**：两种模型时间 MAPE 完全相同（2.16%），但独立 b=0.754 与先验 b=1.046 差异 29%，独立 exp=5.731 与先验 exp=8.0 差异 28%。原因：4 点分布范围有限（1200-2505 MHz），参数间存在相关性冗余，不同 (b, exp) 组合可通过调整 (a, c, P_static, P_dynamic) 达到同等拟合效果。

**结论**：在有限频点范围内，硬件参数的"绝对真值"难以仅凭 4 点确定；但模型的**预测精度**不受此影响。

### Sweet Spot（独立 4 点拟合）
| 频率 | 时间Δ | 功率Δ | 能耗Δ | tokens/J |
|------|-------|-------|-------|----------|
| 1200 | +7.5% | -31.2% | -26.1% | 2.565 |
| 1300 | +6.4% | -30.9% | -26.5% | 2.580 |
| **1400** | **+5.5%** | **-30.5%** | **-26.7%** | **2.587** ← 最优 |
| 1500 | +4.7% | -30.0% | -26.7% | 2.586 |
| 1600 | +4.0% | -29.2% | -26.4% | 2.576 |
| 1800 | +2.8% | -26.9% | -24.8% | 2.523 |

### 实验工件
- Baseline: `llama32l_4080_dual_baseline_20260427_154024_sd-1`
- Static 1200: `llama32l_4080_dual_static1200_20260427_163600_sd-1`
- Static 1650: `llama32l_4080_dual_static1650_20260427_154717_sd-1`
- Static 1800: `llama32l_4080_dual_static1800_20260427_164400_sd-1`
- 预测脚本 v1 (2点): `.context/raw_predict_4080_llama32l.py`
- 预测脚本 v2 (4点): `.context/raw_predict_4080_llama32l_v2.py`
