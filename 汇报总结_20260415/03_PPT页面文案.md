# PPT 页面文案

## 第 1 页：标题页

标题：

`分布式训练 GPU 能效优化`

副标题：

`基于受控锁频、Zeus 统计与网络感知预测的实验研究`

补充语：

- `Megatron-DeepSpeed baseline 对照 | V100 InfiniBand + RTX 4080 Ethernet`

## 第 2 页：研究问题与汇报主线

标题：

`Baseline 很容易启动，但不一定是最节能的运行点`

正文：

- baseline 是 Megatron-DeepSpeed 的默认训练路径，但默认 GPU 行为不一定处在能效最优点。
- 这次汇报要回答的不是“能不能降频”，而是“能否在不显著拉长训练时间的前提下降低功耗与能耗”。
- 因此汇报顺序必须是：先看 baseline/static 受控证据，再看 predictor 的跨拓扑结果。

## 第 3 页：公平对比口径

标题：

`除了时钟策略，其他都不变`

正文：

- baseline: `EXPERIMENT_MODE=baseline`
- static: `EXPERIMENT_MODE=static`
- 模型、数据、拓扑、batch、训练步数、ZeRO 与 precision 保持一致
- Zeus 统一提供 `time / avg_power / energy`

底部强调语：

`因此这里所有“功耗节省”都应理解为：在相同 workload 与拓扑下，相对 baseline 的受控变化。`

## 第 4 页：系统与方法

标题：

`实验层、观测层、预测层`

正文：

- 实验层：`run_experiment.sh` 统一 baseline/static 启动，并保留 topology / hostfile / preflight 工件
- 观测层：Zeus 统一记录时间、平均功率和总能耗
- 预测层：通过 network benchmark 连续调整 cross-node penalty，并加入 cluster-capacity / per-node power scaling

## 第 5 页：实验环境与证据来源

标题：

`先区分 artifact-backed 结果与历史摘要`

正文：

- Baseline vs Static 主对照案例来自 V100 双机历史 Zeus 摘要
- IB 与 Ethernet predictor 结果来自当前 workspace 内的 formal replay 工件
- 这一步的目的，是避免把“当前本地可复核结果”和“memory-bank preserved 历史摘要”混成同一类证据

## 第 6 页：案例 A 对照结果

标题：

`V100 双机 TP=1, PP=4, DP=4：功耗下降约 24%-25%`

正文：

- 这一页直接列出 baseline 与 `1252/1260/1267 MHz` 三个 static 点
- 同时给出 `ΔTime / ΔPower / ΔEnergy`，而不是只给一个 headline
- 这组结果说明：runtime 基本不变甚至略优，但平均功率与总能耗都稳定下降

## 第 7 页：案例 B 对照结果

标题：

`V100 双机 TP=2, PP=2, DP=4：功耗下降约 33%-36%`

正文：

- 这一页直接列出 baseline 与 `1072/1080/1087/1125 MHz` 四个 static 点
- runtime 相对 baseline 仅约 `-2.4%` 到 `+2.2%`
- 平均功率下降约 `32.8%` 到 `36.3%`，是当前最强的一组受控节能证据

## 第 8 页：受控对照结论

标题：

`可以稳妥写出的主结论`

正文：

- 在两组代表性 V100 双机案例里，runtime 相对 baseline 基本不变
- avg power 相对 baseline 显著下降
- energy 相对 baseline 同步下降
- 因此可以支撑“合适固定频点确实存在”这一命题

## 第 9 页：为什么还需要 predictor

标题：

`预测层不是证据本身，而是找频点的助手`

正文：

- 真实节能证据仍来自 baseline/static + Zeus
- predictor 的价值是减少在新拓扑、新网络上的 sweep 成本
- 当前 predictor 的关键改动，是 benchmark-driven continuous alpha，而不是“直接把 benchmark 时间当罚时”

## 第 10 页：IB formal replay

标题：

`IB formal replay：当前 predictor 的主要精度证据`

正文：

- 当前 formal aggregate 指标是 `11.48% / 3.28% / 7.86%`
- 这一页应显式展示逐频点 observed / predicted 对比，而不是只报单个 MAPE
- 这里更重要的结论是：当前正式 paired replay 的剩余误差主要在 runtime，而不再是 power/energy

## 第 11 页：Ethernet formal replay

标题：

`Ethernet formal replay：慢网络场景下 predictor 仍可工作`

正文：

- 当前 formal aggregate 指标是 `5.16% / 12.38% / 10.42%`
- 这一页需要明确提醒：Ethernet 结果对应 `TP=1, PP=2`，与 IB 主结果的 `TP=4, PP=1` 不同
- 因此它更适合支撑“predictor 可迁移到慢网络场景”，而不是直接和 IB 做一维排名

## 第 12 页：结论与边界

标题：

`这份汇报真正证明了什么`

正文：

- baseline 不是天然能效最优点
- 合适 static 频点在当前项目中确实存在
- predictor 用于缩小候选频点，而不是节能收益本身

底部强调语：

`不能把所有“20%+ 节能”都归功于 predictor，也不能把不同 topology/transport 的 MAPE 混成同一维比较。`

## 第 13 页：Q&A

标题：

`谢谢`
