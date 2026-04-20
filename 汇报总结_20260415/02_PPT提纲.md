# PPT 提纲

## 第 1 页：标题页

标题：

`分布式训练 GPU 能效优化`

副标题：

`基于受控锁频、Zeus 统计与网络感知预测的实验研究`

## 第 2 页：研究问题与汇报主线

标题：

`Baseline 很容易启动，但不一定是最节能的运行点`

要点：

- baseline 是 Megatron-DeepSpeed 的默认训练路径
- 真正的问题是能否在不显著拉长训练时间的前提下降低功耗
- deck 先展示受控证据，再展示 predictor

## 第 3 页：公平对比口径

标题：

`除了时钟策略，其他都不变`

要点：

- baseline: `EXPERIMENT_MODE=baseline`
- static: `EXPERIMENT_MODE=static`
- 模型、数据、拓扑、batch、训练步数和 ZeRO 保持一致
- Zeus 统一采集 `time / avg_power / energy`

## 第 4 页：系统与方法

标题：

`实验层、观测层、预测层`

要点：

- `run_experiment.sh` 统一 baseline/static 启动
- Zeus 提供统一时间/功率/能耗指标
- predictor 用 benchmark 连续调整 cross-node penalty

## 第 5 页：实验环境与证据来源

标题：

`先区分 artifact-backed 结果与历史摘要`

要点：

- Baseline vs Static 两个主案例来自 V100 双机历史 Zeus 摘要
- IB 与 Ethernet predictor 结果来自当前本地 formal replay 工件
- 不把不同拓扑/不同网络的结果混成一个 headline

## 第 6 页：案例 A 对照结果

标题：

`V100 双机 TP=1, PP=4, DP=4：功耗下降约 24%-25%`

要点：

- 以相对 baseline 的柱状图展示 3 个 static 点的 `ΔTime / ΔPower / ΔEnergy`
- 右侧保留精简对照表和 baseline 锚点
- 明确支撑来源是 preserved Zeus summary

## 第 7 页：案例 B 对照结果

标题：

`V100 双机 TP=2, PP=2, DP=4：功耗下降约 33%-36%`

要点：

- 以相对 baseline 的柱状图展示 4 个 static 点的 `ΔTime / ΔPower / ΔEnergy`
- 右侧保留精简对照表和 baseline 锚点
- runtime 变化仅约 `-2.4%` 到 `+2.2%`
- 是当前最强的一组受控节能证据

## 第 8 页：受控对照结论

标题：

`可以稳妥写出的主结论`

要点：

- runtime 相对 baseline 基本不变
- avg power 相对 baseline 显著下降
- energy 相对 baseline 同步下降
- 用 `runtime delta vs power delta` trade-off 图直接展示 7 个点的整体分布
- 由此支撑“合适固定频点确实存在”

## 第 9 页：为什么还需要 predictor

标题：

`预测层不是证据本身，而是找频点的助手`

要点：

- baseline/static 提供真实节能证据
- predictor 减少在新拓扑上的 sweep 成本
- 关键改动是 benchmark-driven continuous alpha

## 第 10 页：IB formal replay

标题：

`IB formal replay：当前 predictor 的主要精度证据`

要点：

- 当前 formal 指标：`11.48% / 3.28% / 7.86%`
- 显式展示逐频点 observed / predicted 对比
- 说明剩余误差主要在 runtime 而非 power

## 第 11 页：Ethernet formal replay

标题：

`Ethernet formal replay：慢网络场景下 predictor 仍可工作`

要点：

- 当前 formal 指标：`5.16% / 12.38% / 10.42%`
- 结果需结合 `TP=1, PP=2` 的拓扑背景解释
- 不直接与 IB 做一维排名比较

## 第 12 页：结论与边界

标题：

`这份汇报真正证明了什么`

要点：

- baseline 不是天然能效最优点
- 合适 static 频点在当前项目中确实存在
- predictor 用于缩小候选频点，不是节能收益本身

## 第 13 页：Q&A

标题：

`谢谢`
