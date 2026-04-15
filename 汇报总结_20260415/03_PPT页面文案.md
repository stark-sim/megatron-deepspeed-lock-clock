# PPT 页面文案

## 第 1 页

标题：

`Baseline 很容易启动，但不是天然最节能`

正文：

- 在 Megatron-DeepSpeed 中，baseline 对应默认 GPU 行为下的原始训练路径。
- 这个路径最符合用户直觉，但未必落在能效最优点。
- 因此我们关注的问题不是“能不能降频”，而是“能不能在不明显拉长训练时间的前提下降低功耗”。

## 第 2 页

标题：

`公平对比的关键：除了时钟策略，其他都不变`

正文：

- baseline 使用 `EXPERIMENT_MODE=baseline`
- static 使用 `EXPERIMENT_MODE=static`
- 模型、数据、拓扑、训练步数和 batch 配置保持一致
- Zeus 统一采集训练时间、平均功率和总能耗

## 第 3 页

标题：

`为什么演示选用 1.5B 级小模型`

正文：

- 比 7B 更适合 live demo 和汇报展示
- 启动快，失败成本更低
- 仍保留 Qwen-style 架构和正式 launcher 路径
- 适合完整展示 baseline -> static -> compare 的受控实验流程

## 第 4 页

标题：

`展示流程：validate -> baseline -> static -> compare`

正文：

- 第一步，先做 preflight，确认 GPU、数据集、tokenizer 和 launcher 正常
- 第二步，运行 baseline，建立默认参考点
- 第三步，运行一个或多个 static 频点
- 第四步，直接从 Zeus 指标生成对比表

## 第 5 页

标题：

`结果展示只看三类核心指标`

正文：

- 总训练时间 `time_s`
- 平均功率 `avg_power_w`
- 总能耗 `energy_j`

底部强调语：

`如果 static 相比 baseline 的时间变化很小，但功率和能耗明显下降，那么这个频点就是值得保留的候选点。`

## 第 6 页

标题：

`预测层的作用不是替代实验，而是减少 sweep 成本`

正文：

- 真实节能证据来自 baseline/static 对照实验
- 预测层负责帮助用户缩小值得验证的频点范围
- 因此更准确的定位是 `frequency-selection assistant`
- 这让整套方法更容易迁移到新拓扑和新网络环境

## 第 7 页

标题：

`最终要证明的是：系统能找到“合适频点”`

正文：

- baseline 给出默认起点
- static 给出受控对照
- Zeus 给出统一证据
- predictor 让“找频点”这件事从手工试错变成可辅助的流程
