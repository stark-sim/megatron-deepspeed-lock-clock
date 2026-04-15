# PPT 提纲

## 第 1 页：问题定义

标题：

`Baseline 很容易启动，但不一定是最节能的运行点`

要点：

- baseline 是 Megatron-DeepSpeed 的默认使用方式
- 默认运行点不一定是能效最优点
- 真正的问题是：能否在不拖慢训练的情况下节能

## 第 2 页：公平对比口径

标题：

`除了时钟策略，其他都不变`

要点：

- baseline: `EXPERIMENT_MODE=baseline`
- static: `EXPERIMENT_MODE=static`
- 模型、数据、拓扑、batch 和训练步数全部保持一致
- Zeus 统一采集 `time / avg_power / energy`

## 第 3 页：为什么做小模型 demo

标题：

`用 1.5B 级 demo 展示完整流程`

要点：

- 比 7B 更适合现场演示
- 启动快，风险低
- 仍然复用正式实验的 canonical launcher
- 更适合展示 baseline -> static -> compare 的完整链路

## 第 4 页：展示流程

标题：

`validate -> baseline -> static -> compare`

要点：

- 先做 `VALIDATE_ONLY=1`
- 再跑 baseline
- 再跑一个或多个 static 频点
- 最后生成 Zeus 对比表

## 第 5 页：结果表

标题：

`我们只看三类核心指标`

要点：

- 总训练时间
- 平均功率
- 总能耗
- 相对 baseline 的变化百分比

## 第 6 页：预测层的角色

标题：

`预测层不是证据本身，而是找频点的助手`

要点：

- baseline/static 对照实验负责提供真实证据
- predictor 不替代真实实验
- predictor 的价值是减少 sweep 成本
- 更准确的定位是 `frequency-selection assistant`

## 第 7 页：最终结论

标题：

`我们要证明的是“系统能找到合适频点”`

要点：

- baseline 给出默认起点
- static 给出公平对照
- Zeus 给出统一证据
- predictor 让找频点的过程更高效、可复用
