# Academic PPT Redesign

## Goal

将现有偏展示型 PPT 收口为更具学术汇报风格的版本，避免“先摆 20%+ 成果、后补支撑”的表达方式。

## Design Decisions

1. 把 deck 主线改为：
   - 研究问题与公平对照口径
   - Baseline vs Static 的受控证据
   - Predictor 的方法、结果与边界
2. 明确区分两类证据：
   - `artifact-backed local formal replay`
   - `memory-bank preserved Zeus summaries`
3. 在 PPT 中显式加入：
   - Baseline 与 Static 的时间 / 平均功率 / 总能耗对比表
   - 相对 Baseline 的变化百分比
   - 当前可以稳妥宣称的 headline 与不能过度宣称的内容
4. 在学术化证据链稳定后，再强化为图表版：
   - 案例页用相对 baseline 多序列柱状图
   - 汇总结论页用 `runtime delta vs power delta` trade-off 图
   - predictor 结果页用 observed/predicted 对照图与 APE 图

## Planned Slide Structure

1. 标题页
2. 研究问题与汇报主线
3. 公平对比口径
4. 系统与方法
5. 实验环境与证据来源
6. 案例 A：Baseline vs Static
7. 案例 B：Baseline vs Static
8. 受控对照结论
9. Predictor 的角色
10. IB formal replay
11. Ethernet formal replay
12. 结论与边界
13. Q&A
