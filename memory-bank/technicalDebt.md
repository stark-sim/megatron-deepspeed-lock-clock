# Technical Debt

## Known Debt Items
| Item | Severity | Effort | Impact | Added |
|------|----------|--------|--------|-------|
| ~~`2x4` 采集脚本未强制 GPU 8-11，导致实际运行在 GPU 0-3 并与外部进程竞争~~ | High | Low | 已修复：`.context/dual8_tp4pp1dp2_collect_20260402.sh` 使用 `--include` 强制绑定 GPU 8-11 | 2026-04-02 |
| `2x4` 三频数据存在时间异常：1080/1155MHz 比 990MHz 慢 27-34% | High | Medium | 数据质量存疑，不建议直接用于 predictor 校准 | 2026-04-02 |
| 同机其他用户的 CPU 密集型进程（如 `nsys stats`）会显著影响训练 step time | High | Low | 预检脚本已检查 GPU 空闲，但未检查 CPU 负载是否来自其他用户；需要把 CPU 负载阈值收紧或加入实验前等待机制 | 2026-04-02 |
| `run.json` finalize is inconsistent for some completed runs | Medium | Medium | Requires log/event fallback during analysis | 2026-03-11 |
| Distributed teardown emits noisy TCPStore/NCCL warnings after successful completion | Low | Medium | Makes logs noisier and can obscure true failures | 2026-03-11 |
| Remote launcher changes must be synced manually from local workspace | Medium | Low | Easy to forget and causes validation drift | 2026-03-11 |
| `sd-1/sd-2` 的 `tp4bit` 环境缺少 `zeus` | High | Low | Ethernet 短跑虽然能成功，但无法产出标准 energy/power interval 指标，正式 transfer 评估受阻 | 2026-04-07 |
| `sd-1/sd-2` 当前只同步了 benchmark/data 修复路径，未完全同步 tracker/runtime 文件 | High | Medium | 成功 run 仍缺 `run.json/events.jsonl`，迫使分析退回手工日志和 provisional replay | 2026-04-07 |
| Experiment comparison still relies on ad-hoc parsing commands | Medium | Medium | Slower analysis across many sweep points | 2026-03-11 |
| Offline freq model still ignores `num_attention_heads` / `num_key_value_heads` / `swiglu` in analytic feature formulas | Medium | Medium | Guardrails now reject mixed workloads, but true cross-shape transfer still lacks explicit modeling | 2026-03-15 |
| Predictor baseline thermal coefficient (0.65) is trend-inferred, not rigorously calibrated | Medium | Low | Current value reproduces "static 1800 > baseline 2505" qualitatively but may deviate quantitatively; needs dedicated baseline-vs-static same-frequency calibration | 2026-04-28 |
| Multi-GPU desync overhead is merged into thermal throttle coefficient | Low | Medium | ΔT_desync (NCCL wait for slowest GPU) and ΔT_thermal (effective frequency drop) are distinct physical effects; separating them would improve predictor interpretability and cross-topology transfer | 2026-04-28 |

## Refactoring Priorities
1. Add a small comparison/report script for multiple `run_id`s.
2. Improve tracker finalization robustness for no-checkpoint runs.
3. Consider reducing teardown warning noise if it masks genuine failures.

## Code Smells
- Important experiment truths are split across `run.json`, `events.jsonl`, and logs.
- Remote and local script versions can temporarily diverge during active iteration.

## Cleanup Plans
- Continue deleting obsolete checkpoint directories after verifying power-only results are preserved in logs/artifacts.
- Keep using `DISABLE_CHECKPOINT=1` for future sweep runs unless checkpoint recovery is explicitly needed.

## Dependency Debt
- None urgent beyond keeping Torch/DeepSpeed/Zeus integration validated after environment changes.

## Architecture Debt
- Experiment execution and experiment analysis are still coupled through manual shell snippets rather than a dedicated summary tool.

## [2026-04-02] 数据质量异常分析

### 观察
`2x4` 三频采集数据显示异常时间趋势：
```
990 MHz  : 1756.2s  (baseline)
1080 MHz : 2363.4s  (+34.6% slower) ⚠️ 异常
1155 MHz : 2234.9s  (+27.3% slower) ⚠️ 异常
```

### 预期 vs 实际
- **预期**：频率越高，运行时间越短（假设无 thermal throttling）
- **实际**：高频运行时间反而更长

### 可能原因（按概率排序）
1. **系统负载干扰** (高概率)：1080/1155 MHz 运行时可能有其他用户进程
2. **过热保护降频** (中概率)：高频触发 thermal throttling
3. **网络波动** (中概率)：跨节点通信质量不一致
4. **启动时间差异** (低概率)：不同时间点系统状态不同

### 诊断结果（2026-04-02 更新）
- **根因已确认**：采集脚本未强制 GPU 8-11，导致 DeepSpeed 回退到 GPU 0-3。
- **直接证据**：
  - 所有三个运行的 `logs/*.log` 和 `run.json` 均显示 `CUDA_VISIBLE_DEVICES=0,1,2,3`
  - 1080/1155 MHz 功率反而低于 990 MHz，符合 GPU 空闲等待特征
  - step time 呈现间歇性 spikes，不是 thermal throttle 的持续模式
- **已排除**：thermal throttling、网络质量波动、启动时间差异本身

### 缓解措施
- 后续实验必须：
  1. ~~检查 GPU 空闲状态再启动~~ → 已用 `scripts/preflight_check.sh` 自动化
  2. **修复采集脚本**，显式传入 `CUDA_VISIBLE_DEVICES=8,9,10,11`
  3. 记录 GPU 温度/频率遥测
  4. 考虑同频点重复测量验证稳定性
  5. 在同一时间窗口内完成对比实验
