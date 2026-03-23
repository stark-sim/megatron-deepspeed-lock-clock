# Tech Context

## Repository
- Local workspace: `/Users/stark_sim/conductor/workspaces/megatron-deepspeed-lock-clock/athens`
- Remote workspace: `/home/sd/Megatron-DeepSpeed`
- Current local branch: `stark-sim/validate-real-topo`
- Remote git branch observed during runs: `main`
- Remote repo reality on [2026-03-16]: git still reports base commit `6629a33`, but the live validation environment is a dirty working tree with local modifications/untracked `analysis/`, `scripts/run_experiment.sh`, `.context/`, `experiments/`, and dataset/index-cache state.

## Core Stack
- Python-based Megatron-DeepSpeed training stack.
- DeepSpeed launcher for distributed training.
- NVML-based GPU frequency control.
- Zeus-based power monitoring integrated into training.
- Offline Python analysis layer for frequency-curve prediction implemented in `analysis/freq_model/` and `scripts/predict_freq_sweet_spot.py`.

## Remote Runtime Context
- [2026-03-22] Real dual-node static validation for `TP=2, PP=2, DP=4` on `sd@v100x16-{1,2}` now has three current-branch points at `1072/1080/1087 MHz` using `.context/dual_static_tp2pp2dp4_batch_20260322.sh` plus `NCCL_SOCKET_IFNAME=tailscale0`, `NCCL_IB_DISABLE=1`, `NCCL_RAS_ENABLE=0`, and `TORCH_NCCL_BLOCKING_WAIT=1`. Measured 20-step cluster-est metrics are roughly `466.5 s / 2123 W / 990.6 kJ`, `447.4 s / 2198 W / 983.7 kJ`, and `459.3 s / 2168 W / 995.8 kJ`. Compared with same-frequency single-node static references, runtime overhead is about `+10.7% / +6.9% / +10.3%` while cluster power drops about `-9.6% / -6.9% / -9.1%`.
- Primary V100 transfer-validation host: `sd@v100x16-1` (`DGX2-1`, `16 x Tesla V100-SXM3-32GB`, NvLink, Python `3.10.12`).
- On `sd@v100x16-1`, non-login SSH shells expose `python3` but not a `python` alias; use `python3` for ad-hoc `.context/` replay utilities.
- Secondary host checked on [2026-03-16]: `user@sd-1` (`4 x RTX 4080 SUPER 16GB`, Python `3.12.3`, repo path `/home/user/Megatron-DeepSpeed`). Keep it out of the active predictor-validation loop until the model is stable or we intentionally define a 4-GPU validation set.
- [2026-03-18] Dual-node V100 bring-up target is now `sd@v100x16-1` (`DGX2-1`) + `sd@v100x16-2` (`DGX2-2`) over `192.168.205.201/202` on `enp6s0`; `tailscale0` is reachable too, but local LAN is lower-latency for first 32-GPU startup checks.
- [2026-03-18] `sd@v100x16-2` has `/home/sd/Megatron-DeepSpeed` plus dataset files, but it is a repo snapshot without `.git`; for predictor/launcher parity, code had to be resynced from `sd@v100x16-1`.
- [2026-03-18] The `tp4bit` Conda env on both DGX2 nodes is the current dual-node bring-up target, but `sd@v100x16-1` cannot safely reuse copied Apex Python 3.12 extensions from `sd@v100x16-2`: the latest pilot fails on `amp_C` undefined `c10::cuda` symbols, so `DGX2-1` needs a local Apex rebuild against its own installed `torch`.
## Data / Tokenizer Context
- Dataset prefix: `/home/sd/Megatron-DeepSpeed/data/chinese_wiki_megatron_text_document`
- Tokenizer snapshot: `/home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9`

## Current Runtime Conventions
- `LR_WARMUP_ITERS` is made short-run friendly.
- `EVAL_INTERVAL` and `EVAL_ITERS` can be disabled safely for short runs.
- `DISABLE_CHECKPOINT=1` is the preferred mode for energy-only sweep jobs.
- `screen` sessions should use `env -i` with explicit `PATH` and `PYTHONPATH` to avoid environment leakage.
- Local offline freq-model unit tests currently need `python -m pytest --noconftest ...` because repo-level `tests/conftest.py` imports `deepspeed`, which is not available in this workspace environment.

- Remote repo note: `/home/sd/Megatron-DeepSpeed/run_experiment.sh` is stale relative to `/home/sd/Megatron-DeepSpeed/scripts/run_experiment.sh`; use the `scripts/` launcher for current short-run energy workflows.

## Cross-Node Penalty Model (当前状态 2026-03-23)
Goal: add a correction factor to the frequency predictor so that `T_multi(freq, N)` can be predicted from single-node calibration plus network overhead.

**验证状态：已通过真实双节点数据验证**

| 拓扑 | 时间 APE | 功率 APE | 数据来源 |
|------|----------|----------|----------|
| TP2PP4DP2 | 1.8% | 0.68% | 实测 (1185/1200/1215 MHz) |
| TP2PP2DP4 | 8.4% | 2.77% | 实测 (1072/1080/1087 MHz) |
| TP1PP4DP4 | 2.1% | 0.57% | 实测 (1185/1192/1200 MHz) |

所有拓扑时间预测误差均在 10% 以内，模型已具备实用精度。

**校准拓扑物理特征（8+8 GPU 双节点）：**

| 拓扑 | pp_cross_node_edge | dp_cross_node_group | 说明 |
|------|-------------------|---------------------|------|
| TP2PP4DP2 | 0.0 (PP 在节点内) | 1.0 (所有 DP 组跨节点) | 实测 |
| TP2PP2DP4 | 0.0 | 0.333 | 实测，单节点时间已实测更新 |
| TP1PP4DP4 | 0.0 | 0.333 | 实测 |

**拟合参数（基于真实数据）：**
- `alpha_dp = 5.73e-10 s/byte`（跨节点 DP allreduce 惩罚）
- `alpha_pp = alpha_tp = 0`（所有校准拓扑 PP 不跨节点）
- `beta_pp_wait`, `beta_pp_edge` 用于 PP 等待压力
- 功率模型：`power_base_drop ≈ 0.08`，APE < 3%

**关键校准数据（cross_node.py）：**
- TP2PP4DP2: 3 个静态频点（1185/1200/1215 MHz），单/双节点均实测
- TP2PP2DP4: 3 个静态频点（1072/1080/1087 MHz），单节点实测于 2026-03-23
- TP1PP4DP4: 3 个静态频点（1185/1192/1200 MHz），单/双节点均实测

**Key finding**: overhead varies significantly by topology — NOT a uniform factor. The dominant driver is likely:
- `n_pp_activations_cross_node = num_microbatches_per_pipeline × 2` (fwd+bwd)
  - TP2PP4DP2: 8 micro-batches (GBS=16/DP=2) → heaviest PP cross-node traffic
  - TP1PP4DP4 and TP2PP2DP4: 4 micro-batches (GBS=16/DP=4) → lighter
- `dp_allreduce_volume = model_size / TP` also varies with TP

**Proposed mathematical form:**
```
T_multi(N) = T_single + T_cross_node(N, topology)

T_cross_node = α_PP × C_PP + α_DP × C_DP

C_PP = (N-1) × num_microbatches × act_size_bytes   # PP activation sends
C_DP = (N-1)/N × model_size_bytes / TP              # DP AllReduce
```
Where α_PP and α_DP are network-speed coefficients (bytes/sec), calibrated from single→dual measurements.

**Node-count scaling hypothesis**: overhead scales as f(N-1), so:
- 2-node: factor (N-1)=1
- 4-node: factor (N-1)=3 → ~3× more overhead than 2-node
- 8-node: factor (N-1)=7 → ~7× more overhead

**Key insight for frequency curve**: cross-node comm time is fixed (bandwidth-limited), compute time decreases with higher freq. Therefore:
- Multi-node frequency-time curves are flatter at high frequencies (comm becomes the bottleneck)
- Per-GPU power is lower (~10-12%) because GPUs idle during cross-node waits
- The predictor must apply this as an additive time correction, not a multiplicative one

**Status**: the predictor now uses an explicit three-term cross-node wait decomposition. `analysis/freq_model/features.py` derives `cross_node_pp_bytes`, `cross_node_dp_bytes`, and `cross_node_tp_bytes`, where the TP term is modeled as per-microbatch shard-sync volume scaled by the TP group span across nodes. `analysis/freq_model/cross_node.py` fits the three coefficients with a small non-negative least-squares search over active sets. Current calibration is `alpha_pp≈8.30e-09 s/byte`, `alpha_dp=0`, `alpha_tp≈8.84e-11 s/byte`; replay APE on the three dual-node overhead points is roughly `2.4%`, `3.2%`, and `17.6%`, so the next step is still real static-frequency validation plus possible refinement of the low-overhead topology proxy / anchor.

**Dual-node launch env** (confirmed working 2026-03-20):
- Old `.local` Python 3.10 env: `PATH=/home/sd/.local/bin:...`, `PYTHONPATH=/home/sd/.local/lib/python3.10/site-packages`
- NCCL: `tailscale0`, `NCCL_IB_DISABLE=1`, `NCCL_RAS_ENABLE=0`, `TORCH_NCCL_BLOCKING_WAIT=1`
- MASTER_ADDR: `100.64.0.90` (tailscale IP of DGX2-1)
- Flat tokenizer: `/home/sd/Megatron-DeepSpeed/.context/qwen25_tokenizer_flat`
- index-cache `8bb7b7d55d333c6ccd218b4ed421eaa2` (val) and `02c886db19dc87ae1844eba656640304` (train) present on both nodes
- [2026-03-21] Current `scripts/run_experiment.sh` is not yet safe for dual-node static-frequency validation as-is: in multi-node mode it only locks clocks on the launch node, and its per-run `ds_config.json` lives under a node-local `RUN_DIR`, which breaks remote ranks unless that file is mirrored to peer nodes. The working workaround is a custom wrapper that writes a shared `ds_config` under `.context/`, locks clocks on both nodes via `sudo -n`, and passes an explicit `deepspeed --master_port`.
