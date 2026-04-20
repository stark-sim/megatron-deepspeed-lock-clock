# Tech Context
[2026-04-20] **Python 环境 bring-up 已被脚本化，且当前必须把“安装环境”和“运行时预热”分开处理**：
- 新增脚本：
  - `scripts/setup_python_env.sh`
  - `scripts/activate_runtime_env.sh`
  - `scripts/verify_python_env.py`
- 运行原因：
  - 这个项目的可运行性不只取决于 Python 包是否已安装，还取决于 clean shell 下 `TORCH_EXTENSIONS_DIR`、`TMPDIR`、`PYTHONPYCACHEPREFIX`、`TRITON_CACHE_DIR` 是否指向可写且低延迟的位置
  - `DeepSpeedCPUAdam`、`deepspeed.ops.adam.FusedAdam`、`apex` layer norm / optimizer 路径都可能在首次运行时触发 JIT 或扩展加载；因此单纯 `pip list` 正常不代表训练入口就能启动
- 当前脚本化策略：
  - `setup_python_env.sh` 负责建 `conda` 环境、装包、预编译 `CPUAdam/FusedAdam`、安装 `apex`
  - `activate_runtime_env.sh` 负责把 runtime cache/JIT 写入统一导到 `/dev/shm`
  - `verify_python_env.py --warmup` 负责做一次真实的小步预热，直接暴露 `adam/apex/Megatron` 的首次启动问题
[2026-04-20] **`sd-1/sd-2` 上的 RTX 4080 SUPER 高频锁频范围已通过 runtime 预检确认可继续上探**：
- `nvidia-smi -q -d SUPPORTED_CLOCKS` 的 graphics clocks 高端尾部至少达到 `2820 .. 3105 MHz`
- 这意味着 Ethernet real-model formal sweep 不必在 `1395 / 1650 / 1950 MHz` 等点位人为截断
- 当前 `1500 / 1650 / 1800 / 1950 / 2100 / 2250 MHz` 都已实跑完成；若还需要继续加密曲线，可继续沿 `2400+ MHz` 区间上探，但从当前趋势看，高频端已更像是在验证“稳定但不更优”
[2026-04-20] **V100 内部真实模型同步的源端与断点形态已进一步核实**：
- 源端是 `sd@v100x16-2:/home/sd/models/Qwen2.5-7B`
- 该路径通过软链接实际指向完整 `Qwen2.5-7B-Instruct` 目录，`du -shL` 为约 `15G`
- 四个源 shard 分别约为 `3.7G / 3.6G / 3.6G / 3.4G`
- 目标端 `sd@v100x16-1:/home/sd/models/Qwen2.5-7B-Instruct-full` 当前的 `model-00002-of-00004.safetensors` 仅约 `317M`
- 现场无活跃复制进程，因此“怎么会这么慢”的更准确答案是：之前的顺序同步已经断掉，并非还在持续低速传输
[2026-04-20] **V100 真实权重存在一个“名字误导”和一个“残缺副本”运行时事实**：
- `sd@v100x16-2` 上的完整真实权重并不是以直观目录名出现：
  - `/home/sd/models/Qwen2.5-7B` 实际是软链接到 `Qwen2.5-7B-Instruct`
  - 链路为：`/home/sd/models/Qwen2.5-7B -> /home/sd/cache/modelscope/Qwen/Qwen2.5-7B-Instruct -> /home/sd/cache/modelscope/Qwen/Qwen2___5-7B-Instruct`
  - 该目录当前可见完整 `model-00001..00004-of-00004.safetensors`，总量约 `15G`
- `sd@v100x16-1` 现有的 `/home/sd/models/Qwen2.5-7B-Instruct` 不能再被当作“差两片的可补全副本”：
  - `model-00001-of-00004.safetensors = 1705896960`
  - `model-00004-of-00004.safetensors = 887808000`
  - 对照 `DGX2-2` 源端同名文件 `3945441440` / `3556377672`
  - 结论是 `DGX2-1` 的现有 shard 已经截断，继续在其上补 `00002/00003` 没有意义
- 运行策略含义：
  - 若要恢复 V100 线真实 7B 主实验，当前优先路径应是 `DGX2-2 -> DGX2-1` 的内部复制到新目录，再基于完整目录做 HF->Megatron 转换
  - `DGX2-1` 的外网/代理链路虽然仍不通，但已不再是继续推进真实 7B 的唯一 blocker
[2026-04-19] **真实 Qwen7B Ethernet 线当前需要两个额外的运行时约束**：
- `sd-2` 的根分区空间不足以承受训练输出 checkpoint：
  - 在删除前，`df -h /home/user` 仅余 `43M`
  - 当前训练型目录 `baseline_formal20...` 与 `baseline_smoke5...` 会分别占用约 `31G` / `50G`
  - 对真实 checkpoint benchmarking，当前稳定口径应默认设置 `DISABLE_SAVE_CHECKPOINT=1`
- clean-shell rerun 不能依赖隐式继承的 JIT/build 缓存环境：
  - 若不显式设置 `/dev/shm` 下的 `TORCH_EXTENSIONS_DIR`、`TMPDIR`、`PYTHONPYCACHEPREFIX`，`sd-2` 侧 `CPUAdamBuilder().load()` 会尝试写 `/home/user/.cache/torch_extensions/py310_cu128`
  - 当前该路径会在 `sd-2` 上报 `PermissionError: [Errno 13] Permission denied`
  - 这说明“能在某个旧 shell 成功”并不代表 launcher 自身已经完整编码了这些环境依赖
[2026-04-19] **`Qwen2.5-7B-Instruct` 的 HF 词表元数据与 tokenizer 值不一致，转换器必须按 checkpoint/config 对齐**：
- 在 `sd-1` 上直接核对得到：
  - `config.json.vocab_size = 152064`
  - `AutoTokenizer.vocab_size = 151643`
  - `len(tokenizer) = 151665`
- 仅依赖 tokenizer 值会让 `hf2megads` 在 embedding copy 时触发行数断言或后续 partition shape mismatch。
- 当前稳定修复是：
  - 若 HF 参数字典中存在 `model.embed_tokens.weight`，使用其 `shape[0]` 作为 `token_vocab`
  - 在模型构建前读取 HF `config.json.vocab_size`，把 `args.padded_vocab_size` 提升到满足 TP 可整除的最小合法值
[2026-04-19] **`sd-1/sd-2` 的真实 Qwen7B 本地转换运行时现已确认可用**：
- 两台 Ethernet 节点都可在 `/home/user/miniconda3/envs/tp4bit/bin/deepspeed` 下本地执行 `Qwen2.5-7B-Instruct` 的 `TP=2 / PP=2 / 4 GPUs` 转换
- 成功产物：
  - `sd-1`: `/home/user/Megatron-DeepSpeed/checkpoints/qwen25_7b_instruct_hf2megads_tp2pp2_fixvocab2_20260419_114318`
  - `sd-2`: `/home/user/Megatron-DeepSpeed/checkpoints/qwen25_7b_instruct_hf2megads_tp2pp2_sd2_20260419_114724`
- 两边输出目录都约 `15G`，都含 `latest=global_step0` 与 `67` 个 `global_step0` 顶层条目
[2026-04-19] **当前 launcher 的真实 checkpoint 接入仍是“同一路径 load/save”语义**：
- `scripts/run_experiment.sh` 中 `LOAD_CHECKPOINT=1` 会对同一个 `CHECKPOINT_PATH` 同时追加 `--load` 和 `--save`
- 这意味着若直接用现有 launcher 接真实预训练转换产物做短跑，最稳的方式要么是复用该 checkpoint 目录作为 run path，要么扩展 launcher 支持分离的 `LOAD_CHECKPOINT_PATH` 与 `SAVE_CHECKPOINT_PATH`
- 在正式 baseline/static 前，建议先做一次多节点 `--load` smoke，确认这一路径的目录语义不会污染或覆盖初始化 checkpoint
[2026-04-19] **Megatron topology legality now has an explicit GQA runtime fact**：
- For the current Qwen-style launcher path, `TP` must divide both `num_attention_heads` and `num_key_value_heads`; otherwise model build fails inside `ParallelAttention` with `AssertionError: <kv_heads> is not divisible by <TP>`.
- This surfaced concretely during the `TP>=2` topology comparison: `kv_heads=2` works for `TP=2` but is illegal for `TP=4`, so the final fair comparison switched to `kv_heads=4`.
- Practical implication: when planning topology sweeps that raise `TP`, verify the GQA shape first instead of assuming a previously valid model can be reused unchanged.
[2026-04-19] **Latest dual-node common-workload runtime facts**：
- The stable completed `2x4` dual-node sweep for both Ethernet and IB now uses:
  - `DATA_CACHE_PATH=<repo>/data/index-cache`
  - `TORCH_EXTENSIONS_DIR=/dev/shm/megatron_common_qwen3b_20260419/torch_extensions_*`
  - `TMPDIR=/dev/shm/megatron_common_qwen3b_20260419/tmp`
  - `PYTHONPYCACHEPREFIX=/dev/shm/megatron_common_qwen3b_20260419/pycache`
- Required per-node cache hashes for this workload:
  - Ethernet (`sd-1/sd-2`): `33c91528b53c7a971dc9e5a3b24c9665`
  - IB (`v100x16-1/v100x16-2`): `2025292d291ff386fedc1b73e7aace6c`
- Runtime implication: node-local `/dev/shm/.../index-cache` is not a drop-in replacement for `data/index-cache` on multi-node launches unless those hash files are preseeded on every node.
[2026-04-19] **Canonical launcher environment propagation has been hardened**：
- `scripts/run_experiment.sh` now rewrites managed `.deepspeed_env` keys from the current launch environment instead of carrying forward stale backup lines.
- This change was required to stop failed reruns from reusing old `TMPDIR` / `TORCH_EXTENSIONS_DIR` values and to ensure remote workers inherit the current `PATH`/`PYTHONPATH` needed for `ninja` and JIT extension builds.
[2026-04-12] **Current remote-vs-local artifact retention state for paper inputs**：
- The fresh formal IB run directories on `sd@v100x16-1` remain available and still hash-match the local copies under:
  - `.context/ib_formal_rerun_20260410/source_curated/`
  - `.context/ib_formal_rerun_20260410/target_final/`
- The Ethernet artifact situation is less stable:
  - `user@sd-1:/home/user/Megatron-DeepSpeed/experiments/eth_qwen3b_2x4_target_curve_20260408_sd-1` now retains only the root directory, `ds_config.json`, and `logs/`
  - a fresh search on `user@sd-2` no longer finds `eth_qwen3b_1x4_source_static*` or `*source_curve*20260409*` directories under `/home/user`
- Practical implication: for the current paper, the authoritative retained Ethernet formal artifacts are now the local curated directories:
  - `.context/eth_2x4_curve_eval_20260409/eth_qwen3b_1x4_source_curve_20260409_curated/`
  - `.context/eth_2x4_curve_eval_20260409/eth_qwen3b_2x4_target_curve_20260408_sd-1/`
- The current local audit summary is stored at `.context/paper/local_artifact_audit_20260412.md`
[2026-04-11] **Local paper figure generation is now part of the reproducible build path**：
- `.context/paper/generate_figures.py` uses the local workspace formal artifacts as the single source of truth:
  - `.context/ib_formal_replay_20260410/comm_bench_2x4_20260410_ibhist.json`
  - `.context/comm_bench_2x4_eth0_20260406_175803.json`
  - `.context/ib_formal_rerun_20260410/source_curated/`
  - `.context/ib_formal_rerun_20260410/target_final/`
  - `.context/transfer_eval_ib_2x4_to_2x8_rerun_topology_fixed_live_ib_20260411/transfer_prediction.json`
  - `.context/transfer_eval_ib_2x4_to_2x8_rerun_topology_fixed_live_ib_powerfix_20260411/transfer_prediction.json`
- `.context/paper/Makefile` now runs `make figures` automatically before `make` / `make quick`
- Because login-shell `python3` can resolve to the Xcode shim without `matplotlib`, the stable local figure-build interpreter is currently `$(HOME)/miniconda3/bin/python3` when present
- To avoid permission/caching noise under sandboxed local builds, figure generation now sets:
  - `MPLCONFIGDIR=.context/paper/.texlive-cache/matplotlib`
  - `XDG_CACHE_HOME=.context/paper/.texlive-cache/xdg-cache`
- Current paper build health after adding figures:
  - `main.pdf` still compiles successfully
  - total length remains `6 pages`
  - `main.log` warning surface remains limited to one bibliography underfull
[2026-04-11] **Local paper build path is now verified with BasicTeX plus user-mode TeX packages**：
- `pdflatex` is now available at `/Library/TeX/texbin/pdflatex` from the locally installed BasicTeX distribution
- The paper build currently also depends on user-mode TeX packages installed via `tlmgr --usermode`, confirmed for:
  - `ieeetran`
  - `algorithms`
  - `multirow`
  - `subfigure`
  - `courier`
- With those packages present, `.context/paper/Makefile` can successfully run both `make quick` and `make`
- Important local-build note: BasicTeX itself may be present while `make` still fails unless the above packages are installed into the user tree
[2026-04-11] **DGX2 formal artifact placement rule confirmed by remote inspection**：
- For the fresh dual-node IB formal runs, `DGX2-1` (launch / node0 side) is the authoritative writer for the full experiment artifact tree: `run.json`, `events.jsonl`, `hostfile_snapshot.json`, `preflight.json`, `topology.json`, `command.sh`, and `logs/*.log`
- `DGX2-2` can expose the same run directory name but may only contain local fragments such as `ds_config.json`; absence of `run.json` or logs on `DGX2-2` is therefore not evidence that the run was synthetic
- Reliable two-node sanity checks should use three signals from `DGX2-1`: `hostfile_snapshot.json` includes both hosts, `run.json/topology/preflight` match the intended GPU slice and world size, and the primary log contains `v100x16-2:`-prefixed training iterations
[2026-04-11] **Formal IB replay tooling/runtime facts after closing the fresh `2x4 -> 2x8` rerun**：
- `scripts/evaluate_transfer_prediction.py` currently enforces exact non-topology workload equality and includes `global_batch_size` in that gate; it will reject the canonical `2x4 / DP=2 / GBS=8 -> 2x8 / DP=4 / GBS=16` transfer with `source and target workloads differ outside topology fields`
- For this project's canonical world-size transfer, the stable path is a topology-fixed replay: calibrate on source samples, predict on target workload features, and explicitly preserve distinct `source_override` / `target_override` in the output artifact instead of forcing a fake workload match
- On the fresh transport-consistent rerun pair, switching benchmark provenance from live dual-node IB (`.context/ib_formal_replay_20260410/comm_bench_2x4_20260410_ibhist.json`, large-message `busbw≈18.76 Gbps`) to the historical single-node `111.48 Gbps` record changes time MAPE only `11.48% -> 11.43%`
- The fresh rerun also shows a strong local-GPU-count power scaling signal: source `2x4` observed `avg_power_w` is `591/629/675 W`, while target `2x8` observed `avg_power_w` is `1189/1257/1353 W`, almost exactly `~2x`; current predictor outputs remain near the source scale (`564/615/660 W`), so the dominant `~51.6%` power MAPE is likely tied to missing `4 GPU/node -> 8 GPU/node` power scaling rather than cross-node transport alpha
- After adding explicit `reference_gpus_per_node -> target.gpus_per_node` scaling in the power path, the same replay collapses `avg_power_mape` to `3.28%` and `energy_mape` to `7.86%` while leaving `time_mape` and `alpha_dp` unchanged
- Practical implication: the remaining mismatch on the fresh IB rerun is now mostly in base throughput / power scaling, not in the benchmark-bandwidth scalar alone
[2026-04-10] **Current `2x8` IB runtime caveat after the successful `990 MHz` retry**：
- The local/remote `megatron/training.py` now contains a `DISABLE_ZEUS_MONITORING` env gate, but the successful retry log `ib_dual16_tp4pp1dp4_diag_nozeus_990_20260410_202433_DGX2-1` still printed `Zeus power monitoring started`; current inference is that this env var is not yet propagating to rank0 through the present launcher path, so the retry should be interpreted as a plain successful rerun rather than proof that Zeus was bypassed
- Even after a run exits cleanly and both DGX2 nodes temporarily return to `0 MiB / 0%`, `DGX2-1` can be reoccupied quickly by external `VLLM::Worker_TP8..15`; before launching the next formal point, always recheck `GPU 8-15` idle state on both nodes
[2026-04-10] **Validated DGX2 dual-node `2x8` IB runtime facts**：
- The canonical `2x8` IB target path now succeeds with:
  - `MASTER_ADDR=192.168.205.201`
  - `GLOO_SOCKET_IFNAME=enp6s0`
  - `NCCL_SOCKET_IFNAME=enp6s0`
  - `NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9`
  - `NCCL_IB_DISABLE=0`
  - hostfile entries `v100x16-1 slots=8` / `v100x16-2 slots=8`
  - `DS_INCLUDE=v100x16-1:8,9,10,11,12,13,14,15@v100x16-2:8,9,10,11,12,13,14,15`
  - `LOCAL_GPU_INDICES=8,9,10,11,12,13,14,15`
  - smoke config `TP=4 / PP=1 / DP=4 / ZeRO-1 / GBS=16 / MICRO=1 / TRAIN_STEPS=2`
- For this `2x8` workload, both nodes must carry the mmap dataset cache hash `d1158a21c6d1be91201644dbce18ab32` under `/home/sd/Megatron-DeepSpeed/data/index-cache/`; missing `_doc_idx.npy`, `_sample_idx.npy`, or `_shuffle_idx.npy` on `DGX2-2` causes remote `GPTDataset` construction to fail before training starts
- After syncing that cache hash to `DGX2-2`, the detached smoke `ib_dual16_tp4pp1dp4_smoke_1080_20260410_192035_DGX2-1` completed cleanly and wrote correct `visible_gpu_indices=[8..15]` / `nproc_per_node=8` metadata
[2026-04-10] **Validated DGX2 dual-node launcher state after rerun hardening**：
- The canonical `2x4` IB smoke path now succeeds with:
  - `MASTER_ADDR=192.168.205.201`
  - `GLOO_SOCKET_IFNAME=enp6s0`
  - `NCCL_SOCKET_IFNAME=enp6s0`
  - `NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9`
  - `NCCL_IB_DISABLE=0`
  - hostfile entries `v100x16-1 slots=4` / `v100x16-2 slots=4`
  - `DS_INCLUDE=v100x16-1:8,9,10,11@v100x16-2:8,9,10,11`
  - `LOCAL_GPU_INDICES=8,9,10,11`
- `scripts/run_experiment.sh` now needs `.deepspeed_env` overlay sanitization: when preserving an existing root `.deepspeed_env`, only emit valid `KEY=VALUE` lines before appending `MEGATRON_*`; blank lines can break DeepSpeed launcher parsing
- `scripts/experiment_utils.sh:is_local_host_alias()` must recognize hostfile aliases via real interface IPs (`hostname -I`), not just by comparing hostnames, because on `DGX2-1` the alias `v100x16-1` resolves to `100.64.0.90` while `DGX2-1` itself resolves only to `127.0.1.1`
- In successful dual-node runs, Zeus currently reports monitored GPUs as `[0,1,2,3]` even when the absolute assigned devices are `8,9,10,11`; treat this as visible-device-relative logging unless contradictory NVML evidence appears
[2026-04-10] **Current DGX2 rerun availability / runtime caveats**：
- `DGX2-1` is not presently available for a clean `2x4` IB rerun: `lb`'s `vllm serve /share-data/models/Llama-3.1-70B-Instruct --tp 16` occupies all 16 GPUs, with `VLLM::Worker_TP0..15` holding about `9436 MiB` on every device
- Because of that host-level occupation, `scripts/preflight_check.sh --gpu-indices 8,9,10,11` can fail even when there are no training processes from this project on the requested slice
- On the same day, a clean single-node `deepspeed` bring-up on `DGX2-2` proved that `run_experiment.sh + .deepspeed_env + experiment_tracker.py` can now propagate `LOCAL_GPU_INDICES=8,9,10,11` into `run.json`
- However, that run also exposed a distinct runtime issue on `DGX2-2`: `deepspeed.runtime.utils.see_memory_usage()` crashes during model build because the imported `psutil` module lacks `virtual_memory`
- Root cause of the `psutil` issue is now concrete: on `DGX2-2`, `/home/sd/.local/lib/python3.10/site-packages/psutil/` contains only `__pycache__/` and `tests/` (no `__init__.py` / source files), so `/usr/bin/python3` imports it as an empty namespace package; on `DGX2-1`, `psutil` resolves normally to `/usr/lib/python3/dist-packages/psutil/__init__.py`
[2026-04-10] **DGX2 IB launch provenance and transport facts clarified**：
- `DGX2-1` control-plane IP is `192.168.205.201` on `enp6s0`; `DGX2-2` is `192.168.205.202`
- On `DGX2-1`, `ibdev2netdev` shows `mlx5_0..3,6..9 -> ibp53s0, ibp58s0, ibp88s0, ibp93s0, ibp184s0, ibp189s0, ibp225s0, ibp230s0`; `ibstat` reports all eight ports `State: Active`, `Rate: 100`, `Link layer: InfiniBand`
- A live dual-node communication benchmark now works with:
  - `MASTER_ADDR=192.168.205.201`
  - `GLOO_SOCKET_IFNAME=enp6s0`
  - `NCCL_SOCKET_IFNAME=enp6s0`
  - `NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9`
  - `NCCL_IB_DISABLE=0`
- The recovered 2026-04-03 raw training artifacts do **not** record `NCCL_SOCKET_IFNAME` / `NCCL_IB_DISABLE` in `run.json`; the nearest collection wrappers for the same run-id pattern (`.context/dual8_tp4pp1dp2_collect_20260402.sh` and `.context/watch_and_recollect_20260402.sh`) still export `NCCL_SOCKET_IFNAME=tailscale0` and `NCCL_IB_DISABLE=1`
- The currently cited `111.48 Gbps` "IB benchmark" comes from a separate single-node record (`.context/transfer_2x4_to_2x8_ib_20260403.py` / `.context/paper/experimental_data.md`), not from the recovered DGX2 training artifact directories themselves
- Practical implication: benchmark provenance and training-artifact provenance must be paired before a replay can be labeled formal `IB` evidence
[2026-04-09] **Cross-node Predictor 网络基准路径更新**：
- **问题**: 用户确认 benchmark 不应直接充当最终通信罚时，也不应继续使用 `>50 Gbps` 这类二元分支去切换参数。
- **解决**: `fit_cross_node_penalty_model()` 改为先保留原始拟合得到的 `alpha_pp/alpha_dp/alpha_tp` 与 `beta_pp_*` 结构，再按 benchmark 的带宽/抖动结果连续缩放这些参数。
- **实现细节**:
  - 移除了 `high-speed vs slow-network` 的硬阈值路径
  - 新增按消息规模插值 benchmark `busbw` 的逻辑，用它估计当前网络下的代表性通信速度
  - `alpha_*` 按 `reference_bandwidth / observed_bandwidth(size)` 连续缩放，并叠加 jitter 因子
  - `CalibrationParams` 继续保留 benchmark 摘要与曲线字段，便于后续诊断/扩展，但当前 predictor 仍走“调参后的 alpha 模型”而非“benchmark 直接计时”
- **代码变更**:
  - `analysis/freq_model/cross_node.py`: 连续 benchmark-scaled `alpha/beta`
  - `analysis/freq_model/network.py`: 提取 benchmark 曲线
  - `analysis/freq_model/calibrate.py`: 持久化 benchmark 摘要/曲线元数据
  - `analysis/freq_model/model.py`: 保持 additive alpha penalty 主路径
- **验证**:
  - focused tests: `3 passed` for core cross-node/network cases
  - wiring tests: `2 passed` for calibration + CLI annotation path

[2026-04-09] **Transfer base-anchor cluster scaling 已加入**：
- **问题**: Ethernet `1x4 -> 2x4` transfer 的主要高估并不只来自 cross-node penalty；拆账后发现 target `base_step_time` 本身已显著偏慢，说明 source 校准得到的 base compute/memory anchor 被原样沿用到了更大的 target cluster。
- **解决**: 在 `model.py` 中新增 cluster-capacity scaling，把 base compute/memory limit 按 `(target_total_gpus / reference_total_gpus) * (target_pipeline_parallel_efficiency / reference_pipeline_parallel_efficiency)` 缩放；并在 `calibrate.py` 中持久化 `reference_total_gpu_count` 与 `reference_pipeline_parallel_efficiency`。
- **功耗处理**: 该 cluster scale 只进入 global throughput，不进入 power utilization 的 compute-limit 分母，因为当前 `avg_power_w` 更接近每节点功耗而不是全局 cluster 总功耗。
- **效果**:
  - Ethernet `2x4` formal `total_time_mape`: `40.17% -> 5.16%`
  - `step_time_mape`: `40.17% -> 5.16%`
  - `avg_power_mape`: `20.99% -> 12.38%`
  - `total_energy_mape`: `10.58% -> 10.42%`

[2026-04-03] **Cross-node Predictor 动态网络基准集成 - 已完成**：
- **问题**: 固定 cross-node 惩罚系数导致 IB 环境下 MAPE 98.5% 的严重高估
- **解决**: `fit_cross_node_penalty_model()` 现在接受 `network_bench_result` 参数，根据实测带宽动态调整惩罚系数；其最初实现是阈值分支，已在 [2026-04-09] 被连续缩放方案取代
- **实现细节**:
  - 高速网络 (>50 Gbps): alpha_dp = 5e-13 s/byte (接近零惩罚)
  - 慢速网络 (<50 Gbps): alpha_dp = 8.41e-10 s/byte (传统惩罚)
  - 惩罚降低倍数: ~1700x，匹配 IB 与 tailscale 的实际带宽差异 (111 Gbps vs 0.2 Gbps)
- **代码变更**:
  - `analysis/freq_model/cross_node.py`: 新增网络速度检测逻辑
  - `analysis/freq_model/calibrate.py`: 传递 network_bench_result 到 cross-node 模型
  - `.context/torch_nccl_comm_bench.py`: 通信基准测试脚本
  - `.context/run_comm_bench.sh`: 多节点启动脚本
- **验证**: 单元测试确认 IB 环境惩罚系数降低 1700x，35/35 测试通过
- **状态**: 实现完成，待完整 2x4 -> 2x8 transfer 端到端验证

[2026-04-03] **论文框架搭建完成**：
- **标题**: "Dynamic Network-Aware Cross-Node Performance Prediction for Distributed Deep Learning"
- **目录**: `.context/paper/`
- **结构**: IEEEtran 格式，8页会议论文模板
- **章节**: Abstract, Introduction, Background, Methodology, Experiments, Results, Related Work, Conclusion
- **数据**: `experimental_data.md` 整理所有关键实验数据
- **构建**: Makefile 支持快速编译和完整编译

[2026-04-03] **多拓扑扩展路线制定**：
- **Tier-1 (网络层)**: IB, RoCE 25/50/100G, Ethernet 1/10G, Tailscale VPN
- **Tier-2 (拓扑层)**: TP/PP/DP 组合变化，2x4/2x8/2x16/4x8 规模
- **Tier-3 (硬件层)**: V100 (当前), A100, H100
- **路线图文档**: `memory-bank/multi_topology_roadmap.md`

[2026-04-03] **Cross-node Predictor 设计改进（已废弃，改为上述实现）**：
- ~~问题发现: 当前 predictor 的 cross-node 通信惩罚是固定参数~~
- ~~实现思路: 在 `analysis/freq_model/network.py` 中添加轻量级 all-reduce benchmark~~
- **参考数据** (IB 环境):
  - 2x4 (DP=2) vs 2x8 (DP=4) step time 几乎相同 (~20s)
  - 单节点 NCCL busbw 测试: 111.48 Gbps (256MB message)
  - 说明在 IB 下，跨节点通信 overhead 可以忽略
[2026-04-03] **NCCL 网络配置重大更新**：
- **旧配置（已废弃）**：`NCCL_SOCKET_IFNAME=tailscale0`，`NCCL_IB_DISABLE=1`。该配置导致 NCCL 被迫通过 tailscale VPN 进行 socket 通信，实测 all-reduce 带宽仅 ~0.21 GB/s，tailscale 进程 CPU 占用 400%+，训练 step time 被拖慢约 4 倍。
- **新配置（已启用）**：移除 `NCCL_IB_DISABLE` 和 `NCCL_SOCKET_IFNAME` 硬编码，显式指定 `NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9"`，`NCCL_DEBUG=WARN`。
- **IB 设备状态**：`ibstat` 确认 8 个 IB 端口 (`mlx5_0..mlx5_3`, `mlx5_6..mlx5_9`) 均为 `State: Active, Physical state: LinkUp, Rate: 100, Link layer: InfiniBand`。
- **效果**：`2x4 TP=4 PP=1 DP=2` 的 step time 从 tailscale 下的 ~70s 骤降至 ~18-19s，确认此前所有 tailscale 数据均被严重污染。

[2026-03-26] User runtime caution for future real `2x16` / 32-GPU launches: one GPU may have a small background task, so occasional OOM is possible. Do not preemptively treat this as a blocker; only investigate if a real OOM actually appears during a run.


[2026-03-26] Some newly completed dual-node static runs can finalize without `freq_policy.mode/static_clock_mhz` in `run.json` even though Zeus interval metrics are present. Predictor-side artifact loading now tolerates this by inferring `static<freq>` from naming metadata; do not assume `freq_policy` is always populated on finished runs.


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
- [2026-04-05] User updated the next validation path: `user@sd-1` and `user@sd-2` are approved targets for Ethernet transport verification. On first login, verify GPU count, `eth0` path, repo parity, Python/Torch/DeepSpeed versions, and whether `.context/torch_nccl_comm_bench.py` can run before attempting full transfer validation.
- [2026-04-05] Ethernet preflight result on `user@sd-1` and `user@sd-2`: both hosts expose `4 x RTX 4080 SUPER`, default `/usr/bin/python3` is `3.12.3`, and repo path is `/home/user/Megatron-DeepSpeed`. Direct `python3` cannot import `torch`; usable training runtime currently comes from `source ~/miniconda3/etc/profile.d/conda.sh && conda activate tp4bit`.
- [2026-04-05] The currently reachable SSH context on `sd-1` / `sd-2` is sufficient for Ethernet benchmarking but not RDMA experiments: `ip -br addr` shows `eth0@if...` backed by `veth`, `/dev/infiniband` is absent, `rdma link show` returns no devices, and `/sys/class/infiniband` is empty. Treat these hosts as Ethernet-only unless the user later provides a different login path.
- [2026-04-05] PCI enumeration on both hosts shows Broadcom `BCM57416 NetXtreme-E Dual-Media 10G RDMA Ethernet Controller` NICs, but the active validation path should use standard NCCL socket transport over `eth0` with `NCCL_IB_DISABLE=1`.
- [2026-04-07] Repo parity for the benchmark path has been restored by direct file sync from the local workspace. Both `sd-1` and `sd-2` now have the updated predictor modules plus `.context/torch_nccl_comm_bench.py`, `.context/run_comm_bench.sh`, and `.context/test_cross_node_model.py`.
- [2026-04-07] First successful Ethernet benchmark artifact is `.context/comm_bench_2x4_eth0_20260406_175803.json` (fetched locally from `sd-1`). Predictor-side summarization yields `transport_label=eth0|ib_disable=1`, `effective_bandwidth_gbps≈0.2026`, `small_message_jitter_cv≈0.0153`, and `large_message_jitter_cv≈0.0072`, which clearly places this path in the slow-network branch.
- [2026-04-07] The original `1/4/16/64/256 MB` benchmark recipe is too aggressive for the current 10GbE path: the `256 MB` stage timed out under the existing `timeout 300` wrapper. The launcher now supports `SIZES_MB`, `BENCH_WARMUP_ITERS`, and `BENCH_ITERS` overrides; the successful configuration used `SIZES_MB="1 4 16 64"`, `BENCH_WARMUP_ITERS=3`, and `BENCH_ITERS=8`.
- [2026-04-07] `2x2` Ethernet target smoke is now proven runnable on `sd-1/sd-2` with `GPU 2,3` per host, `PP=2 / DP=2 / ZeRO-1 + CPU offload`, `NCCL_SOCKET_IFNAME=eth0`, and `NCCL_IB_DISABLE=1`. The successful log is `/home/user/Megatron-DeepSpeed/experiments/eth_qwen3b_target_validate_fixpy_20260407_sd-1/manual_cpuoffload_r5.log`; the representative steady-state point is `step2≈10.976s` with `skipped=0`.
- [2026-04-07] To align the workload, a matching single-node source smoke was run on `sd-2` (`/home/user/Megatron-DeepSpeed/experiments/eth_qwen3b_source_gb4_manual_20260407_sd-2/manual_gb4.log`) with the same model/GBS/PP and only the topology changed. The representative source point is `step2≈8.296s` with `skipped=0`.
- [2026-04-07] A local provisional time-only replay using those two points and the Ethernet benchmark predicts target `step_time≈14.683s` for an observed `10.976s`, so current Ethernet `step_time` APE is `≈33.8%`. This is materially worse than the previously established IB `<10%` result and should currently be treated as a temporary lower-confidence checkpoint rather than a final headline number.
- [2026-04-07] Before the 2026-04-08 recovery, the `sd-1/sd-2` remote tree was only partially synchronized with the local runtime path: benchmark-related files and `megatron/data/*` fixes were present, but short runs still omitted `run.json/events.jsonl`, which exposed that the local `megatron/training.py` / `megatron/experiment_tracker.py` / `megatron/power_monitor.py` / `pretrain_gpt.py` path had not yet been fully mirrored to these hosts.
- [2026-04-07] Before `zeus-ml` was installed into `tp4bit`, training on `sd-1/sd-2` printed `[Zeus] Failed to start: No module named 'zeus'` and could not produce standard energy/power interval metrics. This is now resolved for the Ethernet path.
- [2026-04-08] Standard-artifact `2x4` Ethernet smoke now succeeds on `sd-1/sd-2` using all four GPUs per host (`GPU 0-3`, `TP=1`, `PP=2`, `DP=4`, `ZeRO-1 + CPU offload`, `GBS=4`, `train-iters=2`). The representative run is `/home/user/Megatron-DeepSpeed/experiments/eth_qwen3b_2x4_fullgpu_smoke_20260408_sd-1/manual_2x4.log`, with `step1≈16.150s`, `step2≈8.303s`, and `skipped=0`.
- [2026-04-08] `run.json/events.jsonl` and Zeus interval metrics are now restored on `sd-1/sd-2`, but the current manual bring-up path still leaves `run.json.hostfile` and `run.json.topology` empty because only `scripts/run_experiment.sh` exports `MEGATRON_HOSTFILE_JSON` and `MEGATRON_TOPOLOGY_JSON`.
- [2026-04-08] Remote runtime parity on `sd-1/sd-2` must include `megatron/gpu_freq_manager.py` whenever `megatron/experiment_tracker.py` is synced. A partial sync reproduces `ImportError: cannot import name 'collect_nvml_device_snapshot'` before training startup.
- [2026-03-18] Dual-node V100 bring-up target is now `sd@v100x16-1` (`DGX2-1`) + `sd@v100x16-2` (`DGX2-2`) over `192.168.205.201/202` on `enp6s0`; `tailscale0` is reachable too, but local LAN is lower-latency for first 32-GPU startup checks.
- [2026-03-18] `sd@v100x16-2` has `/home/sd/Megatron-DeepSpeed` plus dataset files, but it is a repo snapshot without `.git`; for predictor/launcher parity, code had to be resynced from `sd@v100x16-1`.
- [2026-04-19] V100 rerun preflight re-confirmed that this old split state still exists: `DGX2-1` currently runs a dirty historical worktree near `6629a33`, while `DGX2-2` still exposes `/home/sd/Megatron-DeepSpeed` as a plain snapshot without `.git`. Treat any “latest-code rerun on V100” claim as invalid until both nodes are resynced from the local workspace.
- [2026-04-19] The current V100 copies of `data/qwen_data_text_document.bin` and `.idx` are tiny placeholder files (`968B` / `242B`), not the full real dataset used on the Ethernet line. Do not launch any “real dataset” V100 experiment against these files.
- [2026-04-19] The real Qwen checkpoint used on the Ethernet line, `qwen25_7b_instruct_hf2megads_tp2pp2_real_main`, is not present on either `sd@v100x16-1` or `sd@v100x16-2` yet.
- [2026-04-19] `sd@v100x16-1` already has a full HF snapshot for base `Qwen2.5-7B` at `/home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796/`; its config confirms the same `28L / 3584 / 18944 / 28 / kv4` architecture as `Qwen2.5-7B-Instruct`.
- [2026-04-19] Cross-environment transfer from `user@sd-1` to `sd@v100x16-1` for the `Qwen2.5-7B-Instruct` HF snapshot is functionally possible but currently very slow. Both `scp -r` and `ssh+tar` create the expected target snapshot path `.../models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28`, but throughput is low enough that this should not be treated as a fast setup path.
- [2026-04-20] `sd@v100x16-1` has multiple nominal download paths configured, but none currently produce a working HF self-download:
  - direct `huggingface.co` HTTPS: TCP connect times out
  - `http_proxy=http://192.168.205.201:7890`: proxy accepts CONNECT but TLS immediately ends with `unexpected eof while reading`
  - `socks5h://192.168.205.201:7890`: after vendoring `PySocks`, the client can establish a TCP session to the proxy (`ESTAB`), but downstream HTTPS still does not complete within a short timeout
  - `hf-mirror.com` direct: also times out
- [2026-04-20] To enable SOCKS tests on `DGX2-1` without pulling packages from the internet, a tiny local vendor copy of `PySocks` (`socks.py`, `sockshandler.py`) was placed under `/home/sd/Megatron-DeepSpeed/.context/pysocks_vendor/`; use `PYTHONPATH=/home/sd/Megatron-DeepSpeed/.context/pysocks_vendor:$PYTHONPATH` when retrying SOCKS-based HTTP clients.
- [2026-04-20] `sd@v100x16-1` contains two misleadingly promising local model roots that are both incomplete for conversion use:
  - `/home/sd/models/Qwen2.5-7B-Instruct/`
    - has tokenizer files and `model.safetensors.index.json`
    - but only `model-00001-of-00004.safetensors` and `model-00004-of-00004.safetensors` are present
  - `/home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796/`
    - currently only exposes `config.json`
    - no weight shards are cached locally
- [2026-04-20] A local conversion attempt on `DGX2-1` using `/home/sd/models/Qwen2.5-7B-Instruct` and the updated converter failed at tokenizer construction before reaching shard loading: the current remote Transformers/Qwen2 tokenizer path throws `TypeError: expected str, bytes or os.PathLike object, not NoneType` when pointed directly at that model directory. The host already has a working flat tokenizer directory at `/home/sd/Megatron-DeepSpeed/.context/qwen25_tokenizer_flat/`, so future retries should use that tokenizer path if/when a complete model directory becomes available.
- [2026-03-25] User correction: this project's canonical remote runtime is still the user-site Python 3.10 environment exposed via `~/.local/bin` and `~/.local/lib/python3.10/site-packages`, which is also what previous successful Megatron-DeepSpeed examples used. Do not treat the `tp4bit` Conda env as the default runtime for this repo.
- [2026-03-25] Reframe the previous Apex note as a non-canonical-path issue: the observed `amp_C` undefined `c10::cuda` symbol failure came from trying a `tp4bit` / copied-extension path, not from the established `.local` runtime that had already worked for this project. Future 32-GPU bring-up should first be reproduced in the `.local` environment before considering any Apex rebuild.
## Data / Tokenizer Context
- [2026-03-25] For the 32-GPU `TP=4, PP=1, DP=8, GBS=32, train-iters=2` smoke path, the train dataset cache hash is `87922b4c8bcca7cc5c0cfc95a72dd19d`. `DGX2-2` must have the matching `.dsc`, `_doc_idx.npy`, `_sample_idx.npy`, and `_shuffle_idx.npy` files under `/home/sd/Megatron-DeepSpeed/data/index-cache/` before the multi-node launch, or ranks on node1 fail with `FileNotFoundError` during `GPTDataset` construction.
- Dataset prefix: `/home/sd/Megatron-DeepSpeed/data/chinese_wiki_megatron_text_document`
- Tokenizer snapshot: `/home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9`

## Current Runtime Conventions
- `LR_WARMUP_ITERS` is made short-run friendly.
- `EVAL_INTERVAL` and `EVAL_ITERS` can be disabled safely for short runs.
- `DISABLE_CHECKPOINT=1` is the preferred mode for energy-only sweep jobs.
- `screen` sessions should use `env -i` with explicit `PATH` and `PYTHONPATH` to avoid environment leakage.
- Canonical remote launcher env for this project is `PATH=/home/sd/.local/bin:$PATH` with `PYTHONPATH=/home/sd/.local/lib/python3.10/site-packages`.
- Local offline freq-model unit tests currently need `python -m pytest --noconftest ...` because repo-level `tests/conftest.py` imports `deepspeed`, which is not available in this workspace environment.

[2026-04-01] **OOM 解决方案**：对于 `2x4` 等跨节点配置，20-step 阶段可能遇到 CUDA OOM。已通过设置环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 解决，该配置允许 CUDA 分配器使用可扩展段来更好地管理大内存分配。

- Remote repo note: `/home/sd/Megatron-DeepSpeed/run_experiment.sh` is stale relative to `/home/sd/Megatron-DeepSpeed/scripts/run_experiment.sh`; use the `scripts/` launcher for current short-run energy workflows.

[2026-04-19] **Remote Qwen checkpoint inventory (verified by SSH)**:
- `user@sd-1`:
  - full, non-quantized HF snapshots: `Qwen2.5-1.5B`, `Qwen2.5-1.5B-Instruct`, `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`, `Qwen3-4B-Instruct-2507`, `Qwen3-8B`
  - quantized only / not suitable as primary training checkpoint: `Qwen2.5-7B-Instruct-GPTQ-Int8`
  - other partial/non-CausalLM artifacts also exist, e.g. `Qwen2-Audio-7B-Instruct`
- `user@sd-2`:
  - full, non-quantized HF snapshots: `Qwen2.5-1.5B`, `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`, `Qwen3-0.6B`, `Qwen3-4B-Instruct-2507`
- `sd@v100x16-1`:
  - full HF snapshots: `Qwen2.5-0.5B`, `Qwen2.5-14B`
  - cache stubs / incomplete entries also exist for `Qwen2.5-1.5B`, `Qwen2.5-3B`, `Qwen2.5-7B`, `Qwen3-8B`, but they should not be treated as ready-to-run checkpoints
- `sd@v100x16-2`:
  - full local model dirs: `/home/sd/models/Qwen2.5-7B` (4/4 shards present), `/home/sd/models/Qwen3-8B`
  - `/home/sd/models/Qwen3-8B` currently appears incomplete: local dir has only `3/5` shard files while index references `5/5`; do not treat it as ready until repaired
- Practical implication:
  - easiest symmetric Ethernet candidates already present on both `sd-1/sd-2`: `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`, `Qwen3-4B-Instruct-2507`
  - easiest small IB candidate currently present as a full checkpoint is asymmetric (`0.5B` only on `v100x16-1`, `7B` only on `v100x16-2`), so IB formal runs will likely need checkpoint syncing before launch

## Cross-Node Penalty Model (当前状态 2026-03-23)
[2026-03-26] **Implementation status**: the predictor now has explicit plumbing for transport quality. `DerivedModelFeatures` carries `network_transport_label`, `network_effective_bandwidth_gbps`, and jitter metrics; `CalibrationParams` carries reference transport quality; and `model.py` multiplies cross-node penalties by relative bandwidth/jitter factors. The current script entry point for this is `scripts/predict_freq_sweet_spot.py --network-benchmark-json <path>`.

[2026-03-26] **Current real training transport for new dual-node controls**: `NCCL_SOCKET_IFNAME=tailscale0`, `NCCL_IB_DISABLE=1`, `NCCL_RAS_ENABLE=0`, `TORCH_NCCL_BLOCKING_WAIT=1`. Do not assume IB is active for the current measured `2x8` / `2x16` controls.

[2026-03-26] **Measured same-transport collective quality (`2 nodes x 8 GPUs`)**: under the above transport, a lightweight all-reduce benchmark gives `busbw ≈ 0.207-0.212 GB/s`. Large-message behavior is relatively stable (`cv ≈ 0.008-0.012` for `64-256 MB`), while smaller `16 MB` collectives are more jittery (`cv ≈ 0.136`). This suggests the predictor should distinguish baseline bandwidth from message-size-sensitive jitter.

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
