# Active Context

## Current Focus
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
