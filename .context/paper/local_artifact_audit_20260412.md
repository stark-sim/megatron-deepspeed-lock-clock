# Local Artifact Audit for Paper Inputs

Date: `2026-04-12`

## Scope

This audit checks whether the experiment artifacts currently required by the paper are
stored in the local workspace, rather than existing only on remote machines.

## Required Local Artifact Sets

The current paper text, `experimental_data.md`, and `generate_figures.py` depend on the
following local artifact sets:

1. `.context/ib_formal_replay_20260410/comm_bench_2x4_20260410_ibhist.json`
2. `.context/comm_bench_2x4_eth0_20260406_175803.json`
3. `.context/ib_formal_rerun_20260410/source_curated/`
4. `.context/ib_formal_rerun_20260410/target_final/`
5. `.context/eth_2x4_curve_eval_20260409/eth_qwen3b_1x4_source_curve_20260409_curated/`
6. `.context/eth_2x4_curve_eval_20260409/eth_qwen3b_2x4_target_curve_20260408_sd-1/`
7. `.context/transfer_eval_ib_2x4_to_2x8_rerun_topology_fixed_live_ib_powerfix_20260411/transfer_prediction.json`
8. `.context/transfer_eval_ib_2x4_to_2x8_rerun_topology_fixed_live_ib_powerfix_20260411/transfer_prediction_report.md`
9. `.context/transfer_eval_eth_qwen3b_1x4_source_curve_20260409_curated_to_eth_qwen3b_2x4_target_curve_20260408_sd-1/transfer_prediction.json`
10. `.context/transfer_eval_eth_qwen3b_1x4_source_curve_20260409_curated_to_eth_qwen3b_2x4_target_curve_20260408_sd-1/transfer_prediction_report.md`

All of the above exist locally in this workspace.

## Verification Results

### InfiniBand formal runs

The six current formal IB run directories required by the paper were compared against
their authoritative remote copies on `sd@v100x16-1`.

Compared runs:

- `ib_dual8_tp4pp1dp2_formal_990_20260410_20260410_161719_DGX2-1`
- `ib_dual8_tp4pp1dp2_formal_1080_20260410_20260410_162533_DGX2-1`
- `ib_dual8_tp4pp1dp2_formal_1155_retry_20260410_170335_DGX2-1`
- `ib_dual16_tp4pp1dp4_diag_nozeus_990_20260410_202433_DGX2-1`
- `ib_dual16_tp4pp1dp4_formal_1080_20260411_110907_DGX2-1`
- `ib_dual16_tp4pp1dp4_formal_1155_20260411_111702_DGX2-1`

For each run, the local and remote SHA256 digests match for the key artifact files:

- `run.json`
- `events.jsonl`
- `command.sh`
- `notes.md`
- `ds_config.json`
- `hostfile_snapshot.json`
- `preflight.json`
- `topology.json`

Conclusion: the current paper's formal IB source/target run artifacts are fully saved
locally and match the authoritative remote copies.

### Ethernet target curve

The local directory
`.context/eth_2x4_curve_eval_20260409/eth_qwen3b_2x4_target_curve_20260408_sd-1/`
contains:

- the three target run subdirectories with `run.json`, `events.jsonl`, `command.sh`,
  and `notes.md`
- top-level `ds_config.json`, `hostfile_snapshot.json`, `index.jsonl`, and
  `topology.json`

The current remote directory on `user@sd-1`,
`/home/user/Megatron-DeepSpeed/experiments/eth_qwen3b_2x4_target_curve_20260408_sd-1`,
now retains only:

- the root directory itself
- `ds_config.json`
- `logs/`

Conclusion: the local workspace now holds a more complete retained copy of the Ethernet
target curve artifact set than the current remote machine.

### Ethernet source curve

The local directory
`.context/eth_2x4_curve_eval_20260409/eth_qwen3b_1x4_source_curve_20260409_curated/`
contains the three source runs:

- `eth_qwen3b_1x4_source_static1005_20260408_sd-2`
- `eth_qwen3b_1x4_source_static1200_20260408_sd-2`
- `eth_qwen3b_1x4_source_static1395_r1_20260409_sd-2`

Each retained local run contains:

- `run.json`
- `events.jsonl`
- `command.sh`
- `notes.md`

On `user@sd-2`, a fresh search under `/home/user` did not find any of the above
`eth_qwen3b_1x4_source_static*` directories, nor a surviving `*source_curve*20260409*`
bundle.

Conclusion: the curated local source-curve directory is the retained copy that the
paper should rely on.

## Final Status

Status: `PASS`

The experiment artifacts currently required by the paper are present in the local
workspace. No additional remote-to-local copy operation was required for the current
paper dependency set.

Important retention note:

- The IB formal run artifacts exist both locally and on the authoritative DGX2 launch
  node, with matching hashes.
- The Ethernet target bundle is now more complete locally than on `sd-1`.
- The Ethernet source curve appears to survive only in the local curated directory.
