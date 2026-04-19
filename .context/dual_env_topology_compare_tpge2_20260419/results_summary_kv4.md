# 双环境拓扑对比结果（TP>=2，kv4 修正版）

## 实验口径

- 运行环境：
  - Ethernet：`sd-1 + sd-2`，每机 `GPU 0,1,2,3`
  - IB：`v100x16-1 + v100x16-2`，每机 `GPU 8,9,10,11`
- 共同 workload：
  - 模型：`36 layers / hidden 2048 / ffn 11008 / heads 16 / kv_heads 4`
  - 数据集：`qwen_data_text_document`
  - 并行拓扑：
    - `TP=2 / PP=2 / DP=2`
    - `TP=4 / PP=2 / DP=1`
  - 训练参数：`micro_batch_size=1 / global_batch_size=4 / seq_length=2048 / train-iters=20`
  - 优化配置：`ZeRO-1 + CPU optimizer/offload`
  - 频率策略：`baseline`

## 说明

- 最初尝试使用 `kv_heads=2` 跑 `TP=4 / PP=2 / DP=1` 时失败，原因是 `num_key_value_heads=2` 不能被 `TP=4` 整除。
- 为保证两种拓扑在同一个合法模型上公平对比，本轮正式结果统一改为 `kv_heads=4` 后重跑。

## Zeus 汇总

| 环境 | 拓扑 | 总时间 (s) | 总能耗 (J) | 平均功率 (W) |
| --- | --- | ---: | ---: | ---: |
| Ethernet | `TP=2 / PP=2 / DP=2` | 129.386 | 40490.202 | 312.942 |
| Ethernet | `TP=4 / PP=2 / DP=1` | 158.058 | 48914.498 | 309.471 |
| IB | `TP=2 / PP=2 / DP=2` | 155.205 | 105340.952 | 678.719 |
| IB | `TP=4 / PP=2 / DP=1` | 127.520 | 97714.232 | 766.267 |

## 同环境内拓扑变化

以 `TP=2 / PP=2 / DP=2` 为基准：

| 环境 | 对比 | 时间变化 | 能耗变化 | 平均功率变化 |
| --- | --- | ---: | ---: | ---: |
| Ethernet | `TP=4 / PP=2 / DP=1` 相对 `TP=2 / PP=2 / DP=2` | +22.16% | +20.81% | -1.11% |
| IB | `TP=4 / PP=2 / DP=1` 相对 `TP=2 / PP=2 / DP=2` | -17.84% | -7.24% | +12.90% |

## 工件位置

- Ethernet manifest：`artifacts/eth/eth_topology_compare_tpge2_kv4_20260419_manifest_20260419_093732.txt`
- IB manifest：`artifacts/ib/ib_topology_compare_tpge2_kv4_20260419_manifest_20260419_173732.txt`
- Ethernet runs：
  - `artifacts/eth/eth_topology_compare_tpge2_kv4_20260419_tp2pp2dp2_20260419_093732_sd-1/`
  - `artifacts/eth/eth_topology_compare_tpge2_kv4_20260419_tp4pp2dp1_20260419_094109_sd-1/`
- IB runs：
  - `artifacts/ib/ib_topology_compare_tpge2_kv4_20260419_tp2pp2dp2_20260419_173732_v100x16-1/`
  - `artifacts/ib/ib_topology_compare_tpge2_kv4_20260419_tp4pp2dp1_20260419_174155_v100x16-1/`
