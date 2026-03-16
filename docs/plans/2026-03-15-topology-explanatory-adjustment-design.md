# Topology-Explanatory Adjustment Design

## Goal

Refine the offline frequency predictor so topology transfer errors are corrected through interpretable distributed-training mechanisms, not through another layer of topology-specific coefficients.

The immediate trigger is the new contrast across three validated 16×V100 topologies:

- `TP=2, PP=4, DP=2`: earlier transfer miss, with the model over-favoring lower frequencies.
- `TP=2, PP=2, DP=4`: transfer success, with the predicted `1080 MHz` default matching the measured minimum-energy point.
- `TP=2, PP=1, DP=8`: new transfer miss, with the predicted `975 MHz` default too low and the measured `1050 MHz` control winning on both total energy and total time.

The design target is to preserve the current model's good behavior on medium-bubble topologies while preventing it from inheriting the same low-frequency bias in no-bubble regimes.

## Why The Current Concept Breaks

The current predictor already distinguishes `TP`, `PP`, and `DP`, but it still compresses too much of the distributed-training story into a single communication-heavy correction path:

- a single communication-share proxy,
- a pipeline-efficiency term,
- and a topology-weighted correction intensity.

That is enough to capture a rough ordering between deeper and shallower `PP`, but it is not enough to explain why `PP=1` behaves differently from `PP>1` even when total communication volume is still nontrivial.

The main conceptual problem is that the model currently mixes together:

1. communication that is exposed directly on the critical path,
2. communication that is often overlappable with compute,
3. and communication that mostly changes operator scaling rather than creating idle waiting.

This makes the current correction layer too eager to interpret “more distributed work” as “lower frequencies become safer,” which is not consistently true.

## Official Performance Facts To Preserve

### 1. Data Parallelism is often communication-heavy but overlap-friendly

PyTorch's DDP design note explains that gradients are bucketed, and asynchronous `allreduce` launches when a bucket becomes ready during backward. Its performance advantage explicitly comes from overlapping allreduce with backward compute.

Implication for our model:

- `DP` communication should not automatically be treated as fully exposed wall-clock overhead.
- A larger `DP` factor can increase communication work without producing the same low-frequency sensitivity as pipeline waiting.

References:

- PyTorch DDP note: https://docs.pytorch.org/docs/stable/notes/ddp.html

### 2. ZeRO-1 changes memory redundancy more than forward/backward communication shape

DeepSpeed's ZeRO documentation states that Stage 1 partitions optimizer states while retaining the computational granularity and communication efficiency of classic data parallelism.

Implication for our model:

- For the current experiments (`ZERO_STAGE=1`), we should not model `DP` as if it introduced the stronger gather/scatter exposure of parameter-partitioning regimes.
- `DP` in our current stack is closer to gradient-synchronization pressure with overlap opportunity than to a hard per-layer blocking penalty.

References:

- DeepSpeed ZeRO docs: https://deepspeed.readthedocs.io/en/stable/zero3.html

### 3. Pipeline Parallelism introduces uniquely exposed waiting and activation traffic

DeepSpeed's pipeline tutorial explains that pipeline training works by splitting a batch into micro-batches. Activations are sent forward across stages, gradients are sent backward, and gradient accumulation is used to extract pipeline parallelism.

This means `PP` adds two effects that are qualitatively different from plain `DP`:

- exposed stage-to-stage activation/gradient transfers,
- and bubble / idle waiting when the number of micro-batches is not large relative to pipeline depth.

Implication for our model:

- `PP` should own the bulk of the “exposed low-frequency sensitivity” mechanism.
- Bubble should not merely inflate total communication volume; it should modulate how much of the topology cost is actually on the critical path.

References:

- DeepSpeed pipeline tutorial: https://www.deepspeed.ai/tutorials/pipeline/
- DeepSpeed pipeline docs: https://deepspeed.readthedocs.io/en/latest/pipeline.html

### 4. Tensor Parallelism is synchronization-heavy, but not the same as pipeline waiting

Megatron-LM papers frame tensor parallelism as intra-layer model parallelism. It enables larger models, but introduces extra synchronization around layer internals. NVIDIA's newer docs also note that increasing tensor-side distributed work can shorten compute sections and reduce communication-overlap headroom.

Implication for our model:

- `TP` should affect throughput scaling and synchronization pressure.
- But `TP` should not be treated as if it created the same “idle low-frequency-friendly” regime as `PP` bubble.
- The current model likely under-separates these two meanings.

References:

- Megatron-LM paper: https://arxiv.org/abs/1909.08053
- Efficient Large-Scale Training with Megatron-LM: https://arxiv.org/abs/2104.04473
- Megatron Core distributed optimizer / overlap docs: https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/dist_optimizer.html
- Megatron Core overlap config overview: https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.distributed.distributed_data_parallel_config.html

## What Our Three Topologies Are Actually Saying

For the validated 50-step experiments, the derived topology proxies look like this:

- `TP=2, PP=4, DP=2`
  - microbatches per step: `8`
  - pipeline efficiency: `0.7273`
  - bubble fraction: `0.2727`
  - communication share: `0.1918`
- `TP=2, PP=2, DP=4`
  - microbatches per step: `4`
  - pipeline efficiency: `0.8000`
  - bubble fraction: `0.2000`
  - communication share: `0.1851`
- `TP=2, PP=1, DP=8`
  - microbatches per step: `2`
  - pipeline efficiency: `1.0000`
  - bubble fraction: `0.0000`
  - communication share: `0.1535`

The important pattern is not just that communication share falls from `PP=4` to `PP=1`. The deeper point is this:

- when `PP` is present but moderate (`PP=2`), low-frequency bias is still plausible and the current predictor works,
- when `PP` disappears (`PP=1`), that same low-frequency bias should mostly disappear,
- but the current correction stack continues to act as if lower frequency is protected by exposed waiting.

This strongly suggests that the model is carrying too much of the `PP>1` correction behavior into `PP=1` topologies.

## Design Principle

The next adjustment should not ask:

- “How do we fit `TP=2, PP=1, DP=8` better?”

It should ask:

- “Which distributed-training costs are exposed on the critical path, which are overlapable, and how should frequency sensitivity differ across those categories?”

That distinction is expected to remain useful beyond the current V100/Qwen setup.

## Proposed Conceptual Refactor

### A. Split topology cost into three interpretable channels

Replace the current single topology-heavy correction interpretation with three channels:

1. `pipeline_exposed_cost`
   - stage-to-stage activation / gradient transfers,
   - bubble / idle waiting,
   - stage scheduling inefficiency,
   - should go to ~0 when `PP=1`.

2. `dp_overlapable_cost`
   - gradient synchronization and ZeRO-1-style DP overhead,
   - usually bucketed and at least partially overlapable with backward or optimizer work,
   - should grow with `DP`, but should not create the same low-frequency push as `pipeline_exposed_cost`.

3. `tp_sync_cost`
   - intra-layer synchronization pressure from tensor parallel collectives,
   - affects throughput scaling and overlap headroom,
   - should primarily influence compute-side throughput curvature, not idle-time-style low-frequency correction.

### B. Separate “volume” from “exposure”

Each channel should have two notions:

- amount of work (`bytes`, `events`, or normalized proxy),
- exposure on the critical path.

For example:

- `PP` can have moderate bytes but high exposure.
- `DP` can have substantial bytes but lower exposure because of overlap.
- `TP` can have smaller explicit byte volume than `DP` but strong synchronization effect on operator throughput.

### C. Attach low-frequency correction only to exposed cost

The current failure suggests the low-frequency correction should not be driven by total topology complexity. It should be driven mostly by exposed cost, especially pipeline-exposed cost.

In practice, that means:

- strong low-frequency correction only when `PP>1` and bubble exposure is material,
- weak or near-zero low-frequency protection when `PP=1`,
- `DP` should mainly change scale and residual power/runtime calibration, not sweet-spot direction by itself.

## Concrete Feature Changes

### 1. Add `pipeline_exposed_fraction`

Candidate inputs:

- `pipeline_model_parallel_size`,
- `microbatches_per_step`,
- existing bubble fraction,
- optional stage-count-based transfer term.

Expected behavior:

- exactly `0` when `PP=1`,
- increases as `PP` rises,
- increases when micro-batches become scarce relative to pipeline depth.

### 2. Add `dp_overlap_fraction`

Candidate inputs:

- `data_parallel_size`,
- current `ZERO_STAGE`,
- optional fixed overlap prior for ZeRO-1 / DDP-like runs.

Expected behavior:

- nonzero when `DP>1`,
- but bounded so it cannot dominate topology correction by itself,
- smaller exposed fraction for current ZeRO-1 than for more aggressive parameter-sharded regimes.

### 3. Add `tp_sync_fraction`

Candidate inputs:

- `tensor_model_parallel_size`,
- existing `tp_penalty`,
- optional compute-shortening prior.

Expected behavior:

- influences throughput scaling directly,
- does not act like idle-time protection,
- interacts with compute saturation more than with low-frequency correction.

## Model-Level Adjustment Rules

### Rule 1: Throughput low-frequency correction must be gated by `pipeline_exposed_fraction`

This is the main long-term concept to preserve.

If `PP=1`, the model should not carry over the same low-frequency optimism learned from `PP>1` regimes.

### Rule 2: Power correction should treat `DP` and `TP` as scale effects before topology effects

In the new `TP=2, PP=1, DP=8` miss, predicted total energy stayed numerically close because lower power and longer runtime nearly canceled out. That is a warning sign that power and runtime scales are both biased in a compensating way.

So power-side adjustment should:

- first improve absolute power scaling for `DP`-heavy, no-bubble runs,
- only then influence frequency ranking.

### Rule 3: `DP` should mostly affect how much communication can hide, not whether waiting exists

This is different from `PP`, where waiting is structural.

## Recommended Implementation Order

1. Refactor `analysis/freq_model/features.py`
   - add the three explicit topology channels,
   - keep the old aggregate communication share temporarily for compatibility.

2. Refactor correction intensity in `analysis/freq_model/model.py`
   - low-frequency throughput correction should depend mostly on `pipeline_exposed_fraction`,
   - a smaller residual term may depend on `dp_overlap_fraction` and `tp_sync_fraction`.

3. Keep calibration objective unchanged initially
   - do not immediately add more calibration knobs.
   - first test whether the explanatory feature split alone removes the `PP=1` miss while preserving the `PP=2` success.

4. Re-run the three known topology checks
   - `TP=2, PP=4, DP=2`
   - `TP=2, PP=2, DP=4`
   - `TP=2, PP=1, DP=8`

## Success Criteria

The redesign is successful if it produces all of the following simultaneously:

- keeps `TP=2, PP=2, DP=4` near the correct `1080 MHz` neighborhood,
- shifts `TP=2, PP=1, DP=8` upward enough that `1050 MHz` is no longer outside the recommended set,
- avoids reintroducing the original severe low-frequency bias on `TP=2, PP=4, DP=2`,
- preserves a physically interpretable story for why each topology moves the sweet spot.

## Long-Term Concepts To Preserve

These ideas should remain part of the project vocabulary even if formulas change:

- topology cost is not one thing;
- communication volume and critical-path exposure are different;
- `PP` mainly contributes exposed waiting and stage transfer;
- `DP` mainly contributes overlapable synchronization in the current ZeRO-1/DDP-like regime;
- `TP` mainly contributes intra-layer synchronization and compute-shortening effects;
- low-frequency protection should track exposed waiting, not generic distributed complexity.

## Recommended Next Action

Implement only the feature/concept split first, without adding new fitted coefficients. The first iteration should test whether better causal structure alone improves transfer ranking.
