# MegaBake: fresh performance, the Mirage gap, megakernel-only optimizations, and overfitting

2026-10-18, `main` at f70b54c. H200 MIG `3g.71gb` slice (60 SMs, ~2.2 TB/s, clocks shared with other tenants, 10–15% noise). bf16, 40 warm calls, KV capacity 128, 4-token prefill. Raw data: `artifacts/full_model_2026-10-18/*/report.json`. Research evidence: `results/*.json` (this directory) (10 items, file:line citations inside). Gain estimates below are analytical, not measured.

## 1. Fresh performance (µs)

| Model | Decode call, MegaBake | Decode kernel, MegaBake | Decode kernel, Inductor | Ratio | Decode call, Inductor (not graphed) | Prefill call, MegaBake | Prefill call, Inductor + CUDA graphs |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM-135M | 2,138 | 1,502 | 1,382 | 1.09 | 11,925 | 2,130 | **1,966** |
| SmolLM-1.7B | 3,311 | 2,974 | 2,425 | 1.23 | 10,323 | 3,339 | **3,060** |
| Qwen2-0.5B | 2,380 | **1,584** | 1,636 | 0.97 | 10,739 | 2,395 | **2,209** |
| DeepSeek-V3 draft 0.6B | 2,423 | **1,591** | 1,631 | 0.98 | 10,613 | 2,457 | **2,342** |
| Beam DFlash2 draft | 3,041 | 2,591 | 2,463 | 1.05 | 4,344 | 3,095 | **2,835** |
| gemma-2b | 4,306 | 3,590 | 3,107 | 1.16 | 8,681 | 4,061 | **3,582** |
| Gemma-3-270M | 2,798 | 2,080 | 1,032 | **2.02** | 9,048 | 2,597 | **1,595** |
| Qwen3.5-0.8B | 6,506 | 5,621 | 2,033 | **2.77** | 7,565 | fails (`cumsum`) | 4,414 |

Results are unchanged from 2026-10-17 within noise.

- **Decode call:** MegaBake wins 1.2–5x, but only against an Inductor run that is *not* CUDA-graphed. Inductor skips graphs because the KV caches are mutated inputs, so it makes 380–630 launches per call. **We have never measured against a graphed decode.** Every published megakernel reports against vLLM/SGLang with graphs on.
  - **Estimate for a graphed decode:** Inductor's 1.7B kernels (2.42 ms) plus graph bubbles is about 2.5–3.0 ms. MegaBake's call is 3.31 ms (2.84 ms pipelined).
- **Decode kernel time:** MegaBake ties or wins only on Llama/Qwen2-shaped models. It loses by 2–2.8x on Gemma 3 (sliding-window and sandwich-norm shapes) and on Qwen3.5 (hybrid gated-delta layers). This is the clearest measured sign of fitting to the tested family.
- **Prefill:** MegaBake loses to Inductor + CUDA graphs on every model, even at 4 tokens, and loses 3–21x at 128–300 tokens (ROADMAP §6).
- **Memory-bandwidth utilization:** about 50% on 1.7B and about 8% on 135M. Published hand-built megakernels reach 50–86% (ForgeMegakernel) and 78% (Hazy) on H100.

## 2. How far we are from Mirage

The Mirage megakernel compiler (MPK) is a persistent runtime with per-model builders. It does not use the Mirage superoptimizer: nothing in `python/mirage/mpk` calls the search.

| MPK idea | MegaBake | Notes |
|---|---|---|
| Event counters instead of grid barriers | **Built, not wired** | `make_event_schedule` (`schedule.py:487`), `publish_event`/`wait_for_event` (`runtime.py:210-245`) exist. `codegen.py:194` sends every table launch (every full model) to the barrier-only path. ROADMAP step 10. |
| Monotonic counters across iterations | Done | Same idea as MPK's iteration-tagged task ids. |
| gcd/lcm per-tile event grouping (37–118x fewer events) | Missing | |
| Polling | Partial | `atomic_cas` polling; MPK uses acquire load + `nanosleep`. |
| swapAB small-M linear | Done, broader | 16/32/64 token tiles vs MPK's 16. |
| Residual add in linear epilogue | Missing | Only the bias epilogue exists. |
| Fused QKV / gate-up weights | Missing | Gated-MLP matcher always rejected (`regions.py:2240`). |
| RMSNorm into linear | Partial | Folding recomputes the norm per CTA, not inside the GEMM. MPK's own Qwen3 builder no longer uses `rmsnorm_linear`. |
| Split-K with reduce-add into residual | Partial | Deterministic split-K built, off by default. MPK enables it only on B200. |
| Attention with q/k-norm, RoPE, KV append in one task | Partial | KV append and split-context are done; RoPE and q/k-norm are separate. No paged KV. |
| Cross-task weight preload into shared memory | Missing | ROADMAP step 9 (L2 variant). |
| On-device batching, argmax, multi-step loop | Missing | |
| MoE task family | Missing | |
| Scheduler/worker SM split, NVSHMEM | Not useful | No benefit on 60 SMs or a single GPU. |
| Superoptimizer search | Don't port | Fixed op vocabulary and a ≤5/9-op graph limit. Its equivalence test ignores casts and rounding. MPK itself doesn't use it. Generated MPK tasks ran at 0.16–0.275x hand-written speed. |

**Summary:** the frontend (generic FX, declared ops) is ahead of MPK. The runtime is one step behind: events are written but not emitted for full models. The task library lacks epilogue/prologue fusions and cross-op preload.

## 3. Optimizations only a single megakernel can do

A graph of ~500 kernels, even with CUDA graphs and programmatic dependent launch (PDL), cannot do the following:
- PDL waits for the whole previous grid to finish and flush, so there are no tile-level dependencies.
- It only overlaps adjacent kernels, and only when SMs are free. With 214 KB-smem GEMMs at 1 CTA/SM, only the tail overlaps.
- Idle SMs can't be handed future work.
- Pipelines restart empty in every kernel.

| Optimization | Why a kernel graph can't | MegaBake | Est. decode gain (135M / 1.7B) |
|---|---|---|---|
| Tile-level event dataflow (no global barrier, no straggler wait) | Kernel boundary = full barrier + flush | Built, not wired in table launch | −17–30% / −5–10% |
| Cross-op weight streaming (TPU 12-bank ring; Hazy smem pages; L2 prefetch by waiting/idle CTAs) | Next kernel can't start loading until launched; smem doesn't survive kernels | Missing (only TMA descriptor prefetch) | −3–7% / −8–15% |
| Keep the GEMM pipeline alive across phases (no drain + refill per GEMM) | Every kernel starts with an empty pipeline | Missing: new TMA pipeline per phase (`gemm.py:727,888`) | −3–7% (overlaps with the row above) |
| Idle-SM work: 32-tile projections leave 28 of 60 SMs free | Free SMs can't receive future work | Missing | Included in prefetch |
| Recompute-instead-of-barrier fusion (fold producers into consumers) | Needs a new fused kernel per pattern | Done (interval folding) | Already banked |
| Joint tile/SM-share tuning for concurrent phases | Concurrent kernels contend unpredictably | Missing | Small, generic |
| Multi-step decode loop with on-device argmax + next embedding | CUDA-graph while-nodes partly can, so measure against them | Missing | −25% / −10% per token end-to-end (host time, not kernel) |
| In-kernel speculative verify (8 rows ≈ 1 row of weight traffic in swapAB) | Possible but needs host round trips | Missing | 2–4x tokens/s at acceptance 3–4 (large effort) |
| Device-side token count instead of size buckets | Graphs need one capture per size (vLLM: ~67) | Missing (buckets pad) | Removes up to 2x padding waste |

**What is not megakernel-only:** epilogue/prologue fusion, swapAB and split-context attention exist in kernel-graph systems too. They are needed for parity, not an advantage.

**Don't build:** a dynamic work queue for batch-1 decode. Event Tensor measured 0.82–0.89x vs static scheduling; Hazy measured −1.9%.

## 4. Overfitting: confirmed

The compiler core contains no model names, and worker/SM counts come from the device. The fitting comes from these choices:

1. **No search for real models.** `compiler.py:429-431` forces `max_plans=1` for every table launch, which is every full model. Plans are the greedy seed, never measured.
2. **One GEMM config per launch.**
   - Fixed 64x64x64, 8 stages, literal shared-memory sizes (`gemm.py:1711, 1879-1881`).
   - swapAB is all-or-nothing per launch, with `SWAP_AB_MAX_ROWS=64`; the comment cites a 60-SM measurement (`gemm.py:1890-1896`). One GEMM above 64 rows flips every GEMM, which matches the prefill cliff.
   - Split-K is off because it showed "no gain on 60 SMs".
3. **Constants fitted to SmolLM and the slice:**
   - Fold limits 2/6/6 (`schedule.py:1384-1386`). With 256 threads x 4 elements, maps fold only up to 2,048 elements, which is SmolLM-1.7B's hidden size.
   - `DECODE_MAX_CHUNKS=16`, `DECODE_MIN_CHUNK_KEYS=64` (`attention.py:422`).
   - Tolerances justified by SmolLM depth (`compiler.py:236-248`).
4. **bf16 and the `sm90a` string only.** Providers check the architecture name, not capabilities. fp16 or any other GPU silently falls to slow generic bodies. `bodies/blackwell.py` is a `mma.sync` correctness baseline. `TargetFacts` has no L2, bandwidth, or barrier latency, so no cost model is possible yet.
5. **Llama-shaped matchers and wrappers.**
   - RMSNorm matches one aten spelling; RoPE matches rotate_half only. The head-axis stride heuristic already broke Gemma 3 once.
   - The published SmolLM decode numbers go through the hand-written `StatefulSmolLM`, not the generic `DecodeStep`.
   - GPU end-to-end tests use only the SmolLM fixture.
6. **The fresh numbers show it.** Llama/Qwen2-shaped models reach kernel parity. Gemma 3 and Qwen3.5 run 2–2.8x slower.

**References are not generic either.**
- MPK hard-codes 96/64 tasks per linear ("enough to cover all Qwen3"), a 153,600-row vocabulary pad, and split-K only when the compute capability is 100.
- vLLM hard-codes RMSNorm eps ∈ {1e-5, 1e-6} in 11 fusion patterns and gates one fusion on `hidden_size == 2880`.
- Inductor disables GEMM autotune below 68 SMs.

The working pattern (Inductor, CUTLASS nvMatmulHeuristics, Triton `prune_configs_by`, TileLang Carver) is: **tables keyed by device, never by model, clamped to the problem shape, ranked by an analytic model, corrected by measuring the top-k.**

## 5. Recommended order of work

| # | Action | Expected gain | Generality | Effort |
|---:|---|---|---|---|
| 0 | **Honest baselines:** a CUDA-graph decode (static KV, manual capture) and Inductor `max-autotune` in `benchmarks/full_model.py`. Report MBU. Add per-CTA start/end stamps to `tools/phase_profile.py` to measure barrier skew. | Decides what's real | — | S |
| 1 | **Events in the table launch.** Per-phase completion counters in the device table, reusing `make_region_event_schedule`. Acquire-load polling. Add a static deadlock/race validator. | 135M −17–30%, 1.7B −5–10% | Generic | M |
| 2 | **Richer `TargetFacts` + per-GEMM candidates + cost model.** Query L2, bandwidth, smem, capabilities, MIG; run a one-time calibration of barrier/event latency and bandwidth vs CTAs. Each GEMM picks tile/stages/swapAB/split-K; smem = max over chosen. Measure top 3–5 per shape; cache by (device fingerprint, shape, dtype). Re-enable search for table launches. | 1.7B −0.2–0.45 ms; fixes the >64-token prefill cliff | Removes the overfit constants | M–L |
| 3 | **Cross-phase weight prefetch.** L2 prefetch (`cp.async.bulk.prefetch.L2`) by waiting/idle CTAs first, then smem handoff with persistent pipeline state. Depth from bandwidth × latency and the L2 budget. | 1.7B −8–15% | Generic, derived from target | M |
| 4 | **Epilogue/prologue contracts in `bodies/interface.py`:** residual add, SiLU·mul from a paired gate/up GEMM, RMSNorm as a row-scale prologue (check bf16 exactness: HF rounds before the matmul), horizontal QKV merge. | 135M −8–15%, 1.7B −4–7% | Generic if expressed as contracts, not patterns | M |
| 5 | **Semantic matching:** match norm/RoPE/activation by scalar-expression normal form (eps and scale as values) instead of aten spellings. Retire `StatefulSmolLM`. | Fixes Gemma/Qwen3.5-class gaps | Generic | M |
| 6 | **Fold limits, attention chunking, tolerances** from cost and dtype/depth instead of constants. | Small on SmolLM; large on wide models | Generic | S |
| 7 | **Prefill:** tensor-core prefill attention and the >64-row GEMM path. Meanwhile, optionally fall back to Inductor + graphs above a measured crossover. | 3–21x at 128–300 tokens | — | L |
| 8 | **Multi-step decode loop** with on-device argmax; later, speculative verify. | −10–25% per token end to end; 2–4x with speculation | Generic | L |
| 9 | **CI overfitting guard:** compile a varied tiny-model zoo (LayerNorm, GELU, MQA, head_dim 256, sliding window, MoE, gated delta) against synthetic targets (60/108/132/148 SMs; sm80/90/100/120). Fail on any silent fallback to a generic body. | Prevents regressions | — | M |
| 10 | **Real SM100 provider** (TMA, tcgen05, TMEM, CLC) and an SM80 provider (cp.async, mma.sync). | Portability | HW-generic | L |

**Combined estimate for items 1–4:** 1.7B decode kernel about 2.0–2.3 ms (Inductor 2.42 ms), 135M about 0.8–1.0 ms (Inductor 1.38 ms). Measure after each step; the error bars are wide.

## 6. Lessons from vLLM's DeepSeek V4.1 stack (`flash_mla_mega_attn.py` and siblings)

Six further research items, added 2026-10-18.

### Is `flash_mla_mega_attn.py` the ideal output?

Copy its structure, not the kernel. The kernel is hard-coded to one model and one GPU family:
- head dim 512, RoPE 64, GPT-J RoPE;
- quant group 32, 8 heads per `wo_a` group, query heads padded to 64 or 128;
- SM100 only, NVFP4 cache.

It wins only at TP1, by up to 1.45x on GB300. From TP2 to TP8 it is within a few percent of the unfused path, and at small s_q it costs about 2 µs extra.

The structure is what MegaBake should generate automatically:
1. A tuned inner pipeline with declared absorbed work and input/output layouts.
2. Weights permuted once at load so the neighbouring GEMMs speak the kernel's layout.
3. A gate on both applicability and measured profitability.

It is an ideal body for one interval, not the whole program. Its limits show what the megakernel adds:
- The quantize epilogue forces FlashMLA to drop split-KV, leaving one cluster per token: 1 of 148 SMs busy for about 29 µs at s_q=1.
- `wo_a` waits for every token, and the KV-insert kernel is still a separate launch.

A megakernel can fill the idle SMs with prefetch and gate `wo_a` tiles on per-token counters. It can also keep both split-KV and a nonlinear epilogue, by running the epilogue in MegaBake's existing last-chunk merge.

### Fusion kinds and how generic they are

| Fusion | Mechanism | Compiler-expressible? | MegaBake |
|---|---|---|---|
| Weightless Q RMSNorm → softmax scale | `sm_scale·rsqrt(Σx²/d+eps)`; Q never rewritten | Yes: per-row scalar hook | Missing |
| RoPE in the load prologue | Rotate in registers on the way into tensor memory | Yes: elementwise prologue | Partial: folded recompute, still round-trips global memory |
| Inverse RoPE in the epilogue | fp32 conjugate rotation on the last 64 dims (position-dependent, so no weight can carry it) | Yes: generic epilogue, not a special case | N/A for tested models |
| Sink + normalization + quant scale in one multiply | `1/(l+exp2(sink−m))` × ue8m0 scale | Yes: epilogue expression | Missing (no sinks, no quant) |
| Quantized store in the consumer's layout | Scale granularity and MN-major layout dictated by DeepGEMM | Yes: typed edge formats | Missing (bf16 only) |
| Layout pushed into constant weights | `fused = standard[perm]` on `wq_b` rows and `wo_a` columns | Yes: layout assignment plus a bake-time pass | Missing: parameters are bound by reference |
| Weight TMA before the dependency wait (`fused_wo_a.py`, PDL) | One-kernel-deep cross-op prefetch | Megakernel generalizes it to any depth | Missing (ROADMAP step 9) |
| In-cluster split-K over DSMEM (`fused_wo_a.py`) | Owner CTA sums peers' `st.async` partials, then requantizes | Yes | Split-K uses global fp32 partials and is off |
| Data-as-flag sync (`all_reduce_mhc.py`) | -0.0 sentinels, 3 generations, no separate flag | Yes, for producer/consumer pairs | Missing |
| Warp roles, TMEM lane packing, mixed fp8/NVFP4 dequant | — | No: stays hand-written | — |

**Permutation exactness:**
- Permuting a producer's output rows (plus bias and per-row scales) is free and bit-exact.
- Permuting a consumer's input columns changes the reduction order; vLLM's tests use a 1e-3 tolerance.
- Folding a norm weight into the next GEMM is not bit-exact and needs opt-in numerics.

**Pitfalls to avoid:**
- vLLM's `param.copy_(fn(param))` is not idempotent, so a reload double-permutes.
- Mirage keeps both weight copies and ties its interleave to a hard-coded Qwen3 task count.

### Prerequisites this surfaced

- **`transform_params` stage.** MegaBake binds parameters by reference to the user's live tensors (`compiler.py:876-901`), so nothing can own a permuted or packed copy. Fused QKV, paired gate/up, the RoPE epilogue and norm folding all need this first. It should own derived buffers, detect changes to the original weight, and add a transform revision to the cache key.
- **Typed edge formats.** `EdgeFormat` = dtype + scale spec (dtype, group, static/dynamic) + data layout + scale layout. Bodies declare ranked input/output alternatives, and the planner assigns them with a bytes-plus-phases cost. This fixes vLLM's own gap: its quant key carries no scale layout, so producers inspect the consumer's class.
- **Hook slots on the GEMM.** Today bias with alpha=beta=1 is the only epilogue. Operands must be K-major and the output row-major. Every provider accepts only single-program regions. `bodies/interface.py:84-178` already has most of the contract schema.
- **Real traffic.**
  - `kv_attention` takes one `int32[1]` valid length per layer for the whole batch (`lowering.py:1861`), so every sequence must have the same length. Per-sequence lengths are small effort and the largest throughput gain: about 12–15x tokens/s at batch 16 (unmeasured).
  - A paged KV contract.
  - A ragged mixed prefill/decode step, so weights are read once per step.
  - Online-softmax attention with bounded scratch.
  - A device-side token count instead of buckets.
- **Speculative decoding.** Verification is decode with k+1 rows, which fits the ≤64-row swapAB path. `dspark.py` drafts a whole block in one pass. A draft-and-verify loop in one launch is megakernel-only.

### Proposed architecture

1. A small library of model-agnostic mega bodies with typed prologue/epilogue hook slots: GEMM, attention, split-KV decode attention, row reduction, scan, grouped GEMM.
2. A contract matcher that absorbs neighbouring ops into those hooks as expressions. RoPE style, norm variant and eps, activation, bias and casts come from the graph, not the body.
3. A layout-assignment pass over edge formats.
4. A bake-time weight transform pass with a commutation proof and a cache revision.
5. The event/prefetch megakernel scheduler.

**Rules:**
- Hand-written code holds only inner pipelines plus hook slots.
- A new body family enters only if two or more model families use it, or it has no generic lowering.
- CI includes held-out models never used for tuning.
- Agent-written kernels (AutoMegaKernel, KernelBench-style) are for drafting bodies offline behind verification, never in the compile path. KernelBench-Verified's best aggregate speedup fell from 1.43x to 0.88x once hard-coded outputs were removed.

### Added work items (merge into §5)

| Action | Est. gain (unmeasured, overlapping) | Effort |
|---|---|---|
| `transform_params` stage (owned derived weights, cache revision) | Enabler | M |
| GEMM hook slots: RMSNorm prologue, residual and SiLU·mul epilogues, paired gate/up, fused QKV | 1.7B decode −0.35–0.5 ms (→ about 2.5 ms) | M |
| Per-sequence KV lengths, then paged KV and a ragged mixed step | ~12–15x tokens/s at batch 16 | S → L |
| Typed edge formats; fp8 weights as opt-in or for FP8 checkpoints | 1.7B about −0.9 ms, but different numerics: not a like-for-like win | M–L |
| In-cluster DSMEM split-K with a searched split factor | May reverse "no gain on 60 SMs" for narrow projections | M |
| Port `primitives.py` helpers; data-sentinel waits for one producer/consumer pair | Enabler for events | S |
| Speculative verify as a k+1-row bucket, then an in-launch draft/verify loop | 2–4x tokens/s at acceptance 3–4 | L |
| Generalize RoPE matching (GPT-J interleaved, partial rotary) | Coverage | S |

## Open questions (experiments in the JSON files)

- How much of each barrier interval is straggler wait vs work? Needs per-CTA stamps; this decides item 1's real value.
- What L2 share does the MIG slice actually give for prefetch? Measure.
- Does the RMSNorm GEMM prologue stay bit-exact against HF's bf16 rounding?
- Does split-K help on a full 132-SM H100/H200? It was off only because of the 60-SM result.
- Does the plan cache key include matcher and body source hashes? (vLLM agent flagged this; unverified.)
