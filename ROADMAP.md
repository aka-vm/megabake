# MegaBake roadmap

**Updated 2026-10-21; steps 1-2b measured in `artifacts/step2b_2026-10-21` (steps 1-2 in `artifacts/step12b_2026-10-21`), step 3 in `artifacts/step3_2026-10-21`. This file says what to do next. The analysis behind it is the [2026-10-18 megakernel gap report](docs/research/2026-10-18-megakernel-gap/report.md); the 2026-10-20/21 measurements below supersede the report's estimates where they disagree.**

Where we are, in three lines:

- Against a real CUDA-graphed Inductor decode, MegaBake wins on 6 of 8 models when calls are pipelined (5–55%), and roughly ties when each call synchronizes. The old "1.2–5x" wins came from host syncs in the ungraphed baseline.
- Decode is bound by fixed per-phase latency, not by grid barriers: barrier release is ~11% of the 135M kernel. Gemma 3 (1.3x) and Qwen3.5 (1.9x) decode still lose to graphed Inductor. Prefill calls lose on every model except DeepSeek draft (a tie).
- Cross-GEMM fusion is built (step 1): residual adds and SiLU·mul run in GEMM epilogues, gate/up as one paired GEMM, RoPE inside decode attention.
- Weight streaming into shared memory across phases is built and off (step 3): the preloads themselves change decode by −0.7% to +1.5%. Weights still stream only inside GEMM phases by default.

## Done: step 0, honest baselines (2026-10-20)

- [x] **Graphed decode baseline.** `benchmarks/full_model.py --variants inductor_graphed` captures one Inductor decode step in a CUDA graph by hand; during capture `kv_attention` runs a static-capacity version that is bit-identical to the reference.
- [x] **Per-CTA stamps.** `tools/phase_profile.py --cta` splits every barrier interval into the slowest CTA's work, straggler wait and barrier release.
- [x] **Per-phase fixed latency cut.** CTAs skip phases they have no tiles in, slot addresses resolve in one load, swapAB GEMMs store from registers, and single-chunk decode attention skips its statistics exchange: decode kernel −1–15% across models.
- [x] Re-ran all 8 models (`artifacts/full_model_2026-10-21`).
- [ ] Inductor `max-autotune` variant (on the 60-SM slice Inductor disables GEMM autotuning; `is_big_gpu` needs 68 SMs).
- [ ] Graphed baseline for `--decode-contexts` and bucketed prefill.

## Done: step 1, cross-GEMM fusion (fusion built; 0.9 ms target moved to step 3, then step 10)

A SmolLM-135M decoder layer ran 15 phases in 6 barrier intervals; it now runs 9 phases in 5 intervals (7 serial steps, q/k/v run in parallel), and the kernel 278 phases instead of 603:

```text
before: [rope q, rope k] | attention | o_proj | [add(R), rmsnorm(R), gate, up] | [silu·mul(R), down] | [add(R), rmsnorm(R), q, k, v]
now:    attention(+rope) | o_proj(+residual) | [rmsnorm(R), gate·up(+silu·mul)] | down(+residual) | [rmsnorm(R), q, k, v]
```

- [x] **GEMM epilogue expressions.** A GEMM fuses with the elementwise chain that consumes its result (`regions._gemm_epilogue_candidates`); the chain becomes the contraction's `post`, generated into a function the swapAB direct-store epilogue evaluates per element (`bodies/gemm.py` `epilogue_source`). GEMMs of one shape share one kernel body and pick their epilogue at run time (`gemm_table` epilogue-kind column), with the epilogue's tensors built after the mainloop.
- [x] **Paired GEMM.** Two projections of one activation share a 128-row tile: TMA loads 64 rows of each weight into the halves of the A stage from two descriptors, so the chain (SiLU·mul, GELU·mul) reads both accumulators of a row in one thread. No derived weight copies.
- [x] **RoPE in decode attention.** The split-context decode body evaluates the query's and new key's elementwise producer for its row into shared scratch and appends that key to the cache (`regions._attention_prologue_candidates`).
- [x] **Common-subexpression elimination** on the captured graph (`frontend.eliminate_common_subexpressions`): per-layer recomputations (the fixture's RoPE tables and causal mask) run once.
- [x] Bit-exact: the 2-layer table-launch tests compare logits and KV state with eager at zero tolerance.
- [ ] **135M decode kernel ≤ 0.9 ms: not met** (1,282 → 1,059–1,137 µs across runs). What remains is latency in four small GEMM intervals per layer (~3–4 µs each, mostly pipeline fill: an o_proj tile moves ~90 KB) and attention (~6 µs on 9 CTAs). Split-K was re-measured and is still slower (1,141 vs 1,075 µs). The target moved to step 3, which attacked the GEMM fill latency and did not meet it either (below).

## Done: step 2 and 2b, prefill regressions and stable GEMM code generation

Both regressions came from one compiler behaviour: ptxas warning **C7519**, "wgmma.mma_async instructions are serialized due to program dependence on compiler-inserted WG.AR in divergent path". When it fires, every tensor-core instruction of the megakernel waits for the previous one (SmolLM-1.7B prefill: +43% SASS instructions, mostly R2UR and WARPGROUP fences, +20% time). It fires when any trap in the kernel sits behind a per-lane condition (2b below). Check a kernel with `tools/wgmma_serialization.py`.

- [x] The 1.7B regression from the fused o_proj epilogue was not the fusion: one more inlined GEMM variant alone caused it (2,455 → 3,051 µs with the o_proj GEMM unfused but in its own variant). Fixed by sharing one body per GEMM shape (2,460 µs).
- [x] The direct register store's per-element branch triggered C7519 on gemma-2b; plain GEMMs now store with one predicated instruction (gemma-2b prefill 3,755 → 3,164 µs, below 2026-10-18).
- [x] **Root cause (2b): traps behind per-lane conditions.** The gather bodies' index bounds checks ran `llvm.intr_trap()` under each lane's own condition. ptxas then cannot assume that whole warps reach a later wgmma, so it serializes every wgmma of the kernel. It isn't register allocation: what toggled with unrelated changes was only whether ptxas could see the divergence. Every trap now goes through `runtime.trap_if`, which votes across the warp first (same semantics: the kernel traps if any lane is out of range). Found by compiling dumped PTX directly with a CUDA 13.4 ptxas (~1 s per try; `CUTE_DSL_KEEP=ptx`) and narrowing to the embedding gather's bounds check.
- [x] `tools/wgmma_serialization.py` clean on all 15 compilable kernels (8 models decode, 7 prefill; Qwen3.5 prefill needs step 5). Prefill kernel (measured): SmolLM-1.7B 3,043 → 2,455 µs (−19%), Qwen2-0.5B 1,710 → 1,366 (−20%), DeepSeek draft 1,709 → 1,345 (−21%), each now below 2026-10-18. Other models flat.
- [x] **Tried and reverted: one GEMM body per configuration** (problem shape read from the GEMM row at run time instead of one body per shape). It did nothing for C7519 or prefill (1.7B prefill span identical) and made decode 2–4% slower in a same-session A/B (135M 1,031 → 1,066 µs): the in-kernel shape rebuild moves the whole kernel's register allocation (174/244 → 128) plus ~0.1–0.25 µs per GEMM phase. The old kernel capped at 128 registers reproduces the 1.7B loss exactly, and makes 135M *faster* (1,008 µs), so check step 3 gains against a capped baseline too. The measured table below was taken with it in, so its decode numbers are 2–4% pessimistic.

## Done: step 3, weight streaming into shared memory (built, off: no runtime gain)

A GEMM's weights do not depend on the phases before it. At the grid barrier that closes an interval, each CTA arrives, then warp 0 loads the weight (CUTLASS A of a swapAB GEMM) of its first tile in the next interval into the GEMM's pipeline stages, as many leading K tiles as the stages hold. The GEMM phase finds a tag in shared memory, skips its mbarrier init and loads only the activations for those K tiles. ARCHITECTURE §5 has the mechanism. `MEGABAKE_WEIGHT_STREAM_STAGES` caps the streamed stages; it is 0 (off) by default.

- [x] Built for every swapAB table GEMM whose weight no region writes, paired GEMMs included. Bit-exact: with streaming on, the 2-layer table-launch decode matches eager at zero tolerance (`test_table_launch_streams_weights_in_stateful_decode`). GPU tests and `tools/wgmma_serialization.py` (all 15 kernels) are unchanged with it on. Off, the kernel's PTX is identical to the kernel before step 3 (checked on the 2-layer fixture and DeepSeek-draft decode), so the measured table below still holds.
- [x] **Measured: the preloads do not pay.** The A/B that decides is against the same kernel with the streaming code compiled in but a zero stream table, so no CTA preloads (`dead_stream*/`, 2 interleaved rounds, decode kernel):

  | Model | Off | Streaming on | Code in, no preloads | Preloads alone |
  | --- | ---: | ---: | ---: | ---: |
  | SmolLM-135M | 1,163 | 1,078 | 1,084 | −0.5% |
  | SmolLM-1.7B | 2,650 | 2,606 | 2,624 | −0.7% |
  | Qwen2-0.5B | 1,352 | 1,374 | 1,360 | +0.9% |
  | DeepSeek-V3 draft | 1,373 | 1,419 | 1,398 | +1.5% |

  Against "off" alone, streaming looked like −7% on 135M (`ab*/`, `sweep/`). That is code shape: the kernel's registers went 174 → 239, and that alone makes 135M 6.8% faster. Gemma-2b, Gemma-3-270M, Beam and Qwen3.5 are flat with streaming on (`ab_models/`).
- [x] **Why: issuing blocks.** Once the SM's TMA queue is full, each further load blocks warp 0 until one drains. Issuing 13 stages takes a median of ~1,000 and a p90 of ~4,000 cycles (clock64). So a CTA waits for most of its own preload before it leaves the barrier, and the loads overlap only barrier wait it would have spent anyway. The per-CTA profile shows it: slowest-CTA work falls (135M 809 → 667 µs per call), but each preload-issuing interval's span grows ~1.5 µs (135M) to ~2 µs (1.7B), because the CTA that arrives last pays the issue.
- [x] **Tried, no gain:** any stage count (2, 4, 8, all); L1-prefetching the stream-table entry; `prefetch.tensormap` of the target descriptor (the first TMA costs only ~160 cycles); a two-interval lookahead for CTAs without tiles in the next interval; adaptive issue (two stages per barrier poll, so the last CTA issues at most two).
- [ ] **135M ≤ 0.9 ms and 1.7B ≤ 2.2 ms: not met** (now ~1,160 and ~2,650 µs off). At peak bandwidth 1.7B decode needs ~1.55 ms. gate·up already runs at ~90% of peak and the LM head at peak. The loss is in the 32-tile projections (q/k/v/o/down on 32 of 60 SMs, 70–72% of peak), attention (~4.7 µs × 24) and barrier release (~1.3 µs × 121). The targets move to steps 7 and 10.
- [ ] Streaming can pay only with a producer warp that the CTA's other work does not wait for. Today every body synchronizes all 256 threads. That needs generic and attention bodies on 7 warps, with named barriers.

## Then: the ordered list

Effort: S small, M medium, L large. Gains are estimates unless marked measured.

| # | Step | Est. gain | Effort | Done when |
|---:|---|---|---|---|
| 4 | Qwen3.5 and Gemma 3 decode: profile with `phase_profile.py --cta`, remove generic-body fallbacks; semantic matching by scalar-expression normal form (norms, RoPE incl. GPT-J and partial rotary, activations); retire `StatefulSmolLM` | Closes the 1.3x / 2.2x gaps | M | Both within 1.1x of graphed Inductor in decode |
| 5 | Declared gated delta rule (chunked prefill: `cumsum`, triangular solve) | Coverage | M | Qwen3.5 prefill compiles |
| 6 | Host work per call (Dynamo guards, binding: ~600 µs today) | Wins the synced comparison | M | Synced decode call ≤ kernel + 150 µs |
| 7 | `TargetFacts` + per-GEMM configs + cost model: query L2, bandwidth, capabilities and MIG; each GEMM chooses tile, stages, swapAB and split-K; measure the top 3–5 and cache by device fingerprint; re-enable search for table launches (`compile_plan` forces `max_plans=1`). The 32-tile projections of 1.7B run at 70–72% of peak bandwidth on 32 of 60 SMs | 1.7B −0.2–0.45 ms; removes the >64-row prefill cliff | M–L | No GEMM constant fitted to a model or the 60-SM slice remains; 1.7B decode kernel ≤ 2.2 ms |
| 8 | CI generality guard: a tiny-model zoo (LayerNorm, GELU, MQA, head_dim 256, sliding window, MoE, gated delta) against synthetic targets (60/108/132/148 SMs; sm80/90/100/120); fail on any silent generic-body fallback; keep held-out models | Prevents regressions | M | Runs in CI |
| 9 | Split-K for projections with fewer tiles than workers (built, off: no gain on the 60-SM slice) | Narrow projections | S | Re-measured after 1 and 7 |
| 10 | Event counters in the table launch | ≤ barrier release share (~11% measured) | M | Only after 1 and 3, if barrier release is still ≥ 10% of the kernel; 135M decode kernel ≤ 0.9 ms |
| 11 | Derive fold limits, attention chunking and tolerances from cost, dtype and depth | Small on SmolLM, large on wide models | S | No fitted constant left in `schedule.py`/`attention.py` |
| 12 | Per-sequence KV lengths → paged KV → ragged mixed prefill/decode step | ~12–15x tokens/s at batch 16 | S → L | Batch 16 with different lengths |
| 13 | Prefill: tensor-core attention, >64-row GEMM path; optional Inductor fallback above a measured crossover | 3–21x at 128–300 tokens | L | Prefill ≤ Inductor + graphs at 128 tokens |
| 14 | Multi-step decode loop with on-device argmax; then speculative verify | −10–25% per token; 2–4x with speculation | L | — |
| 15 | Typed edge formats; opt-in fp8 weights | 1.7B about −0.9 ms, different numerics | M–L | — |
| 16 | In-cluster DSMEM split-K with a searched split factor | Narrow projections | M | — |
| 17 | Real SM100 provider (tcgen05, TMEM, CLC) and SM80 provider | Portability | L | — |

**Measured and rejected:**

- **Shared-memory weight streaming at the barrier before each GEMM interval** (step 3): −0.7% to +1.5% from the preloads themselves; issuing blocks warp 0 on the TMA queue. Built and off (`MEGABAKE_WEIGHT_STREAM_STAGES=0`).

- **L2 weight prefetch** of the next GEMM interval's weights (`cp.async.bulk.prefetch.L2`): 16–24% slower issued by every CTA, 8–10% slower issued only by idle CTAs (`artifacts/chunk3_2026-10-20`, `chunk4_2026-10-20`). Built and off (`MEGABAKE_L2_PREFETCH_BYTES=0`).
- **Events as the first runtime step.** The report estimated −17–30% on 135M; the per-CTA profile bounds the gain by barrier release, ~11%.

**Do not build:** a dynamic work queue for batch-1 decode (measured slower than static scheduling elsewhere), or a port of the Mirage superoptimizer (report §2).

## What exists and what is missing

| Mechanism | State | Step |
|---|---|---|
| FX capture, declared attention ops, generic `DecodeStep`, size buckets, `fallback=inductor` | Done | — |
| Primitive families (views, pointwise, scatters, padding, 1-D conv, input writes) | Done; scans, triangular solve and multi-axis reductions missing | 5 |
| swapAB small-M GEMM (16/32/64), register direct store, bias epilogue | Done, one configuration per launch | 7 |
| Interval folding (recompute instead of barrier), idle-phase skipping | Done | 11 (limits) |
| Counting grid barrier (~0.7 µs), constant kernel arguments | Done | — |
| Split-context decode attention | Done | — |
| Graphed decode baseline, per-CTA profile | Done | — |
| GEMM epilogue expressions, paired gate/up GEMM, one body per GEMM shape | Done (swapAB launches) | 1 |
| Common-subexpression elimination on the captured graph | Done | 1 |
| RoPE in decode attention | Done (split-context decode body) | 1 |
| RMSNorm as GEMM prologue, q/k-norm in attention, RoPE in prefill attention | Missing | 1, 4 |
| Tile/phase events | Built, not wired for full models | 10 |
| Deterministic split-K | Built, off | 9, 16 |
| L2 weight prefetch | Built, off (measured slower) | — |
| Shared-memory weight streaming across phases | Built, off (preloads measured −0.7% to +1.5%) | — |
| Device cost model, per-GEMM tuning, plan search for full models | Missing | 7 |
| Paged KV, per-sequence lengths, on-device token count | Missing | 12 |
| Fast prefill attention and >64-row GEMM | Missing | 13 |
| Multi-step loop, on-device argmax, speculative verify | Missing | 14 |
| fp16, fp8, non-SM90 performance providers | Missing | 15, 17 |
| MoE task family | Missing | not scheduled |

## Measured state (2026-10-21, steps 1-2b)

H200 MIG `3g.71gb` slice (60 SMs, ~2.2 TB/s), shared with other tenants (10–15% noise between runs; compare within one run). bf16, 40 calls, KV capacity 128 (context < 100 tokens), 4-token prefill. Kernel is profiled GPU time of one call. Decode before is `artifacts/full_model_2026-10-21`; prefill is shown against 2026-10-18 and steps 1-2 (`artifacts/step12b_2026-10-21`); now is `artifacts/step2b_2026-10-21/full_model`. Reproduce with `benchmarks/full_model.py --variants inductor_cudagraphs inductor_graphed megabake`. Medians in µs.

| Model | Decode kernel, before → now | Decode pipelined: MegaBake / graphed Inductor | Decode synced call: MegaBake / graphed Inductor | Prefill kernel, 10-18 / steps 1-2 → now | Prefill call: MegaBake / Inductor + graphs |
| --- | --- | ---: | ---: | --- | ---: |
| SmolLM-135M | 1,282 → **1,186** | **1,208** / 2,019 | **1,818** / 2,297 | 1,381 / 1,129 → **1,145** | 1,942 / **1,823** |
| SmolLM-1.7B | 2,816 → **2,665** | **2,576** / 3,015 | **3,051** / 3,105 | 2,639 / 3,043 → **2,455** | 3,174 / **2,886** |
| Qwen2-0.5B | 1,395 → **1,364** | **1,406** / 2,165 | **2,147** / 2,382 | 1,605 / 1,710 → **1,366** | 2,154 / **2,088** |
| DeepSeek-V3 draft 0.6B | 1,416 → **1,387** | **1,417** / 2,143 | **2,132** / 2,377 | 1,590 / 1,709 → **1,345** | **2,071** / 2,086 |
| Beam DFlash2 draft | 2,553 → **2,530** | **2,535** / 2,664 | 2,959 / **2,727** | 2,626 / 2,540 → **2,579** | 2,975 / **2,779** |
| gemma-2b | 3,464 → **3,318** | **3,329** / 3,614 | 3,948 / **3,692** | 3,341 / 3,164 → **3,152** | 3,804 / **3,518** |
| Gemma-3-270M | 1,872 → **1,793** | 1,818 / **1,416** | 2,481 / **1,598** | — / 1,564 → **1,573** | 2,278 / **1,513** |
| Qwen3.5-0.8B | 5,057 → **4,445** | 4,490 / **2,319** | 5,366 / **2,502** | fails (`cumsum`) | — / 4,114 |

Before 2b, runs toggled C7519 per model (`artifacts/step12_2026-10-21` measured 1.7B prefill 2,491 but gemma-2b prefill 3,758). GPU tests: 40 passed, 1 known failure (below).

Per-CTA profiles (`artifacts/step12b_2026-10-21/phases_*.json`): 135M decode 278 phases, 151 intervals, ~1.0 ms span (was 603, 184, 1.21 ms before step 1); 1.7B decode 224 phases, 121 intervals, 2.31 ms span.

`benchmarks/full_model.py` collects garbage before the pipelined window: one full collection over the compilers' objects takes ~0.6 s and had landed inside the 40-call window (16–19 ms per call on two models).

What the per-CTA profile showed before fusion (SmolLM-135M decode, 603 phases, 184 intervals, `artifacts/chunk4_2026-10-20/phases_135m.json`):

- The critical path is per-phase work: barrier release is ~136 µs (~0.74 µs each, ~11%) of a ~1.2 ms span.
- Per layer: RoPE 2.7 µs, attention 8.5 µs (9 of 60 CTAs busy), o_proj 3.8 µs, add+norm+gate+up 7.4 µs, silu·mul+down 7.0 µs, add+norm+qkv 6.9 µs.
- On 1.7B the 32-tile projections run on 32 of 60 SMs; split-K never applies to them (two splits need 64 CTAs).

Bandwidth utilization in decode: ~55% on 1.7B and ~10% on 135M.

Correctness: 135M and Qwen2-0.5B decode are bit-exact with eager over advancing calls; 1.7B GEMMs differ from cuBLAS by accumulation order only (≤ 2.6e-4 per region).

## Open questions

- Why does 135M decode run 6.8% faster with 239 registers than with 174, and can the compiler get that without unrelated code? (Step 3's inert-code control; 1.7B and the Qwen2 shape do not gain.)

- What L2 share does the MIG slice give, and why did idle-CTA L2 prefetch still slow decode?
- Does the RMSNorm GEMM prologue stay bit-exact against HF's bf16 rounding?
- Does split-K help on a full 132-SM H100/H200?
- Does the plan cache key cover matcher and body source? `make_plan_key` takes `body_revisions` only when the caller passes them.

## Known smaller gaps

- Compile time grows on large buckets of large models (SmolLM-1.7B, 512 bucket: 41 minutes).
- Long chains of elementwise ops exceed the planner's recursion limit.
- `x + mean * c + bias` with a broadcast is not fused into one region.
- `test_phase_runtime_ports_delay_idle_rebinding_and_largest_grid` fails; `test_task24_compiles_exact_rope_with_supported_stride` is flaky.
- A new body must not call `llvm.intr_trap()` directly: use `runtime.trap_if`, or C7519 returns (check with `tools/wgmma_serialization.py`).
- Whole-kernel register allocation moves decode by up to ±7% with unrelated code. A/B a feature against "off" **and** against the same code compiled in but inert (step 3).
- Gemma 3 checkpoints are gated; there is no HF token in this environment.
