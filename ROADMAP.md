# MegaBake roadmap

**Updated 2026-10-19. This file says what to do next. The analysis behind it is the [2026-10-18 megakernel gap report](docs/research/2026-10-18-megakernel-gap/report.md), which this file links to instead of repeating.**

Where we are, in three lines:

- On Llama/Qwen2-shaped models, decode kernel time equals Inductor's. On Gemma 3 and Qwen3.5 it is 2–2.8x slower.
- The decode end-to-end win (1.2–5x) is only against Inductor *without* CUDA graphs. Prefill loses to Inductor + CUDA graphs on every model.
- The tile-level events that would make the megakernel win are built, but full models do not use them yet.

## Now: step 0, honest baselines

Every later decision depends on knowing the real gap. Do these first.

- [ ] **CUDA-graph decode baseline.** `benchmarks/full_model.py` already has a decode variant named `inductor_cudagraphs`, but it is not graphed: `mode="reduce-overhead"` skips graphs because the KV caches are mutated *inputs* of `StatefulSmolLM`/`DecodeStep`. Add a variant that runs the stock HF model with a `StaticCache` (cache tensors owned by the module, not passed as inputs) under `torch.compile(mode="reduce-overhead")`. This is the setup HF uses for graphed generation. Check from the trace that the replay is one graph.
  - If the HF path does not graph, capture `torch.cuda.CUDAGraph` by hand around the eager decode step. The `kv_attention` reference reads `valid_length.item()` on the host (`lowering.py`, `kv_attention`), which breaks capture, so the baseline must not use the `megabake` attention implementation.
- [ ] **Inductor `max-autotune` variant** for prefill and decode. On the 60-SM slice, Inductor disables GEMM autotuning (`is_big_gpu` needs 68 SMs). Record whether it ran.
- [ ] **Bandwidth utilization column:** weight bytes ÷ kernel time ÷ measured copy bandwidth, for MegaBake and the best baseline.
- [ ] **Per-CTA stamps.** `MEGABAKE_PHASE_TIMER=1` stamps only when block 0 passes a barrier (`codegen.py`, `_phase_timer_stamp`). Add a stamp per CTA when it arrives at and leaves each barrier, so `tools/phase_profile.py` reports per-interval work time against straggler wait. This sets the real value of step 1.
- [ ] Re-run all 8 models and replace the measured-state table with the result.

**Done when** every row of the measured-state table has a CUDA-graph decode number and every barrier interval has a work/wait split.

## Next: step 1, events in the table launch

Full models use the table launch, and it ends every interval with `counting_grid_barrier` (`codegen.py`, `_emit_table_module`). `make_region_event_schedule` (`schedule.py`) and `publish_event`/`wait_for_event` (`runtime.py`) already exist for the small parameter launch.

- [ ] Add per-interval completion counters to the device table, next to `phase_table`'s columns (variant, barrier ordinal, rotation, KV-length slot). A CTA publishes on its interval's counter when it finishes, instead of joining the grid barrier.
- [ ] Before an interval, a CTA waits only on the counters of the intervals it depends on (`derive_dependencies` gives region-level producers). Counters are monotonic across calls: target = (call ordinal + 1) × producer count, as in `wait_for_event`.
- [ ] The KV valid-length update currently runs on block 0 after a barrier, once all readers have finished. Give it an explicit event: the length is written only after every reader of that interval has published.
- [ ] Workspace reuse in `memory.py` is proved by phase lifetimes. With events, a slice can be reused only after all of its readers have published. Add that release dependency, or keep a barrier where reuse needs one.
- [ ] Replace `atomic_cas` polling with acquire loads plus `nanosleep` backoff.
- [ ] Add a static validator: a deadlock check (wait graph acyclic under static assignment) and a race check (each producer write → consumer read pair covered by an event).
- [ ] Keep the barrier path behind a flag as the correctness reference. Check bit-exact outputs on SmolLM-135M and Qwen2-0.5B decode with both paths.

**Done when** there are no grid barriers between independent intervals and 135M decode improves beyond noise. The estimate is −17–30% on 135M and −5–10% on 1.7B, unmeasured.

## Then: the ordered list

Effort: S small, M medium, L large. Gains are analytical estimates from the report and overlap. Step 3 comes early so that steps 4–8 stay generic.

| # | Step | Est. gain | Effort | Done when | Why (report) |
|---:|---|---|---|---|---|
| 0 | Honest baselines | — | S | See above | §1, §5 |
| 1 | Events in the table launch | 135M −17–30%, 1.7B −5–10% | M | See above | §2, §3 |
| 2 | `TargetFacts` + per-GEMM configs + cost model. Query L2, bandwidth, capabilities and MIG; calibrate barrier and event latency once per device. Each GEMM chooses its own tile, stages, swapAB and split-K; measure the top 3–5 and cache by device fingerprint. Re-enable search for table launches (`compile_plan` forces `max_plans=1`) | 1.7B −0.2–0.45 ms; removes the >64-row prefill cliff | M–L | No GEMM constant fitted to a model or the 60-SM slice remains | §4 |
| 3 | CI generality guard: a tiny-model zoo (LayerNorm, GELU, MQA, head_dim 256, sliding window, MoE, gated delta) against synthetic targets (60/108/132/148 SMs; sm80/90/100/120). Fail on any silent generic-body fallback. Keep held-out models never used for tuning | Prevents regressions | M | Runs in CI | §4, §6 |
| 4 | `transform_params`: MegaBake-owned derived weights with change detection and a cache revision | Enabler for 5 | M | Fused QKV survives a weight reload | §6 |
| 5 | GEMM hook slots: residual and SiLU·mul epilogues, paired gate/up, QKV merge, RMSNorm row-scale prologue (check bf16 exactness) | 1.7B −0.35–0.5 ms | M | Expressed as contracts in `bodies/interface.py` | §2, §6 |
| 6 | Cross-phase weight prefetch: L2 prefetch by waiting/idle CTAs, then shared-memory handoff with a persistent pipeline | 1.7B −8–15% | M | 1.7B decode GEMMs ≥ 1.9 TB/s | §3 |
| 7 | Semantic matching by scalar-expression normal form (norms, RoPE incl. GPT-J and partial rotary, activations). Retire `StatefulSmolLM` | Closes the Gemma 3 / Qwen3.5 gap | M | Gemma-3-270M and Qwen3.5 kernel ratio ≤ 1.2 | §4 |
| 8 | Derive fold limits, attention chunking and tolerances from cost, dtype and depth | Small on SmolLM, large on wide models | S | No fitted constant left in `schedule.py`/`attention.py` | §4 |
| 9 | Per-sequence KV lengths → paged KV → ragged mixed prefill/decode step | ~12–15x tokens/s at batch 16 | S → L | Batch 16 with different lengths | §6 |
| 10 | Prefill: tensor-core attention, >64-row GEMM path; optional Inductor fallback above a measured crossover | 3–21x at 128–300 tokens | L | Prefill ≤ Inductor + graphs at 128 tokens | §1 |
| 11 | Declared gated delta rule (`cumsum`, triangular solve) | Coverage | M | Qwen3.5 prefill compiles | — |
| 12 | Multi-step decode loop with on-device argmax; then speculative verify | −10–25% per token; 2–4x with speculation | L | — | §3, §6 |
| 13 | Typed edge formats; opt-in fp8 weights | 1.7B about −0.9 ms, different numerics | M–L | — | §6 |
| 14 | In-cluster DSMEM split-K with a searched split factor | Narrow projections | M | — | §6 |
| 15 | Real SM100 provider (tcgen05, TMEM, CLC) and SM80 provider | Portability | L | — | §4 |

**Do not build:** a dynamic work queue for batch-1 decode (measured slower than static scheduling elsewhere), or a port of the Mirage superoptimizer (report §2).

## What exists and what is missing

| Mechanism | State | Step |
|---|---|---|
| FX capture, declared attention ops, generic `DecodeStep`, size buckets, `fallback=inductor` | Done | — |
| Primitive families (views, pointwise, scatters, padding, 1-D conv, input writes) | Done; scans, triangular solve and multi-axis reductions missing | 11 |
| swapAB small-M GEMM (16/32/64), bias epilogue | Done, but one configuration per launch | 2 |
| Interval folding (recompute instead of barrier) | Done | 8 (limits) |
| Counting grid barrier (~0.7 µs), constant kernel arguments | Done | — |
| Split-context decode attention | Done | — |
| Tile/phase events | Built, not wired for full models | 1 |
| gcd/lcm event grouping | Missing | after 1 |
| Deterministic split-K | Built, off | 2, 14 |
| Residual / SiLU·mul epilogues, RMSNorm prologue, fused QKV and gate/up | Missing | 4, 5 |
| RoPE, q/k-norm in the attention task | Missing (separate or folded) | 5, 7 |
| Cross-op weight prefetch, persistent GEMM pipeline, idle-SM work | Missing | 6 |
| Device cost model, per-GEMM tuning, plan search for full models | Missing | 2 |
| Paged KV, per-sequence lengths, on-device token count | Missing | 9 |
| Fast prefill attention and >64-row GEMM | Missing | 10 |
| Multi-step loop, on-device argmax, speculative verify | Missing | 12 |
| fp16, fp8, non-SM90 performance providers | Missing | 13, 15 |
| MoE task family | Missing | not scheduled |

## Measured state

H200 MIG `3g.71gb` slice (60 SMs, ~2.2 TB/s), shared with other tenants (10–15% noise). bf16, 40 warm calls, KV capacity 128, 4-token prefill, 2026-10-18. Medians in µs. Reproduce with `benchmarks/full_model.py`; raw reports are in the untracked `artifacts/full_model_2026-10-18/`.

| Model | Decode kernel, MegaBake | Decode kernel, Inductor | Ratio | Decode call, MegaBake | Decode call, Inductor (no graphs) | Prefill call, MegaBake | Prefill call, Inductor + graphs |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM-135M | 1,502 | 1,382 | 1.09 | 2,138 | 11,925 | 2,130 | **1,966** |
| SmolLM-1.7B | 2,974 | 2,425 | 1.23 | 3,311 | 10,323 | 3,339 | **3,060** |
| Qwen2-0.5B | **1,584** | 1,636 | 0.97 | 2,380 | 10,739 | 2,395 | **2,209** |
| DeepSeek-V3 draft 0.6B | **1,591** | 1,631 | 0.98 | 2,423 | 10,613 | 2,457 | **2,342** |
| Beam DFlash2 draft | 2,591 | 2,463 | 1.05 | 3,041 | 4,344 | 3,095 | **2,835** |
| gemma-2b | 3,590 | 3,107 | 1.16 | 4,306 | 8,681 | 4,061 | **3,582** |
| Gemma-3-270M | 2,080 | 1,032 | **2.02** | 2,798 | 9,048 | 2,597 | **1,595** |
| Qwen3.5-0.8B | 5,621 | 2,033 | **2.77** | 6,506 | 7,565 | fails (`cumsum`) | 4,414 |

Bandwidth utilization is about 50% on 1.7B and 8% on 135M.

Decode against context length (2026-10-17, ms per call):

| Cached tokens | 16 | 128 | 512 | 1,024 | 2,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 135M MegaBake / Inductor | 1.43 / 10.98 | 1.58 / 11.13 | 1.70 / 11.18 | 1.90 / 11.09 | 2.31 / 10.47 |
| 1.7B MegaBake / Inductor | 2.66 / 8.61 | 3.06 / 8.25 | 3.55 / 8.42 | 4.24 / 8.83 | 5.51 / 8.53 |

Bucketed prefill against Inductor + CUDA graphs (2026-10-17, µs per call):

| Prompt tokens (bucket) | 5 (16) | 20 (32) | 50 (64) | 100 (128) | 300 (512) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM-135M MegaBake / Inductor | 2,400 / 1,906 | 2,738 / 1,897 | 3,880 / 1,862 | 6,785 / 2,156 | 54,095 / 2,579 |
| Qwen2-0.5B MegaBake / Inductor | 2,623 / 2,063 | 3,172 / 1,972 | 4,211 / 2,080 | 7,747 / 2,220 | 67,897 / 3,186 |
| SmolLM-1.7B MegaBake / Inductor | 3,696 / 2,889 | 4,994 / 2,954 | 7,457 / 3,081 | 16,687 / 3,767 | 157,902 / 6,673 |

Correctness: 135M and Qwen2-0.5B decode are bit-exact with eager over advancing calls; 1.7B GEMMs differ from cuBLAS by accumulation order only (≤ 2.6e-4 per region).

## Open questions

- How much of each barrier interval is straggler wait? (Step 0 answers this.)
- What L2 share does the MIG slice give for prefetch?
- Does the RMSNorm GEMM prologue stay bit-exact against HF's bf16 rounding?
- Does split-K help on a full 132-SM H100/H200?
- Does the plan cache key cover matcher and body source? `make_plan_key` takes `body_revisions` only when the caller passes them.

## Known smaller gaps

- Compile time grows on large buckets of large models (SmolLM-1.7B, 512 bucket: 41 minutes).
- Long chains of elementwise ops exceed the planner's recursion limit.
- `x + mean * c + bias` with a broadcast is not fused into one region.
- `test_phase_runtime_ports_delay_idle_rebinding_and_largest_grid` fails; `test_task24_compiles_exact_rope_with_supported_stride` is flaky.
