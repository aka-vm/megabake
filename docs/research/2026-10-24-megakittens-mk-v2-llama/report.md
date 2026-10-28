# MegaKittens `mk-v2-llama` and MegaBake

**2026-10-24. Source: [HazyResearch/Megakernels `mk-v2-llama`](https://github.com/HazyResearch/Megakernels/tree/mk-v2-llama) at `ebf5bc0` (2026-05-26), read from the local clone in `agent_space/Megakernels`. MegaBake at `8b94b7c` plus the working tree. Prompted by the [MegaKittens Internals](https://maharshi-pandya.github.io/megacute/) walkthrough. That page documents `mk-v2` at `af60a62`, the commit this branch forks from.**

## Summary

MegaKittens and MegaBake have the same top-level design. Both capture a PyTorch function with Dynamo and AOTAutograd, turn the FX graph into a tiled DAG, derive dependencies from tile regions, and generate one persistent kernel per graph. That is the standard way to build a `torch.compile` backend that emits a megakernel, and it is where the resemblance ends. Underneath, the two projects bet on opposite things, and each is strong where the other is weak.

- **MegaKittens' runtime is well ahead.** It has fixed warp roles, a loader that runs ahead into the next instruction, shared-memory pages passed between instructions, per-tile counters derived on the host, Blackwell 2-CTA tensor-core GEMMs, and two instruction slots in flight per SM. These are exactly the pieces MegaBake's measurements say it lacks (ROADMAP step 3 and R0–R3).
- **MegaBake's frontend is well ahead, and so is its honesty about generality.** On `mk-v2-llama`, the Llama "compiled" decode is a hand-written Python function that calls hand-fused, model-namespaced custom ops: `torch.ops.megakittens.rms_qkv_rope_append`, `qkv_rope_append70b`, `up_matmul70b` and so on. Their tile sizes are tuned to Llama shapes, and their page-release tables are hand-written. The compiler schedules those ops; it does not discover the fusion. MegaBake compiles unmodified Hugging Face models (eight families today) and forbids model-keyed constants.

**No code is shared.** No MegaKittens identifier appears in `src/`, `tools/` or `benchmarks/`. MegaKittens' last commit (2026-05-26) predates MegaBake's first (2026-07-21). MegaBake's tile-region and event machinery dates from 2026-10-13, before the 2026-10-22 Hazy comparison. The ideas MegaBake plans to take are credited in [ARCHITECTURE.md §10](../../../ARCHITECTURE.md). The repo is MIT licensed.

## 1. What `mk-v2-llama` is

| | |
| --- | --- |
| Fork point | `af60a62` on `mk-v2` (Stuart Sul, 2026-04-26), the commit the blog documents |
| Last commit | `ebf5bc0` "Llama1b cleanup" (2026-05-26); 559 commits on the branch |
| Authors | Stuart Sul (framework), nashbrown (Llama itypes, scheduler, ablation), Enbao Cao (attention and GEMM tuning) |
| Diff vs `mk-v2` | 87 files, +12,559 / −728 |
| Size | ~21k lines excluding the vendored ThunderKittens; of that, `megakittens/` Python ~8.8k, `csrc/` CUDA ~5.8k, examples ~4.3k, tests ~1.3k |
| Hardware | Blackwell only (`sm_100a`/`sm_103a`), CUDA 13, PyTorch 2.9 |
| License | MIT (HazyResearch, 2025) |

What the branch adds on top of `mk-v2`:

- **`fx_parser.py` (583 lines) replaces `tracer.py`.** It reads `tensor_meta` and `val`, handles `auto_functionalized_v2` and maps aten ops to ITypes through `IType.from_torch`.
- **`dag_optimizer.py`** has four passes: dead-node pruning, folding `getitem`/`select`/`slice` into tensor ranges, folding `view`, and folding transposes into GEMM operands.
- **Generic itypes:** `copy`, `reduce`, scalar ops, multiple dtypes, and `pos_id_increment`.
- **Llama-3.2-1B, batch 1** (`itypes/llama1b/`, `csrc/itypes/llama1b/`): `rms_qkv_rope_append`, `attention_partial` (single and multi-partition), `attention_reduction`, `matvec_adds`, `rms_upgate_silu` and `rms_lm_head`. These are ports of the May 2025 VM ops onto the new runtime.
- **Llama-3.3-70B, batch ≥ 512, single GPU** (`itypes/llama70b/`): `rms`, `qkv_rope_append`, `attention_decode`, `o_proj_residual` (plus a half-TMEM variant), `gate_silu`, `up_matmul`, `lm_head` and a shared `matmul_pipeline`.
- **A manual-kernels ablation** (`examples/llama70b/manual_kernels/`, `manual_decode.py`). The same fused ops are built as separate TK kernels and replayed in a CUDA graph, as a baseline for the megakernel.
- **Per-block itype selection** ("one node can produce mixed icodes", `4ab61d1`). A node can emit different itype variants per tile; this was used for o_proj and down at batch 512.
- **A hand-written Llama-1B scheduler** (`examples/llama1b/scheduler.py`, 621 lines). It existed alongside the automatic one ("auto schedule benchmark", `dda1848`) and was deleted in the final cleanup. Only the automatic path remains, but the repo records no comparison between the two.

## 2. How each system works, side by side

### 2.1 Capture and frontend

| | MegaKittens | MegaBake |
| --- | --- | --- |
| Entry | `megakittens.compile(fn)` → `torch._dynamo.optimize(nopython=True, dynamic=False)` ([interface.py](https://github.com/HazyResearch/Megakernels/blob/ebf5bc0/megakittens/interface.py)) | `torch.compile(model, backend=megabake.make_backend(), fullgraph=True)` ([frontend.py](../../../src/megabake/frontend.py)) |
| AOT | `aot_autograd(fw_compiler=..., bw_compiler=...)`, returns `make_boxed_func(Dispatcher)` | `aot_module_simplified` plus pinned pre-grad and post-grad passes; Inductor lowering not invoked |
| Input program | A **hand-written decode function** over stacked weights, calling fused custom ops ([llama1b/compiled_decode.py](https://github.com/HazyResearch/Megakernels/blob/ebf5bc0/examples/llama1b/compiled_decode.py), [llama70b/compiled_decode.py](https://github.com/HazyResearch/Megakernels/blob/ebf5bc0/examples/llama70b/compiled_decode.py)) | **An unmodified HF model**; attention is declared through transformers' `AttentionInterface` (`megabake::attention`, `megabake::kv_attention`) |
| Weight prep | The caller stacks, concatenates and permutes weights (Q/K rows interleaved for RoPE, fused QKV) before the call | None. Weights are bound by reference; derived weights are deferred to a planned `transform_params` stage that proves equivalence |
| PyTorch contract | Flat tensor args; scalars become tensors; one output node; no kwargs | Wrappers for guards, outputs, aliases and mutations are preserved; `fallback=error|inductor`; `megabake.bucketed` for variable lengths |
| Graph breaks / dynamic shapes | Rejected (both are TODOs in the source) | Rejected; size buckets with a `PaddingDependence` check |
| Coverage of plain aten | `mm/bmm/matmul`, ~10 unary, ~7 binary, scalar, copy, naive reduce, `_fused_rms_norm`, cuDNN SDPA. The gpt-fast integration plan ([wip/plan.md](https://github.com/HazyResearch/Megakernels/blob/ebf5bc0/examples/gpt-fast/wip/plan.md)) is still at step 1–4 | Primitive families (contraction, pointwise and cast, reduction, view/copy, scatter/state, pad) cover eight HF decoder families end to end |

**Verdict.** The two frontends are the same kind of code. MegaBake's goes much further: it keeps the PyTorch contract and compiles real models. MegaKittens' frontend works, but on this branch it is not what makes Llama fast.

### 2.2 Graph and tiling

MegaKittens' `Node` carries `in_nodes`, `in_ranges`, `out_ranges` and an `IType`. Each IType declares the following:

- `TensorSpec` granularities and the TMA tile types for each operand;
- `num_instructions` and `block_indices`, one entry per tile, giving 4-D-plus coordinates;
- `access_regions`, the absolute boxes each tile reads and writes;
- an optional `inplace_mapping`;
- a `cpp_template` that becomes the C++ type in the dispatch switch.

MegaBake's equivalents are its `TileAccess`, `RegionDependency` and body `BodyConfig` contracts (`bodies/interface.py`, `schedule.py`). Both systems therefore know, per tile, which bytes it reads and writes. MegaKittens turns that into per-tile barriers. MegaBake turns it into phase levels and, in a path not yet emitted for table launches, into tile events.

### 2.3 Scheduling

MegaKittens' [scheduler.py](https://github.com/HazyResearch/Megakernels/blob/ebf5bc0/megakittens/scheduler.py) runs four passes and has no cost model:

1. **Count and place.** Nodes are taken in topological order. Each node's instructions are appended to one flat list, padded to the cluster size with noops so that a CTA pair always runs the same itype.
2. **Assign tensors.** Intermediate buffers are pooled by `(bytes, device)`. A buffer is reused only when the new producer starts at least `get_sm_count()` instructions after the last consumer of the old one, and a WAR "reuse barrier" is added on top of that.
3. **Derive barriers.** For each edge, it takes the GCD of all producer and consumer box sizes per dimension to get a unit region. It maps unit keys to producer tiles, so each consumer gets a set of producer tiles. Each unique (producer node, producer tile set) becomes one global counter whose target is the size of the set. Views are handled by linearising boxes, then taking the GCD over linear ranges. RAW and WAR barriers are kept in separate lists.
4. **Emit.** Each unique `(itype, src_tensors, dst_tensors)` gets its own icode, and instruction `i` runs on SM `i % sm_count` ([dispatcher.py `_pack_instructions_per_sm`](https://github.com/HazyResearch/Megakernels/blob/ebf5bc0/megakittens/dispatcher.py)).

MegaBake's schedule is built from dependency levels. Persistent CTAs process tile ranges, and a counting grid barrier (~0.7 µs at 60 CTAs) separates phases. Small decode producers fold into their consumer's interval. The bounded plan search exists but is forced to one plan for table launches.

**Strengths of the MegaKittens scheduler.** Barrier derivation is automatic, exact and fine-grained, and it handles views. A consumer tile waits only for the producer tiles that overlap it. That is what lets attention for head *h* start while QKV for other heads is still running, and it was hand-coded in the 2025 VM.

**Weaknesses of the MegaKittens scheduler:**

- Placement is round-robin in topological order with no work estimate, so a long instruction delays everything behind it on that SM.
- Each node is padded to the cluster size, which wastes slots for small nodes.
- The reuse rule ("at least SM-count instructions apart") is a heuristic. The barrier is what makes reuse correct, not the distance rule.
- Nothing balances instructions of different lengths. The global work queue (CLC) is the escape hatch, and it is off by default.

### 2.4 Device runtime

| | MegaKittens | MegaBake |
| --- | --- | --- |
| CTA layout | 384 threads, 12 warps: 8 consumer warps at 224 registers; controller, loader, launcher and storer at 56 ([schema.cuh](https://github.com/HazyResearch/Megakernels/blob/ebf5bc0/csrc/schema.cuh), [megakittens.cuh](https://github.com/HazyResearch/Megakernels/blob/ebf5bc0/csrc/megakittens.cuh)) | Every warp runs every body; bodies synchronize with `sync_threads`. Fixed roles are the plan (ARCHITECTURE §9.5) |
| Instruction pipelining | 2 instruction slots. The controller fetches instruction N+1, computes its page order from N's `lid_release_order` and initialises its semaphores while N runs ([controller.cuh](https://github.com/HazyResearch/Megakernels/blob/ebf5bc0/csrc/controller.cuh)) | None across phases. Weight streaming preloads the first K tiles at the barrier and gained −0.7% to +1.5% (ROADMAP step 3) |
| Shared memory | 7 × 32 KB pages, plus per-slot scratch and 32 dynamic semaphores. Pages are released one by one and handed to the next instruction in a declared order | One union layout; all GEMMs share one shared-memory layout in table launch |
| Tensor memory | Allocated once; `tensor_wait`/`tensor_finish` hand it between instructions | N/A (Hopper) |
| GEMM | 2-CTA `tcgen05` MMA, 4-stage K ring, TMEM accumulators, epilogue chunks stored by TMA | Hopper TMA/WGMMA 64×64×64, 8 stages, swapAB at ≤ 64 rows, paired gate/up GEMM, generated epilogues |
| Cross-SM sync | Global `int32` counters: `red.relaxed.gpu.add` to arrive; spin on `ld.relaxed.gpu` until `== target`, then `fence.acquire.gpu` ([utils.cuh](https://github.com/HazyResearch/Megakernels/blob/ebf5bc0/csrc/utils.cuh)) | Counting grid barrier with release/acquire; events exist but are not emitted for full models |
| Launch | `cuLaunchKernelEx`, grid = SM count, cluster 1 or 2, PDL on. **Not a cooperative launch** | Cooperative launch, required for phase plans (`runtime.py:329`) |
| Per call | `barrier_tensor.zero_()` (one memset kernel), then a launch from a Python dispatcher with descriptor caches | Constant kernel arguments; ~600 µs of host work per call is a known gap |
| JIT | NVRTC from a generated C++ source; cubin cache in memory and on disk | CuTe DSL codegen; plan cache keyed on graph, plan, toolchain and body revisions |

### 2.5 The Llama ops are where the speed is

In the Llama paths, each op is a hand-fused pipeline written for one model:

- **`UpMatmul<…>`** ([up_matmul.cuh](https://github.com/HazyResearch/Megakernels/blob/ebf5bc0/csrc/itypes/llama70b/up_matmul.cuh)) fuses `(x @ Wup^T) * gate`.
  - Its tile constants are fixed: `Mb = 256`, `Kb = 64` and `Nb ∈ {224, 256}`. `TUNED_NB = 224` is chosen only when `m == 512 and n == 28672` ([up_matmul.py](https://github.com/HazyResearch/Megakernels/blob/ebf5bc0/megakittens/itypes/llama70b/up_matmul.py)).
  - Its page layout is hard-coded: `A_LIDS = {0,2,3,5}`, `B_LIDS = {1,4}`, and gate tiles reuse LIDs 1–4.
  - It asserts `static_assert(Config::NUM_PAGES == 7 && LOAD_PIPE_DEPTH == 4)` with a literal release order `{6,5,1,2,3,4,0}`.
  - It re-implements most of `matmul_pipeline.cuh` instead of using it.
- **`matmul_pipeline`** ([matmul_pipeline.cuh](https://github.com/HazyResearch/Megakernels/blob/ebf5bc0/csrc/itypes/llama70b/matmul_pipeline.cuh)) picks one of three literal release orders based on `num_iters % 4`.
- **The Llama-1B matvec pipeline** ([matvec_pipeline.cuh](https://github.com/HazyResearch/Megakernels/blob/ebf5bc0/csrc/itypes/llama1b/matvec_pipeline.cuh)) uses 16-row blocks, 3 input and 3 output stages, and five literal release orders.
- **The 70B benchmark** requires `batch_size % 512 == 0` ("Current 70B matmul itypes require batch_size divisible by 512").

This is the "per-op hand tuning" that the [2026-10-22 report](../2026-10-22-hazy-megakernels/report.md) already advised against copying. It is also why the branch performs well. The commit log records the tuning one step at a time:

| Commit | Change | Claimed effect |
| --- | --- | --- |
| `9282542` | First 1B benchmark | 1,638 tok/s |
| `7f88739` | SM-contiguous instruction layout "for cache locality" | — |
| `5e4d909` | Remove two `.item()` calls | 1,640 tok/s |
| `0da9e35` | Clean up attention partial storer | 1,696 tok/s |
| `4d26dc5` | `cp.async` in consumer instead of TMA | 18 → 15 µs |
| `e4f34c7`, `75f4b73`, `5b3e3cc` | 16-byte smem loads, single TMA store, one less multiply | +15, +40, +16 tps |
| `ca1c224`, `8bea30c`, `68bb3fc` | Fetch weights earlier, fill o_proj wave at b=1024, more gate/up instructions at b=512 | — |

1,696 tok/s is about 590 µs per token for Llama-3.2-1B at batch 1 on a B200. This number comes from commit messages. The repo has no results file, no configuration record and no baseline numbers for the manual-kernels ablation, and the GPU is inferred from the README's Blackwell requirement. It cannot be compared directly with MegaBake's 1,784 µs on an H100. It is in line with Hazy's published B200 figure for the 2025 VM (< 680 µs).

## 3. Similarities

1. **Pipeline shape.** Dynamo → AOTAutograd → FX → tiled DAG → schedule → generated per-graph source → one persistent kernel, one CTA per SM.
2. **Static specialisation.** `fullgraph`/`nopython`, `dynamic=False`, recompile per shape, and tensor indices or shapes baked into generated code (MegaKittens through template parameters, MegaBake through CuTe constants and tables).
3. **Declared ops for what decomposition loses.** Both use `torch.library` custom ops with fake implementations: MegaKittens for every fused Llama op, MegaBake for attention and KV attention.
4. **Region-based dependencies.** Both compute per-tile read and write boxes and derive dependencies from overlap.
5. **Storage reuse with lifetime proofs.** MegaKittens uses a size-keyed pool plus WAR barriers. MegaBake reuses only after all consumers, including async readers, finish.
6. **Positions are runtime data.** MegaKittens reads `pos_id` in the kernel and increments it with an instruction. MegaBake's `DecodeStep` carries the KV state explicitly. Neither recompiles per position.
7. **Same stated direction.** Hazy's README says "fuses PyTorch operator graphs into a single persistent CUDA kernel", and MegaBake's north star says "turns a captured PyTorch model into a planned, composable CUDA megakernel". These describe the same goal, so outside readers will see the two projects as alike.

## 4. Differences that matter

| Axis | MegaKittens `mk-v2-llama` | MegaBake |
| --- | --- | --- |
| Who decides fusion | The model author writes fused ops; the compiler schedules them | The compiler matches and lowers regions; named matchers are optimisations over a generic path |
| Model-keyed constants | Pervasive in the Llama itypes (namespaces, `Nb`, batch multiples, page tables) | Forbidden by principle 3; overfitting audited in the 2026-10-18 report |
| Overlap between ops | Yes: loader runs ahead, page handoff, 2-slot instruction pipeline, per-tile counters | No: grid barrier per phase; streaming built and off |
| Load balance | Static round-robin, optional CLC queue | Static tile ranges with rotations; no idle-SM work yet |
| Hardware | Blackwell only, tcgen05/TMEM/clusters/CLC | Hopper only for performance; the Blackwell body is a correctness baseline |
| Wait semantics | Exact `==` target; spins without sleep (`nanosleep<Config>()` is defined and never called) | Counting barrier; planned events use `>=` against monotonic epochs |
| Co-residency | Assumed (grid = SM count, ordinary launch) | Guaranteed by cooperative launch |
| Per-call host | Memset plus Python dispatcher; the compiled examples are not CUDA-graphed (the manual ablation is) | ~600 µs per call (known gap, ROADMAP) |
| Verification | Per-itype tests against the torch reference op (`test_itypes`, `test_itype_validity`), a three-layer MLP test, in-place tests. The `verify=` and `profile=` flags are accepted and ignored | 208 tests; teacher-forced verification with a bit-exact rerun using separate workspaces; per-region 1% L2; HF `generate` greedy-text check |
| Baselines | Manual TK kernels in a CUDA graph (good idea; no numbers recorded) | Strongest equivalent `torch.compile` + CUDA graphs, plus external hand-tuned baselines |
| Codebase | ~14k lines of their own code, small and readable; adding an op takes 2 files | ~26k lines of Python; `compiler.py`, `regions.py`, `codegen.py` and `lowering.py` are each 2.6–2.9k lines |
| Activity | Inactive since 2026-05-26; Hazy's August 2026 "Retire the Abstractions" post says they now drop the framework for agent-written kernels | Active |

## 5. Pros and cons

### MegaKittens `mk-v2-llama`

**Pros**

- **The runtime does what megakernels exist for.** A loader that never joins consumer syncs, pages released one at a time in a known order, and a controller that prepares instruction N+1 during N together give cross-op overlap. MegaBake measured that it cannot get this from streaming alone.
- **Barrier derivation is correct by construction.** The GCD unit-region method, view linearisation and RAW/WAR split remove the hand-written barrier counts of the 2025 VM.
- **The instruction ABI is typed.** It is a 256-byte `static_assert`ed struct mirrored by a pydantic model, explicit tensor, index and barrier lists, and a C++20 concept that enforces the five role structs.
- **It uses Blackwell fully.** Its building blocks are 2-CTA MMA, TMEM accumulators, cluster TMA multicast, PDL and an optional CLC work queue.
- **The extension surface is small.** One `.cuh` and one `IType` subclass, auto-discovered; the CLAUDE.md rules are short and specific.
- **It has an ablation baseline.** The same fused ops as separate kernels in a CUDA graph isolate what the megakernel itself adds.
- **It performs well.** About 590 µs per token for Llama-1B at batch 1 according to commit messages, and an 80-layer 70B at batch 512 runs on one GPU.

**Cons**

- **Generality is in the framework, not in the fast path.** Llama speed comes from model-specific fused ops and hand-chosen tile, page and pipeline constants. A Qwen or Gemma decode would need a new `itypes/<model>/` directory.
- **Its page tables are fragile.** `lid_release_order` is still written by hand and `static_assert`-pinned to `NUM_PAGES == 7`, so changing `PAGE_SIZE` (documented as "the only knob") breaks every Llama itype at compile time.
- **The scheduler is naive.** Placement ignores cost, each node is padded to the cluster size, and the reuse distance is a heuristic.
- **Synchronisation hazards:**
  - Exact-equality waits deadlock on any over-arrival.
  - Spinning without backoff wastes issue slots on the SM.
  - Without a cooperative launch, the persistent grid must happen to be co-resident. A CTA that waits for one that has not been scheduled, for example because another stream's kernel holds SMs, would hang. This is inferred from the launch code and not observed.
- **Per-call overhead.** A memset kernel runs before every launch, and the dispatch is in Python. The compiled examples measure eager per-token loops.
- **Unfinished edges.** `verify` and `profile` are ignored. The generic `RMSNorm` hard-codes `EPS = 1e-6f` and ignores the traced eps. The README references `itypes/add.cuh`, which does not exist. Transposes that feed anything but a GEMM raise an error. Only one output node is allowed.
- **Little evidence.** There is no results directory, and the numbers live in commit messages. The 1B hand scheduler was deleted without a recorded comparison against the automatic one.
- **Abandoned direction.** The authors have moved away from framework megakernels.

### MegaBake

**Pros**

- **It compiles real models without edits.** Eight HF decoder families run in prefill and stateful decode, with the embedding, all layers, the final norm, the LM head and the KV updates in one kernel.
- **It keeps the PyTorch contract.** Guards, aliases, mutations, output structure, an explicit fallback policy and padding-dependence checks are all preserved.
- **Generality is a rule, not an aspiration.** No model-keyed constants, providers check capabilities, and named matchers sit on top of generic lowering.
- **Verification is stronger.** It catches reuse races with a bit-exact rerun, catches subtle numerical bugs with teacher forcing, and compares against HF greedy text, which already caught a swapped `rotate_half`.
- **Evidence discipline.** Every claim names its workload, GPU and baseline, and the baselines include CUDA graphs and external hand-tuned kernels.
- **Safer synchronisation.** Cooperative launch, documented release and acquire rules, warp-uniform traps, and a `wgmma_serialization` gate.

**Cons**

- **It is slower where it counts.** Llama-1B decode takes 1,784 µs per token on an H100, against 966 µs for Hazy's megakernel and 980 µs for CUDA graphs + PDL. Prefill loses to Inductor on every model.
- **It has no cross-op overlap.** Grid barriers bound every phase, table launches are phase-only, events are not emitted, and weight streaming gave no gain without a decoupled producer.
- **Its structural limits are known.** All GEMMs share one shared-memory layout, fixed 64-row tiles leave q/k/v/o/down on ≤ 48 of 132 H100 SMs, and short decode contexts are split across CTAs.
- **Host overhead** is ~600 µs per call.
- **It is Hopper only and bf16 only.** Blackwell is a correctness path.
- **The code is large.** Several 2–3k-line modules, and the plan search is disabled for full models.

## 6. What this changes for MegaBake

The [2026-10-22 report](../2026-10-22-hazy-megakernels/report.md) already lists what to take: roles and pages first, then counters, then a typed phase record. This branch adds five refinements:

1. **The per-block variant is a real need.** `4ab61d1` lets one node emit different itypes per tile, so that o_proj and down could fill a wave at batch 512. In MegaBake terms, a region's tile map should allow a different `BodyConfig` per tile range, chosen by the SM-coverage rule in ARCHITECTURE §9.
2. **Derive the page release order.** Every Llama itype hand-writes `lid_release_order` and pins `NUM_PAGES`. When MegaBake builds the `ResourcePlan` page pool (§9.6), the release order should come from each body's declared stage usage, so the page size can change without editing bodies. This is the clearest place where MegaBake can beat the design it borrows from.
3. **Keep cooperative launch and `>=` waits** when counters arrive. MegaKittens shows the failure modes of the alternative.
4. **Adopt the ablation baseline.** For each fused body family, the same bodies launched separately in a CUDA graph isolate the megakernel's own contribution. Record the numbers, which the branch does not.
5. **Batch-512 decode is a different regime.** The 70B path uses tensor-core GEMMs with `M = 512` tiles and a 2-CTA MMA. MegaBake's swapAB at ≤ 64 rows plus its WGMMA tiles cover the small-batch case only. If throughput decode becomes a target, the shape-class rule in §9 needs a third class.

## 7. On the resemblance

MegaBake's design is not derived from MegaKittens. The two arrived at the same frontend because Dynamo plus AOTAutograd is the only supported way to build a `torch.compile` backend. They describe their goal in similar words because they have the same goal. The deeper parts differ:

- MegaBake's tile-region dependencies, phase scheduler and event schedule were committed on 2026-10-13, nine days before the Hazy comparison.
- Its frontend contract, generic lowering and verification have no MegaKittens counterpart.
- MegaKittens' runtime (roles, pages, controller, counters) is the part MegaBake does not have yet.

The resemblance will grow as MegaBake builds R0–R3, because those items borrow MegaKittens' runtime ideas on purpose. To keep this clean:

- keep the attribution in ARCHITECTURE §10 and in each commit that lands one of those ideas;
- write the code from MegaBake's own contracts rather than porting `.cuh` files;
- if any MegaKittens or ThunderKittens source is ever vendored (as CUTLASS is in `bodies/cutlass_upstream/`), keep its MIT notice next to it.
