# MegaBake architecture as built (branch v4-impl, HEAD fb665b1, 2026-10-22)

All paths are relative to `/workspace/megabake` unless absolute. "Memory" = the user's notes in `/home/dev/.claude/projects/-workspace-megabake/memory/`. Read-only investigation; nothing was run on the GPU.

## Q1. What are the main modules, and what do ARCHITECTURE.md, ROADMAP.md, north-star.md and docs/ say about the design and where it is going?

### Takeaway
MegaBake is a `torch.compile` backend. It takes post-grad FX, turns it into a region graph, picks a CuTe DSL "body" for each region, statically schedules the bodies into barrier-separated phases, and emits **one cooperative persistent CuTe DSL kernel**. For full models that kernel is a **device-side interpreter over a phase table**. The stated direction is: hook-slot "mega bodies", a cost model with per-GEMM configs, event counters, and cross-op weight streaming, which is essentially what Hazy's design does by hand. Measurements so far have pushed counters and streaming down the priority list.

### Cited Findings
- North star: "a captured PyTorch model → planned, composable CUDA megakernel". MegaBake receives post-grad FX at a version-pinned handoff before Inductor lowering. Each region maps to a "device-callable CuTe DSL body" composed "into one persistent CUDA kernel where legal". Separately launched CuTe wrappers do not count as composable — [north-star.md:1-20](north-star.md)
- Pipeline: Dynamo/AOT FX → `ComputeGraph` → Regions → Body selection → `KernelPlan` → CuTe DSL codegen (one persistent kernel) → verify/measure/cache — [ARCHITECTURE.md:16-26](ARCHITECTURE.md)
- Module map:
  - `frontend.py` + `fx_handler/`: capture.
  - `fx_handler/hf_attention.py`: declared ops `megabake::attention` / `megabake::kv_attention`.
  - `graph.py`: import.
  - `lowering.py`: tensor programs.
  - `regions.py`: matchers.
  - `bodies/`: device bodies.
  - `planner.py`, `plan.py`: selection, tile maps, search.
  - `schedule.py`: deps, phases, events.
  - `memory.py`: storage.
  - `codegen.py`: table or parameter launch.
  - `runtime.py`: cooperative launch, barrier, events.
  - `compiler.py`, `cache.py`, `buckets.py`: orchestration.

  — [ARCHITECTURE.md:28-41](ARCHITECTURE.md)
- Source sizes (lines): `compiler.py` 2912, `regions.py` 2788, `codegen.py` 2706, `lowering.py` 2644, `bodies/cutlass_gemm.py` 2134, `bodies/cutlass_upstream/dense_gemm_persistent.py` 1616, `schedule.py` 1520, `bodies/generic.py` 1326, `bodies/attention.py` 1130, `memory.py` 1020, `planner.py` 794, `plan.py` 781, `bodies/gemm.py` 755, `runtime.py` 492; ~26k lines in total — `wc -l src/**/*.py`
- Principles:
  - "A device body keeps its pipeline": the compiler composes tuned GEMM/attention pipelines and never flattens them.
  - Hand-written code holds only inner pipelines plus hook slots.
  - No model-keyed constants.
  - Evidence decides, against the strongest baseline including CUDA graphs.

  — [ARCHITECTURE.md:7-14](ARCHITECTURE.md)
- Target design (§9) has five parts:
  1. Mega bodies with typed prologue/epilogue hook slots (GEMM, attention, split-KV decode, row reduction, scan, grouped GEMM).
  2. A contract matcher using scalar-expression normal form.
  3. Layout assignment over typed `EdgeFormat`s.
  4. A `transform_params` bake-time stage for derived weights.
  5. An event + prefetch scheduler: per-phase/per-tile counters replace grid barriers, idle CTAs prefetch weights into L2, then shared memory with persistent pipeline state, "Static assignment only".

  Underneath sits a device-keyed cost model (`TargetFacts` + barrier/event latency calibration). — [ARCHITECTURE.md:133-152](ARCHITECTURE.md)
- Reference systems the docs say to borrow from:
  - Mirage MPK: event counters, gcd/lcm grouping, swapAB, epilogue tasks, cross-task preload.
  - Inferact TPU: one continuous weight stream (bank ring).
  - vLLM: declared custom ops.
  - Inductor: primitive families.

  ThunderKittens/Hazy is not in this table. — [ARCHITECTURE.md:154-164](ARCHITECTURE.md)
- Status summary (ROADMAP head):
  - Against graphed Inductor, decode wins on 6/8 models when calls are pipelined (5–55%) and roughly ties when each call synchronizes.
  - Decode is bound by fixed per-phase latency; barrier release is ~11% of the 135M kernel.
  - Gemma 3 (1.3x) and Qwen3.5 (1.9x) still lose.
  - Prefill loses on every model except the DeepSeek draft.

  — [ROADMAP.md:5-10](ROADMAP.md)
- Ordered next steps:
  - 4: Qwen3.5/Gemma 3 semantic matching.
  - 5: gated delta rule.
  - 6: host overhead (~600 µs per call).
  - 7: `TargetFacts` + per-GEMM configs + cost model, and re-enable search for table launches.
  - 8: CI zoo.
  - 9: split-K.
  - 10: event counters in the table launch.
  - 11: derive the fold constants instead of fixing them.
  - 12: paged KV / ragged batching.
  - 13: prefill tensor-core attention.
  - 14: multi-step decode with on-device argmax.
  - 15: fp8.
  - 16: DSMEM split-K.
  - 17: SM100/SM80 providers.

  — [ROADMAP.md:67-86](ROADMAP.md)
- "Do not build": a dynamic work queue for batch-1 decode, or a port of the Mirage superoptimizer — [ROADMAP.md:95](ROADMAP.md)
- `docs/history/` holds earlier design docs (ELEVATOR_PITCH, IMPLEMENTATION_PLAN, PERFORMANCE_REPORT, RESEARCH, SCHEDULER_REUSE, TASKS, performance_story). `docs/research/2026-10-18-megakernel-gap/` holds the gap report and its JSON evidence — `ls docs/`

### Inferences
- MegaBake's own framing of its correspondence to Hazy: bodies play the role of Hazy instructions and the plan plays the role of the ahead-of-time schedule ([agent_space/research_notes/megakernels talk.md:169-177](agent_space/research_notes/megakernels%20talk.md)). In practice the runtime is a barrier-phased BSP machine, not a counter-driven per-SM instruction stream (see Q4).

### Gaps
- I did not read `docs/history/*`. Those are superseded design documents.

---

## Q2. What is the IR stack, what does each layer represent, and where are the boundaries awkward?

### Takeaway
There are about eight layers, all frozen Python dataclasses:

FX `GraphModule` → `ComputeGraph` (Value/Op + a per-op `TensorProgram`) → `RegionCandidate`/`Region` (matcher-fused tensor programs) → `BodySelectionPlan` (`SelectedBody` + `BodyConfig` contracts) → `TileMap` (logical tiles / `WorkItem`s) → `DependencyPlan` → `PhaseSchedule` (or `EventSchedule`) → `KernelPlan` (+ `StoragePlan`) → generated CuTe DSL Python source + host-side phase/GEMM/stream tables (`LaunchMetadata`).

The final "instruction stream" is an integer phase table that the kernel interprets. It is not a per-SM instruction list.

### Cited Findings
- **L0, capture.** `make_backend()` ([src/megabake/frontend.py:575](src/megabake/frontend.py)) is the Dynamo backend. `_prepare_fx_graph` ([frontend.py:437](src/megabake/frontend.py)) applies the pinned AOT/post-grad passes, `split_to_slices` ([frontend.py:107](src/megabake/frontend.py)) and `eliminate_common_subexpressions` ([frontend.py:140](src/megabake/frontend.py)). It rejects symbolic shapes ([frontend.py:427](src/megabake/frontend.py)). Attention enters as a declared op through HF's `AttentionInterface` ([ARCHITECTURE.md:31,45-50](ARCHITECTURE.md); declared op stubs at [src/megabake/lowering.py:217-316](src/megabake/lowering.py)).
- **L1, `ComputeGraph`**:
  - `Value` has id, shape, dtype, strides, `StorageRelation` (aliases), producer and users ([src/megabake/graph.py:30-52](src/megabake/graph.py)).
  - `Op` has operands, results, attributes, `tensor_program`, `effects`, `numerical_rules` and source location ([graph.py:55-66](src/megabake/graph.py)).
  - `ComputeGraph` ([graph.py:96-106](src/megabake/graph.py)) is built by `import_fx` ([graph.py:355](src/megabake/graph.py)) and checked by `verify_compute_graph` ([graph.py:573](src/megabake/graph.py)).
- **L1b, `TensorProgram`** (the per-op/per-region loop-nest IR): `family`, `iteration_domains`, `reduction_axes`, `loads: IndexedLoad`, `init`, `combine` and `post` scalar expressions, `accumulator_dtype`, `store: TensorStore`, `predicates`, `state_accesses: KVCacheAccess`. Scalar nodes are `Constant/Load/Accumulator/Unary/AxisIndex/Binary/Where/Cast`. Indexing uses `AxisAffine`/`DataIndex`. — [src/megabake/lowering.py:16-156](src/megabake/lowering.py)
  - Families: contraction, pointwise/cast, reduction, view/copy, scatter/state, padding, opaque. Not yet lowered: scans (`cumsum`), triangular solve, multi-axis reductions. — [ARCHITECTURE.md:52-66](ARCHITECTURE.md)
- **L2, Regions.** `form_regions` ([src/megabake/regions.py:33](src/megabake/regions.py)) builds `RegionCandidate`s ([graph.py:80-93](src/megabake/graph.py)). These are a fused `tensor_program`, `semantic_facts` and `cost_bytes`, plus `rejections`. Matchers:
  - declared/stateful attention ([regions.py:182,247](src/megabake/regions.py))
  - attention prologue (RoPE into decode attention, [regions.py:329](src/megabake/regions.py))
  - scalar-chain fusion ([regions.py:447-605](src/megabake/regions.py))
  - RMSNorm ([regions.py:608-928](src/megabake/regions.py))
  - RoPE (rotate_half, [regions.py:929-1351](src/megabake/regions.py))
  - decomposed attention ([regions.py:1399-1626](src/megabake/regions.py))
  - GEMM epilogue candidates (`_gemm_epilogue_candidates`, per [ROADMAP.md:30](ROADMAP.md))
- **L3, body selection.** `BodyConfig` ([src/megabake/bodies/interface.py:138-150](src/megabake/bodies/interface.py)) is a bundle of contracts:
  - `SemanticContract` (family, `expression_hooks`)
  - `LayoutContract`
  - `TileContract` (cta_shape, mma_family)
  - `ParticipantContract` (threads, `warp_roles`, regs)
  - `PipelineContract` (stages, barriers, async ops, drain rule)
  - `ScratchContract`
  - `CompletionContract`
  - `ResourceEvidence`

  `DeviceEntry` is the body's entry `run(work_iterator, tensor_bindings, scratch, epilogue)` ([interface.py:53-62](src/megabake/bodies/interface.py)). Providers return `CandidateSet`/`ProviderRejection`. `SelectionPolicy` carries table-launch-wide GEMM knobs (`table_gemm_swap_ab`, `_rows`, `_split_k`) ([interface.py:24-44](src/megabake/bodies/interface.py)). `select_bodies`/`enumerate_body_selections` are in [src/megabake/planner.py:30-102](src/megabake/planner.py). `SelectedBody`/`BodySelectionPlan` are in [src/megabake/plan.py:216-257](src/megabake/plan.py).
- **L4, tile maps.** `TileMap` holds the output `TileGrid`, a `CuTeLayout` per input, and `TileAccess` (output `LogicalTile` + input tiles + reduction range → `WorkItem`). `ConversionBody` holds layout conversions. — [plan.py:128-275](src/megabake/plan.py); `make_tile_maps` [planner.py:465](src/megabake/planner.py), `_insert_layout_conversions` [planner.py:523](src/megabake/planner.py)
- **L5, dependencies.**
  - `RegionDependency` has kinds PRODUCER_TO_CONSUMER, LAYOUT_CONVERSION, STATE_ACCESS, WAR, WAW, with producer/consumer tile lists and a `conservative` flag ([src/megabake/schedule.py:26-65](src/megabake/schedule.py)).
  - Derived by `derive_dependencies` ([schedule.py:214](src/megabake/schedule.py)), including storage-order and state-access deps ([schedule.py:1135-1235](src/megabake/schedule.py)).
- **L6, schedules.**
  - `PhaseSchedule` is a list of `ScheduledPhase`: one region per phase, `work_items`, `WorkerAssignment` per CTA, an optional `PhaseBarrier`, and a `recompute` flag ([schedule.py:68-122](src/megabake/schedule.py)).
  - `EventSchedule` alternative: `ScheduledTask` with `wait_event_ids`/`release_event_ids` per worker, `EventScheduleEvent` with `trigger_count`, and `TilePrerequisite` ([schedule.py:133-196](src/megabake/schedule.py)). Built by `make_event_schedule`/`make_region_event_schedule` ([schedule.py:481,584](src/megabake/schedule.py)). A simulator compares the two (`simulate_event_schedule`, [schedule.py:689](src/megabake/schedule.py)).
- **L7, `KernelPlan`** = selection + dependencies + phases + tile_maps + `StoragePlan` + `LaunchRequirements` ([plan.py:277-296](src/megabake/plan.py)). `StoragePlan` handles workspace lifetimes, boundary `TensorBinding`s, per-worker scratch and KV state ([src/megabake/memory.py:23-145](src/megabake/memory.py)). `verify_plan` is at [memory.py:316](src/megabake/memory.py).
- **L8, emitted artifact.** `emit_cute` ([src/megabake/codegen.py:188](src/megabake/codegen.py)) produces a `GeneratedCuTeModule` (Python CuTe DSL source text) plus `LaunchMetadata`: `phase_table`, `gemm_table`, `stream_table`, `dynamic_slots`, `barrier_epochs`, and so on ([codegen.py:104-171](src/megabake/codegen.py), [codegen.py:1739-1793](src/megabake/codegen.py)). There are two emitters:
  - `_emit_generic_module` (parameter launch, [codegen.py:244](src/megabake/codegen.py))
  - `_emit_table_module` (table launch, [codegen.py:743](src/megabake/codegen.py))

  The table launch is chosen when the graph has >1,000 ops or >24,000 estimated parameter bytes ([src/megabake/compiler.py:184-203](src/megabake/compiler.py)), which is "every full model" ([ARCHITECTURE.md:104](ARCHITECTURE.md)).
- **Plan search.** `search_plans` ([plan.py:444](src/megabake/plan.py)) is bypassed for full models: `compile_post_grad` sets `SearchBudget(max_plans=1, max_compilations=1)` when the table launch is used ([compiler.py:408-410](src/megabake/compiler.py)). "every full model runs its greedy seed plan unmeasured" — [ARCHITECTURE.md:131](ARCHITECTURE.md)

### Inferences (awkward boundaries)
- **The region → body boundary is string-template composition, not an IR.** Bodies are emitted as standalone `@cute.kernel def run(...)` Python source. `codegen._nested_body_source` ([codegen.py:2314-2353](src/megabake/codegen.py)) parses it with `ast`, keeps only `run`, and text-replaces `@cute.kernel`→`@cute.jit`, the `block_idx/thread_idx/grid_dim` lines, and `scratch[block,`→`scratch[`. `_affine_work_iterator` string-replaces `work_iterator[tile]` ([codegen.py:2356-2365](src/megabake/codegen.py)). Memory confirms the fragility: "helper functions in a body's source are dropped, so inline the code" (cute-dsl-kernel-gotchas).
- **Whole-launch GEMM uniformity leaks into selection policy.** One GEMM tile/shared-memory layout per table launch ([codegen.py:1563-1567](src/megabake/codegen.py) raises "table-launched GEMMs must share one tile and shared-memory layout"). swapAB is all-or-nothing per launch ([interface.py:36-43](src/megabake/bodies/interface.py)). Generic bodies are force-widened to the GEMM's 256 threads (`_normalize_generic_selection`, [codegen.py:2104-2130](src/megabake/codegen.py)). So per-region choices are not independent.
- **The event schedule is a full parallel IR that the production path doesn't use.** Table launches emit phases only ([ARCHITECTURE.md:104-106](ARCHITECTURE.md)). `EventSchedule` is emitted only by `_emit_generic_module` ([codegen.py:556-629](src/megabake/codegen.py)). `scheduler="auto"` measures both, but only for small graphs ([compiler.py:452-475](src/megabake/compiler.py)).
- **No IR level owns cross-phase resources (shared memory, pipeline state).** Weight streaming had to piggy-back a "tag" in the GEMM's shared storage (`split_arrival[STREAM_TAG_SLOT]`, [src/megabake/bodies/cutlass_gemm.py:124,815-816](src/megabake/bodies/cutlass_gemm.py)). It also needed a hack to suppress mbarrier init (`_without_mbarrier_init`, [cutlass_gemm.py:128](src/megabake/bodies/cutlass_gemm.py)). This is the gap Hazy's page allocator fills.
- Cost/tuning facts are thin. `TargetFacts` has no L2, bandwidth or barrier latency ([interface.py:12-21](src/megabake/bodies/interface.py)), so no IR level can rank alternatives.

### Gaps
- I did not trace every lowering rule in `lowering.py` or the full body-option enumeration in `planner._enumerate_body_selections` ([planner.py:102-353](src/megabake/planner.py)).

---

## Q3. How are kernel bodies written (CuTe DSL?), what is their structure and warp roles, and how are they composed?

### Takeaway
All device code is **NVIDIA CuTe DSL (Python)**. There are three kinds of body:

1. **Generated** map, reduction and contraction bodies, emitted as Python source strings from `TensorProgram`s.
2. A **generated decode/prefill attention** body, also source text.
3. A **forked CUTLASS Hopper persistent TMA+WGMMA GEMM** class.

In the table megakernel every body runs on the same 256 threads (two warpgroups). Only the GEMM is warp-specialized: one TMA producer warpgroup and one WGMMA/epilogue warpgroup. Other bodies use all threads uniformly and call `sync_threads`.

### Cited Findings
- **Generated map body template.**
  - `@cute.kernel def run(work_iterator, tensor_bindings, scratch, epilogue)`, then a grid-stride loop `for tile in cutlass.range(block, size(work_iterator), workers)`, then hoisted loads, expression, cast and predicated store — [src/megabake/bodies/generic.py:226-245](src/megabake/bodies/generic.py)
  - Emitters: `emit_map_body` [generic.py:152](src/megabake/bodies/generic.py), `emit_reduction_body` [generic.py:258](src/megabake/bodies/generic.py), row reductions [generic.py:482](src/megabake/bodies/generic.py), `emit_contraction_body` (slow fallback) [generic.py:666](src/megabake/bodies/generic.py).
  - Expressions are emitted by `_emit_expression` [generic.py:1030](src/megabake/bodies/generic.py).
  - Generic bodies' warp roles are just `("worker",) * (threads//32)` — [src/megabake/bodies/catalog.py:213-216](src/megabake/bodies/catalog.py)
- **GEMM body** (`G0RangeBodyProvider`, provider name `g0_range`):
  - [src/megabake/bodies/gemm.py:261](src/megabake/bodies/gemm.py) drives `HopperWgmmaGemmPersistentKernel` in `bodies/cutlass_gemm.py`, a fork of pinned CUTLASS `dense_gemm_persistent.py`. `tools/vendor_cutlass.py` shows the diff.
  - Fixed config: 64x64x64 tile, 8 stages; swapAB for ≤64 rows (weight on WGMMA M, tokens on N = 16/32/64) — [ARCHITECTURE.md:35,74](ARCHITECTURE.md)
  - Contract warp roles: 4×`tma_load` + 4×`wgmma_epilogue`, 256 threads, 232 regs — [gemm.py:643-659](src/megabake/bodies/gemm.py)
  - In the kernel: `num_dma_warp_groups=1`, `num_mma_warp_groups=prod(atom_layout)` (1 for 64-wide tiles), `load_warp_id=0` — [cutlass_gemm.py:245-266](src/megabake/bodies/cutlass_gemm.py)
  - Warp 0 runs a `StaticPersistentTileScheduler` producer loop (`producer_acquire` → TMA A/B → `producer_tail`) — [cutlass_gemm.py:1101-1194](src/megabake/bodies/cutlass_gemm.py)
  - MMA warpgroups run `consumer_wait`, WGMMA, then the epilogue — [cutlass_gemm.py:1201-1320](src/megabake/bodies/cutlass_gemm.py)
  - Register reallocation (`setmaxregister`) is disabled in mixed megakernels because other bodies run on the same warps afterwards — [cutlass_gemm.py:268-273](src/megabake/bodies/cutlass_gemm.py)
  - A `fence_proxy("async.global")` orders other CTAs' generic stores before TMA loads — [cutlass_gemm.py:1117-1120](src/megabake/bodies/cutlass_gemm.py)
  - Each table GEMM phase enters through `run_table(prepared, scratch, tma_descs, rotated_block, split_state, bias_address, c_address, epilogue_state, stream_tag)` — [cutlass_gemm.py:659-716](src/megabake/bodies/cutlass_gemm.py)
- **GEMM epilogue hooks.**
  - The region's `post` scalar expression is generated into `@cute.jit def epilogue(accumulator, accumulator_1, o0, o1, tensor_bindings)`, evaluated per output element — [gemm.py:202-222](src/megabake/bodies/gemm.py)
  - GEMMs of one shape share one kernel body and choose the epilogue kind at run time from a `gemm_table` column — [ROADMAP.md:30](ROADMAP.md); [codegen.py:1797-1800](src/megabake/codegen.py)
  - Paired gate/up GEMM: two TMA descriptors fill the halves of a 128-row A stage — [ARCHITECTURE.md:75](ARCHITECTURE.md); [cutlass_gemm.py:790-811](src/megabake/bodies/cutlass_gemm.py)
- **Attention bodies.**
  - Prefill: one warp per query row, keys in sequence — [ARCHITECTURE.md:76](ARCHITECTURE.md)
  - Its warp roles string is `"one query row at a time: lanes split keys, then head dimensions"` ×4 — [src/megabake/bodies/attention.py:260](src/megabake/bodies/attention.py)
  - Split-context decode (flash-decoding): work item = row × chunk. Chunks exchange max/sum through phase scratch with spin waits. The last chunk merges in order (deterministic). An optional RoPE prologue computes q/new-k into shared scratch and appends to the KV cache — [attention.py:764-800](src/megabake/bodies/attention.py)
  - Chunk count comes from a hand cost formula (`decode_chunks`, `DECODE_MAX_CHUNKS`) — [attention.py:748-762](src/megabake/bodies/attention.py)
  - No tensor cores in attention (prefill attention is ROADMAP step 13) — [ROADMAP.md:82](ROADMAP.md)
- **Blackwell** has only a `mma.sync` correctness baseline — [ARCHITECTURE.md:80](ARCHITECTURE.md); [src/megabake/bodies/blackwell.py:74-76](src/megabake/bodies/blackwell.py)
- **Composition into the megakernel (table launch).**
  - Each distinct body variant is inlined once as a nested `@cute.jit` function.
  - Phases of one variant share a `_run_variant_N(phase_index)` runner that binary-searches `phase_index` to its specialised call — [codegen.py:2030-2059](src/megabake/codegen.py)
  - The main loop does `variant = make_warp_uniform(phase_table[phase,0])` and dispatches with `if variant == k:` — [codegen.py:1451-1454, 1673-1688](src/megabake/codegen.py)
  - Tensors are rebuilt from a device address table via `_slot(addresses, phase_table, phase, column, dtype, layout, alignment)` — [codegen.py:1614-1638](src/megabake/codegen.py)
  - TMA descriptors for every GEMM phase are written in the kernel prologue (`build_table_descriptors`), followed by a grid barrier — [gemm.py:89](src/megabake/bodies/gemm.py); [codegen.py:1560-1565](src/megabake/codegen.py)
- **Shared memory** = one GEMM stage layout (union over paired variants) + a generic scratch union + 1 KiB; it must fit `max_shared_memory_per_cta` — [codegen.py:1563-1582](src/megabake/codegen.py)

### Inferences
- In Hazy terms, a MegaBake "instruction" is a **whole phase run by all CTAs over a contiguous tile range**, not a per-SM op with its own page allocation. The GEMM keeps its internal producer/consumer pipeline, but the pipeline is created, filled and drained within each phase. There is no persistent pipeline state across ops, and no producer warp that outlives a body. ROADMAP says making streaming pay "needs generic and attention bodies on 7 warps, with named barriers" ([ROADMAP.md:65](ROADMAP.md)).

### Gaps
- I did not inspect the reduction or prefill-attention source templates line by line.

---

## Q4. How does the device-side runtime work, and what is the scheduler logic?

### Takeaway
The runtime is a **cooperative persistent kernel: grid = SM count, one CTA per SM, 256 threads**. It runs a `for phase in range(N)` interpreter over a host-built **phase table**:

- The table gives each phase a variant id, a barrier ordinal, a **rotation** (static tile-range offset) and a unit count.
- Work is **statically partitioned**: contiguous tile ranges per CTA, rotated so concurrent phases in one interval start on different CTAs.
- Synchronization is a **monotonic counting grid barrier** (one `red.release.add` arrival plus acquire polling) between "barrier intervals". Independent phases share an interval.
- Small producers are **folded** (recomputed redundantly by every CTA) into their consumer's interval.
- Event counters exist but only for small parameter-launch graphs.
- Weight streaming (preload into shared memory at the barrier) is built and off.

### Cited Findings
- **Kernel skeleton** — [codegen.py:1643-1712](src/megabake/codegen.py):
  - Reads the call counter. Block 0 publishes this call's dynamic boundary addresses (from a host-mapped ring), then `counting_grid_barrier`.
  - Optional GEMM descriptor prologue and barrier.
  - Per phase: L1-prefetch the next phase-table row, dispatch the variant if `_rotated < _units`, apply `fence_proxy("async.global")` after recompute phases, then either `counting_grid_barrier(ready, workers, _ordinal + phase_table[phase,1])` or a plain `sync_threads()` when the phase has no barrier.
  - KV valid-length increments happen after the barrier that closes its readers' interval.
- **Launch.** `.launch(grid=[1,1,workers], block=[threads], cooperative=True)` — [codegen.py:1726-1737](src/megabake/codegen.py)
- **Phase table columns** (as used): 0 variant, 1 barrier ordinal (0 = none), 2 rotation (−1 = recompute), 3 KV-length slot, 4/5 phase-scratch offsets, 6+ resolved tensor slots (GEMM rows hold their `gemm_table` row in col 6), then units and optional stream row — [codegen.py:1205-1239, 1366-1387](src/megabake/codegen.py)
- **Work partitioning.**
  - `_worker_assignments` splits work items into contiguous `divmod` chunks per worker — [schedule.py:1504-1520](src/megabake/schedule.py)
  - Worker count = min(max tiles in any phase, SMs × min active blocks) — [schedule.py:1358-1380](src/megabake/schedule.py)
  - Table codegen assigns each phase a rotation offset: phases in one barrier interval start "where the previous phase's tiles ended", modulo grid size — [codegen.py:1220-1232](src/megabake/codegen.py)
  - CTAs with no tiles skip the body (idle-phase skipping, −9% on 135M per memory/event-runtime-milestone).
- **Phase ordering.**
  - `make_phase_schedule`: topological order, then `_level_order` (one dependency level per barrier), then optional `_fold_small_intervals` — [schedule.py:288-393](src/megabake/schedule.py)
  - Folding is enabled whenever `table_launch` — [src/megabake/plan.py:579](src/megabake/plan.py)
- **Interval folding.**
  - Runs of intervals holding only small generic elementwise/copy/reduction/gather/concat bodies are folded into the next interval as recompute phases: every CTA executes all of their tiles. Limits are 2 rounds per phase, 6 rounds and 6 phases total.
  - Folding is refused if the receiving interval writes storage the run touches.
  - The docstring cites Mirage fusing these into linear/attention tasks.

  — [schedule.py:1376-1501](src/megabake/schedule.py). Effect: a SmolLM layer goes from ~11 to ~6 barrier intervals ([ARCHITECTURE.md:88](ARCHITECTURE.md)), and after step 1 to 5 intervals / 9 phases ([ROADMAP.md:21-27](ROADMAP.md)).
- **Grid barrier.**
  - `counting_grid_barrier`: `sync_threads`, then thread 0 does `red.add.release.gpu` on a counter that only grows, then spins with `ld.acquire.gpu` until the counter reaches `(ordinal+1)*workers`, then `sync_threads` — [src/megabake/runtime.py:156-199](src/megabake/runtime.py)
  - Ordinals are dense per call (`_ordinal = _call * barrier_epochs`), so no reset is needed between calls — [codegen.py:1371-1379, 1659-1660](src/megabake/codegen.py)
  - The arrive and wait halves are split so a CTA can issue independent work (weight loads) in between — [runtime.py:175-185](src/megabake/runtime.py)
  - Cost is about 0.7 µs at 60 CTAs ([ARCHITECTURE.md:86](ARCHITECTURE.md)). An older design that polled every worker's flag cost 13 µs per barrier ([runtime.py:102-106](src/megabake/runtime.py)).
- **Events (built, not used for full models).**
  - `publish_event` does `atomic_add release`. `wait_for_event` spins with `atomic_cas` acquire until `(epoch+1)*trigger_count` — [runtime.py:287-324](src/megabake/runtime.py)
  - Event codegen exists only in `_emit_generic_module` — [codegen.py:556-629](src/megabake/codegen.py)
  - Event-schedule tasks are statically assigned to workers ([schedule.py:156-172](src/megabake/schedule.py)). Only ARCHITECTURE notes the missing gcd/lcm grouping.
  - Synchronization rule: drain async ops, then `sync_threads`, then release publish; acquire poll, then `sync_threads`; add proxy fences for TMA — [ARCHITECTURE.md:110](ARCHITECTURE.md)
- **Weight streaming (step 3, off by default via `MEGABAKE_WEIGHT_STREAM_STAGES=0`).**
  - `_weight_stream_schedule` builds a stream table: per closing barrier and per CTA, (GEMM row+1, variant, rotated block) for that CTA's first GEMM tile in the next interval — [codegen.py:1856-1908](src/megabake/codegen.py)
  - The barrier becomes: `counting_grid_arrive`, then warp 0 runs `_gemm_N.preload_table(...)`, then `counting_grid_wait` — [codegen.py:1911-1941](src/megabake/codegen.py)
  - `preload_table` issues TMA loads of A (the weight) for `min(k_tiles, ab_stage, stream_stages)` stages and writes a tag plus the K-tile count into shared memory — [cutlass_gemm.py:718-820](src/megabake/bodies/cutlass_gemm.py)
  - The GEMM phase checks the tag, skips mbarrier init and loads only B for those K tiles — [cutlass_gemm.py:1122-1126](src/megabake/bodies/cutlass_gemm.py); [ARCHITECTURE.md:90-97](ARCHITECTURE.md)
  - Only valid because "only GEMM bodies use the GEMM's shared memory", and only for weights no region writes — [ARCHITECTURE.md:95](ARCHITECTURE.md)
- **L2 prefetch of the next weights.** `cp.async.bulk.prefetch.L2` was built, measured 8–24% slower, and was removed in the 2026-10-22 cleanup — [ROADMAP.md:92](ROADMAP.md); memory/cleanup-2026-10-22
- **Memory planning.** Workspace slices are reused only after all consumers (including async readers) finish; phase lifetimes prove reuse; per-worker scratch is allocated per phase — [ARCHITECTURE.md:108](ARCHITECTURE.md); `assign_storage` [memory.py:148](src/megabake/memory.py). Phases sharing an interval get disjoint phase-scratch regions, reset at each barrier — [codegen.py:1203-1218](src/megabake/codegen.py)
- **Host per call.** Kernel arguments are constant between calls. Boundary addresses go through a host-mapped ring indexed by a device call counter, so "a call packs nothing and does not block" — [ARCHITECTURE.md:45](ARCHITECTURE.md); [codegen.py:1455-1460](src/megabake/codegen.py). Host work per call (Dynamo guards, binding) is still ~600 µs — [ROADMAP.md:75](ROADMAP.md)
- **Split-K.** Deterministic split-K uses tile arrival counters in phase scratch. It is built and off ("no gain on 60 SMs"); two splits would need 64 CTAs — [ROADMAP.md:78,148](ROADMAP.md); [cutlass_gemm.py:670-676](src/megabake/bodies/cutlass_gemm.py)

### Inferences
- Compared with an on-GPU interpreter that has per-SM instruction lists and counters (the talk's summary of Hazy, [megakernels talk.md:88-133](agent_space/research_notes/megakernels%20talk.md)), MegaBake differs in five ways:
  1. One global phase list that every CTA walks. A CTA's work is implied by rotation and units, not by a per-SM queue.
  2. Global barriers between intervals instead of fine-grained counters.
  3. No shared-memory paging. One static union layout, with GEMM stages reserved for GEMMs.
  4. Pipelines do not persist across ops.
  5. All 256 threads take part in every body, so no warp can run ahead (e.g. loading the next op's weights) while others compute.
- Interval folding (redundant recompute of norms/RoPE on every CTA) is MegaBake's substitute for Hazy-style fused "RMSNorm + QKV + RoPE" instructions.

### Gaps
- I did not verify how `_emit_generic_module` maps events onto workers (`_event_worker_loop`, [codegen.py:2062](src/megabake/codegen.py)) beyond the snippets cited.

---

## Q5. What perf problems and debugging pains are recorded (including memory notes)?

### Takeaway
Decode is limited by **per-phase fixed latency and GEMM pipeline fill on few CTAs**, not by barriers. The biggest engineering pain is that **ptxas treats the megakernel as one unit**:

- Register allocation and wgmma serialization (C7519) are whole-kernel effects, so unrelated code can move performance by ±7–20%.
- Every A/B therefore needs an "inert code" control.

Weight streaming and L2 prefetch both failed to pay.

### Cited Findings
**Measured state** (H200 MIG 3g.71gb, 60 SMs, ~2.2 TB/s, shared, 10–15% noise; bf16, batch 1) — [ROADMAP.md:123-152](ROADMAP.md):
- Decode kernel: SmolLM-135M 1,186 µs, SmolLM-1.7B 2,665 µs, Qwen2-0.5B 1,364 µs, gemma-2b 3,318 µs, Gemma-3-270M 1,793 µs, Qwen3.5-0.8B 4,445 µs.
- Pipelined decode against graphed Inductor (µs, MegaBake / Inductor):

  | Model | MegaBake | Graphed Inductor |
  | --- | ---: | ---: |
  | SmolLM-135M | 1,208 | 2,019 |
  | SmolLM-1.7B | 2,576 | 3,015 |
  | Gemma-3-270M | 1,818 | 1,416 |
  | Qwen3.5-0.8B | 4,490 | 2,319 |

- Bandwidth utilization in decode: ~55% on 1.7B and ~10% on 135M (Hazy: 78% on H100, [megakernels talk.md:134-143](agent_space/research_notes/megakernels%20talk.md)).
- Per-CTA profile:
  - 135M decode: 278 phases, 151 intervals, ~1.0 ms span.
  - 1.7B: 224 phases, 121 intervals, 2.31 ms span.
  - Barrier release ~0.74 µs each, ~11% of the span.

**Where the time goes**:
- The remaining 135M decode time is in four small GEMM intervals per layer (~3–4 µs each, "mostly pipeline fill: an o_proj tile moves ~90 KB") and attention (~6 µs on 9 CTAs) — [ROADMAP.md:35](ROADMAP.md)
- On 1.7B:
  - The 32-tile projections (q/k/v/o/down) run on 32 of 60 SMs at 70–72% of peak.
  - Attention costs ~4.7 µs × 24.
  - Barrier release costs ~1.3 µs × 121.
  - gate·up runs at ~90% of peak and the LM head at peak.

  — [ROADMAP.md:64](ROADMAP.md)
- Before fusion, per layer: RoPE 2.7 µs, attention 8.5 µs (9/60 CTAs busy), o_proj 3.8 µs, add+norm+gate+up 7.4 µs — [ROADMAP.md:144-148](ROADMAP.md)

**Decode perf milestone** (memory/event-runtime-milestone, 2026-10-20):
- The old "1.2–5x" wins came from `.item()` host syncs in an ungraphed Inductor baseline.
- The critical path is per-phase work latency; events alone would gain little.
- The levers that worked were idle-phase skipping (−9% on 135M) and flattening the table indirection.
- L2 prefetch by every CTA made decode 16–24% slower.
- Qwen3.5 decode: 5.7 ms vs 2.35 ms graphed.

**C7519 (wgmma serialization)** — [ROADMAP.md:37-45](ROADMAP.md); memory/cute-dsl-kernel-gotchas:
- ptxas warning: "wgmma.mma_async instructions are serialized due to program dependence on compiler-inserted WG.AR in divergent path".
- Cost: +43% SASS instructions (R2UR, WARPGROUP fences) and +20% time on 1.7B prefill.
- Root cause: any `llvm.intr_trap()` behind a per-lane condition anywhere in the megakernel (the gather bodies' index bounds checks) makes ptxas serialize every wgmma.
- Fix: `runtime.trap_if` votes across the warp (`vote_any_sync`) before trapping — [runtime.py:142-154](src/megabake/runtime.py)
- Result: prefill −19 to −21% on 1.7B, Qwen2-0.5B and DeepSeek draft. `tools/wgmma_serialization.py` checks all 15 kernels.
- Earlier, apparent "toggles" (variant count, dropping a body) only changed whether ptxas could see the divergence.

**Register allocation is whole-kernel** — [ROADMAP.md:45,156,170](ROADMAP.md); memory/cute-dsl-kernel-gotchas, memory/weight-streaming-step3:
- A GEMM-only change moved untouched attention by ±1 µs per phase (1.7B attention 4.95 → 5.65 µs when registers went 244 → 128).
- The shape-agnostic table GEMM (188a98e) cost decode +2–4% (registers 174/244 → 128) and was reverted.
- 135M runs 6.8% faster at 239 registers than at 174 for no understood reason (open question).
- Keep `accumulators.fill(0.0)`: `ACCUMULATE=False` keeps accumulators live across the whole phase loop (+40–50 registers).

**Weight streaming, step 3** — [ROADMAP.md:47-65](ROADMAP.md); memory/weight-streaming-step3:
- The apparent −7% on 135M was register shape (174 → 239). Against the "code compiled in, zero stream table" control, the preloads themselves are −0.5% (135M), −0.7% (1.7B), +0.9% (Qwen2), +1.5% (DeepSeek).
- Why: once the SM's TMA queue is full, each further load blocks warp 0. Issuing 13 stages takes a median of ~1,000 and a p90 of ~4,000 cycles, so a CTA waits for most of its own preload before leaving the barrier. Each preload-issuing interval grows ~1.5–2 µs.
- Tried without gain: 2/4/8/all stages, L1-prefetching the stream-table entry, `prefetch.tensormap`, 2-interval lookahead, adaptive issue.
- Conclusion: it needs a producer warp decoupled from the CTA's syncs.

**Other CuTe DSL pitfalls** (memory/cute-dsl-kernel-gotchas):
- `thr_mma.partition_C` gives warpgroup-slice coordinates (every thread sees thread 0's coordinates).
- Layouts built in host `prepare()` cannot be used in the kernel.
- `_nested_body_source` drops helper functions.
- Deep expressions overflow the DSL preprocessor's recursion, so loads are hoisted.
- Per-warp scratch must scale when generic bodies are widened to 256 threads.
- A kernel that writes nothing leaves plausible stale memory.
- Fast debugging loop: `CUTE_DSL_KEEP=ptx` plus a CUDA 13.4 ptxas, ~1 s per try vs ~50 s per model compile.

**Process pains**:
- GPU tests and benchmarks are slow (compiles take 20 s to minutes; SmolLM-1.7B at the 512 bucket takes 41 minutes) — memory/batch-testing-and-benchmarks; [ROADMAP.md:165](ROADMAP.md)
- Interleaved same-session A/B via worktrees is required because the MIG slice is noisy — memory/cute-dsl-kernel-gotchas
- GPU tests need `MEGABAKE_RUN_GPU_TESTS=1`. Baseline: 41 pass, 14 skip, 1 known failure (`test_phase_runtime_ports_delay_idle_rebinding_and_largest_grid`) — memory/cleanup-2026-10-22
- The 2026-10-22 cleanup gained −0.5% to −3.3% purely from code shape and table size — memory/cleanup-2026-10-22

**Overfitting and generality** — [docs/research/2026-10-18-megakernel-gap/report.md §4](docs/research/2026-10-18-megakernel-gap/report.md):
- No plan search for full models.
- One GEMM config per launch.
- Fold limits 2/6/6 and decode chunk constants fitted to SmolLM and the 60-SM slice.
- bf16 and the `sm90a` string only.
- Llama-shaped matchers: RMSNorm has one ATen spelling and RoPE is rotate_half only.

**Open questions** — [ROADMAP.md:154-161](ROADMAP.md):
- the register effect on 135M
- the L2 share of the MIG slice
- whether the RMSNorm GEMM prologue stays bit-exact
- whether split-K helps on 132 SMs
- whether the plan cache key covers matcher and body source

### Inferences
- The recorded failures (streaming blocked on the TMA queue, pipeline fill dominating small GEMMs, all warps synchronizing per body, whole-kernel register coupling) all point at the same missing structure: warp-specialized producer/consumer roles that persist across instructions, plus a shared-memory resource manager. These are exactly the areas a ThunderKittens/Megakernels comparison should examine.

### Gaps
- I did not open the artifacts directories (`artifacts/step3_2026-10-21`, etc.). The numbers are taken from ROADMAP and memory.

---

## Q6. What do the recent git commits reveal about current struggles?

### Takeaway
The last week of commits is a cycle of building a megakernel-only optimization, finding that compiler-level effects (ptxas serialization, register allocation) swamp it, and keeping only the robust part.

### Cited Findings
- `09eeedb` "attempting cross gemm fusion" (2026-10-21, +5,163/−2,147):
  - Vendored CUTLASS `dense_gemm_persistent.py` (1,616 lines) and forked it into `cutlass_gemm.py` (+1,964).
  - Rewrote `gemm.py` (−1,942 net), grew `regions.py` by 683 lines (epilogue candidates), and added `tools/wgmma_serialization.py` and `tools/vendor_cutlass.py`.

  — `git show --stat 09eeedb`
- `188a98e` "removing wgmma serialization": touched `runtime.py` (`trap_if`), `codegen.py`, `cutlass_gemm.py` and generic/attention bodies — `git show --stat 188a98e`. It also contained the shape-agnostic table GEMM.
- `c2b4d57` "retaining only trap fix": reverted the cutlass_gemm/codegen part of 188a98e (−85 lines). This matches ROADMAP "Tried and reverted: one GEMM body per configuration … made decode 2–4% slower" — [ROADMAP.md:45](ROADMAP.md)
- `6165fc1` "experimental weigh streaming enabled" (2026-10-22): +268 lines in `cutlass_gemm.py` (`preload_table`, `_without_mbarrier_init`), +257 in `codegen.py` (stream table, split barrier), +33 in `runtime.py` (`counting_grid_arrive`/`wait`). Off by default after measurement.
- `fb665b1` "agentic cleanup": removed L2 prefetch, unread contract prose fields and duplicate dtype tables (memory/cleanup-2026-10-22).
- Earlier: `9e52793` "Cut per-phase latency in table launches; add graphed decode baseline"; `d370e09` "Use the table launch when kernel parameters would overflow"; `0ab80d6` "Add Blackwell megakernel support" — `git log --oneline -30`

### Inferences
- The project's real bottleneck is no longer feature coverage. Each perf feature has to survive a fragile whole-kernel codegen. The fork-and-patch CUTLASS approach means the GEMM's internal structure (its own tile scheduler, its own per-phase pipeline creation) is inherited rather than designed for megakernel residency.

### Gaps
- None material.

---

## Q7. Does existing research in `agent_space/` already compare MegaBake with HazyResearch?

### Takeaway
There is no dedicated MegaBake-vs-ThunderKittens/Kittens-VM architectural comparison.

- The 2026-10-18 gap report compares MegaBake mainly against **Mirage MPK**, vLLM, the Inferact TPU work and Inductor. It cites Hazy only for headline numbers: 78% bandwidth, smem pages as a streaming example, and a −1.9% result for dynamic scheduling.
- A talk outline (`agent_space/research_notes/megakernels talk.md`) summarizes Hazy's "No Bubbles" design at blog level and maps MegaBake onto it.

The `agent_space/Megakernels` and `agent_space/ThunderKittens` checkouts exist but no notes analyse their code.

### Cited Findings
- **Gap report scope**:
  - §2 is a "How far we are from Mirage" table.
  - §3 lists megakernel-only optimizations, with Hazy cited for "smem pages" under cross-op weight streaming.
  - §4 covers overfitting.
  - §5 gives the recommended order.
  - §6 covers vLLM DeepSeek V4.1 lessons.

  — [agent_space/megakernel_gap_research/report.md:1-213](agent_space/megakernel_gap_research/report.md) (a slightly differing copy is at [docs/research/2026-10-18-megakernel-gap/report.md](docs/research/2026-10-18-megakernel-gap/report.md))
- **Hazy mentions in the report**:
  - "Published hand-built megakernels reach 50–86% (ForgeMegakernel) and 78% (Hazy) on H100" — report.md:24
  - "Cross-op weight streaming (TPU 12-bank ring; Hazy smem pages; …)" — report.md:61
  - "Don't build: a dynamic work queue … Hazy measured −1.9%" — report.md:72
- The report's main estimate, that events would give −17–30% on 135M, was later refuted. The per-CTA profile bounds the gain at ~11% ([ROADMAP.md:93](ROADMAP.md)).
- **JSON evidence files** cover Mirage MPK (runtime, task library, superoptimizer), Inferact TPU, FlashMLA mega-attention, vLLM, Inductor, layout co-design, quantized contracts, mixed prefill/decode, state of the art 2025–26 and the overfitting audit. There is no ThunderKittens- or Hazy-specific file — `ls agent_space/megakernel_gap_research/results/`
- **Talk outline** (`megakernels talk.md`) summarizes Hazy's design:
  - an on-GPU interpreter with per-SM instruction lists computed ahead of time in Python;
  - a 7-instruction set for Llama-1B ("RMSNorm + QKV + RoPE", "Attention", "Attention reduction", "O-proj + residual", "RMSNorm + up/gate + SiLU", "Down-proj + residual", "RMSNorm + LM head");
  - 13 × 16 KiB shared-memory pages with request/release hand-off;
  - counters instead of barriers;
  - 78% of H100 bandwidth, ~2.5x vLLM;
  - a B200 breakdown in which waiting on weights is 30 µs.

  It frames MegaBake bodies as instructions and the plan as the schedule — [agent_space/research_notes/megakernels talk.md:88-177](agent_space/research_notes/megakernels%20talk.md)
- `agent_space/research_notes/CUDA megakernel study plan.md` is a learning plan keyed to ROADMAP steps, not a comparison — [CUDA megakernel study plan.md:1-119](agent_space/research_notes/CUDA%20megakernel%20study%20plan.md)

### Inferences
- New work should not re-derive the Mirage comparison, the overfitting audit or the event-gain estimates. The open ground is a code-level comparison with ThunderKittens/Megakernels (`agent_space/Megakernels`, `agent_space/ThunderKittens`) on four points:
  1. instruction format and per-SM queues;
  2. page allocator and smem hand-off;
  3. warp-role specialization (loader, consumer, storer, launcher) persisting across instructions;
  4. counter granularity.

  These map directly onto MegaBake's recorded failures (Q5).

### Gaps
- I did not read the report's JSON result files, and I did not open any HazyResearch code (out of scope).
