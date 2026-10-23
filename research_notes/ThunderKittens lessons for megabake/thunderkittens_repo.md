# ThunderKittens repo: kernel-body abstractions, the Kittens VM / megakernel prototype, and what TK 2.0 removed

Scope: local clone at `/workspace/megabake/agent_space/ThunderKittens`. HEAD = `67845f5f`, 2026-09-12 ("Merge pull request #210 ... vr200-output-gemms"), so the clone is current as of Sept 2026 and I did not need upstream GitHub. Links below are pinned GitHub URLs for the same commits. `H` = HEAD `67845f5f`. `OLD` = `50ee1f0a` (2025-11-15), the parent of the TK 2.0 commit, used for files that TK 2.0 deleted. Line numbers were checked against the local files.

## Q1. Core abstractions and how they compose into a kernel body

### Takeaway
TK is a header-only C++ template library. It has register tiles (`rt`), shared tiles (`st`, swizzled), Blackwell tensor-memory tiles (`tt`), vectors (`rv`/`sv`), global layouts (`gl`), and `group<N>`-scoped ops (`warp` = `group<1>`, `warpgroup` = `group<4>`). It also wraps mbarriers (`semaphore`), TMA, wgmma and tcgen05. The LCF/LCSF "producer/consumer template" layer is still in `prototype/`. In TK 2.0, though, the flagship Blackwell kernels skip it. They write the warp-specialized body by hand as `if (warpgroup::groupid()==...)` branches over raw primitives.

### Cited Findings
- **Group scoping.** `template<int _GROUP_WARPS> struct group` defines `laneid/warpid/groupid` and `sync(id)` as `bar.sync id, GROUP_THREADS`. It also has `arrive(id)`. Memory, shared, register, mma and util ops are `#include`d *inside* the struct, so every op has a group-parameterized version — [include/ops/group/group.cuh:25-48](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/group/group.cuh#L25-L48). `using warp = group<1>; using warpgroup = group<4>;` — [group.cuh:114-115](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/group/group.cuh#L114-L115). `tma` and `tma::cluster` are nested structs inside `group`, because "We can't use namespaces as this is under struct group" — [group.cuh:63-77](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/group/group.cuh#L63-L77). `everyone::sync` (whole-CTA `bar.sync`) and `everyone::tma::cluster::{arrive,wait}[_aligned]` (`barrier.cluster.*`) — [group.cuh:83-110](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/group/group.cuh#L83-L110).
- **Register tile.** `rt<T, rows, cols, layout=row>` holds `rt_base<T,layout> tiles[height][width]`. Rows and cols must be multiples of the base tile size. Commit `02e9acbd` (2026-06-15) bans bare `char` because "signedness is ISA-dependent" — [include/types/register/rt.cuh:81-111](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/types/register/rt.cuh#L81-L111).
- **Shared tile.** The signature is `st<T, rows, cols, swizzle=true, swizzle_bytes=0>`. The swizzle is computed in `idx()` as `addr ^ (((addr % (swizzle_bytes*8)) >> 7) << 4)`, and `swizzle_bytes` must be 32, 64 or 128 — [include/types/shared/st.cuh:58-129](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/types/shared/st.cuh#L58-L129).
- **Other types.** Global layout is `gl` at [include/types/global/gl.cuh:113](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/types/global/gl.cuh#L113). Tensor-memory tile is `tt` at [include/types/tensor/tt.cuh:57](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/types/tensor/tt.cuh#L57). Register vector is `rv` at [include/types/register/rv.cuh:62](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/types/register/rv.cuh#L62).
- **TMEM allocator.** `tensor_allocator<nblocks_per_sm, ncta, managed=true, exclusive=false>`. `provision()` issues `tcgen05.alloc` + `relinquish_alloc_permit` and must be called by one whole warp. The managed constructor allocates TMEM in warp 0 and then runs fence / `bar.sync 0` / fence. `allocate<TT>(superlane, col_offset)` hands out `tt` views by column offset — [include/types/tensor/tensor.cuh:36-104](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/types/tensor/tensor.cuh#L36-L104).
- **Semaphore.** `semaphore` is an opaque `uint64_t` mbarrier — [include/ops/thread/util/sync.cuh:13-16](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/thread/util/sync.cuh#L13-L16).
  - `init_semaphore` → `mbarrier.init` — [sync.cuh:37-44](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/thread/util/sync.cuh#L37-L44).
  - `wait(sem, phase)` is a `mbarrier.try_wait.parity` spin loop with `bra.uni` on SM90+, and `test_wait`+`nanosleep 5` before that — [sync.cuh:106-140](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/thread/util/sync.cuh#L106-L140).
  - `test_wait` is non-blocking — [sync.cuh:212](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/thread/util/sync.cuh#L212).
- **Smem allocation.** `shared_allocator` (default 1024-byte alignment on SM90+) carves typed N-d arrays out of `extern __shared__` — [include/common/util.cuh:250-275](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/common/util.cuh#L250-L275). Modern kernels use `tma_swizzle_allocator al(...); al.allocate<tile, DEPTH, N>()` — [kernels/gemm/bf16_b200/bf16_b200_gemm.cu:77-82](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/bf16_b200/bf16_b200_gemm.cu#L77-L82).
- **wgmma wrappers (Hopper).** `warpgroup::mma_fence` (`wgmma.fence`), `mma_commit_group`, `mma_async_wait<N>` (`wgmma.wait_group N`) and `mma_AB(...)` — [include/ops/group/mma/warpgroup.cuh:76-101,148](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/group/mma/warpgroup.cuh#L76-L148).
- **tcgen05 wrappers (Blackwell).** `mma/mm/mma2/mm2` and the `_AB/_ABt` variants take a completion `semaphore&`. The "2" variants are cta_group::2 (2-CTA). `mm` overwrites the accumulator and `mma` accumulates — [include/ops/group/mma/tcgen05.cuh:6-100](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/group/mma/tcgen05.cuh#L6-L100). The PTX strings `tcgen05.mma.cta_group::{1,2}.kind::{f16,f8f6f4,i8}` live in [include/ops/thread/mma/tcgen05.cuh:240-331](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/thread/mma/tcgen05.cuh#L240-L331).
- **Grid scheduling helpers (Blackwell).**
  - `clc::schedule/query` wrap `clusterlaunchcontrol.try_cancel` / `query_cancel` for dynamic persistent scheduling — [include/ops/thread/util/util.cuh:237-290](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/thread/util/util.cuh#L237-L290).
  - `pdl::arrive/wait` wrap `griddepcontrol.launch_dependents/wait` — [util.cuh:304-321](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/thread/util/util.cuh#L304-L321).
- **LCF/LCSF layout contract.** A kernel provides a `layout` struct with `globals`, `input_block`, and optional `output_block`, `scratch_block`, `finish_block`, `common_state`, `producer_state` and `consumer_state` types — [prototype/common/templates.cuh:19-26,65-107](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/common/templates.cuh#L19-L107).
  - Missing types default to `empty`.
  - Knobs are read via `FLAG_GETTER` concept-detected constexprs with defaults: NUM_CONSUMER_WARPS=8, NUM_PRODUCER_WARPS=4, INPUT_PIPE_STAGES=1, OUTPUT_PIPE_STAGES=1, MAX_SHARED_MEMORY=MAX/NUM_BLOCKS-512, CLUSTER_BLOCKS=1 — [templates.cuh:32-58](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/common/templates.cuh#L32-L58).
- **LCSF control flow.** The kernel requires `common_setup`, `producer::{setup,load,store}` and `consumer::{setup,compute,finish}` — [prototype/lcsf/lcsf.cuh:11-22](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/lcsf/lcsf.cuh#L11-L22).
  - The template owns the smem ring allocation, all semaphore init, and the phase-bit bookkeeping — [lcsf.cuh:60-132](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/lcsf/lcsf.cuh#L60-L132).
  - The producer loop polls stores and loads with `test_wait` + `__nanosleep(10)` — [lcsf.cuh:134-181](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/lcsf/lcsf.cuh#L134-L181).
  - The consumer loop runs wait → compute → advance rings, then `finish` — [lcsf.cuh:182-219](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/lcsf/lcsf.cuh#L182-L219).
  - Persistence is a `for(task_iter...; true;)` loop that exits when the user's `common_setup` sets `num_iters = -1` — [lcsf.cuh:137-145](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/lcsf/lcsf.cuh#L137-L145).
- **README example.** The README's H100 GEMM (~855 TFLOPs claimed) is written in the LCF template style. The user code calls `warpgroup::decrease_registers<40>` and `increase_registers<232>` itself in `producer::setup` / `consumer::setup` — [README.md:47-140](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/README.md#L47-L140).
- **Template usage in kernels.** Only 18 of 69 `.cu` files under `kernels/` reference `prototype` at HEAD (local `grep -rl prototype kernels --include=*.cu`). These are mostly H100 kernels (bf16/fp8 h100 GEMM, mha_h100_lcf, ring_attn, gemm_ar/rs, mamba2, rotary, fftconv) plus `parallel/ag_gemm*_b200`. The B200 GEMMs (bf16/fp8/mxfp8/nvfp4/int8 b200, nvfp4 b300/vr200) do not use the templates.
- **Hand-written B200 body.** `kernels/gemm/bf16_b200/bf16_b200_gemm.cu` has one `kernel()` with explicit role branches [bf16_b200_gemm.cu:117-252](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/bf16_b200/bf16_b200_gemm.cu#L117-L252):
  - The producer warpgroup does `decrease_registers<56>`.
  - Warp 3 + `elect_leader` is the TMA loader. Warp 2 is the CLC scheduler. Warps 0-1 on cta_rank 0 issue the tcgen05 MMAs (`mm2_ABt`/`mma2_ABt` + `tcgen05::commit<CLUSTER_SIZE>`).
  - The consumer warpgroups do the epilogue (`increase_registers<224>` only when not overlapping MMA and epilogue). They copy TMEM → registers (`warpgroup::load_async` from `tt` subtiles), then registers → smem, then TMA store.
  - All pipeline depths are template parameters of `config<...>` — [bf16_b200_gemm.cu:6-34](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/bf16_b200/bf16_b200_gemm.cu#L6-L34).
- **Educational Blackwell series.** It builds the same kernel in 9 levels — [kernels/gemm/educational_b200/README.md:1-17](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/educational_b200/README.md):
  - plain loop: 6 TFLOPs
  - WMMA: 26
  - TMA: 55
  - tcgen05: 293
  - pipelined warp specialization (TMA loader + MMA issuer): 731
  - epilogue pipelining: 1050
  - 2-CTA cluster: 1285
  - the full kernel: 1540 TFLOPs at M=N=K=4096

### Inferences
- In 2026, the TK "abstraction" that matters for peak-performance Blackwell kernels is the primitive layer: typed tiles, group-scoped ops, semaphore/phasebit helpers, TMA/tcgen05/CLC/PDL wrappers. The callback-template layer (LCF/LCSF) is legacy, Hopper-era scaffolding. Kernel authors now hand-write role branches, semaphore counts and ring indices per kernel.
- Composition is all compile-time. Tile shapes, pipe depths, warp counts and roles are `static constexpr` in a `config` struct, and every combination is a separate instantiation (`run_benchmark<config<256,256,64,8,false,4,8>>`, [bf16_b200_gemm.cu:368-378](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/bf16_b200/bf16_b200_gemm.cu#L368-L378)).

### Gaps
- I did not read every vector/shared-op header line by line (`rv`/`sv` layouts, complex tiles). These notes cover the type definitions and kernel-facing APIs only.

## Q2. Is there a VM / interpreter / megakernel framework? Instruction format, dispatch, scheduling, sync, smem paging, op registration

### Takeaway
There were two. The full **"Kittens Virtual Machine" (`prototype/vm/`)**, which is the megakernel runtime behind the May 2025 "No Bubbles" Llama-1B work, was **deleted from main in the TK 2.0 commit (`e463ed89`, 2026-01-11)**, along with `demos/kvm-runner` and `tests/vm`. The **simpler `prototype/interpreter/`** survives on main (439 lines), but no kernel on main uses it. The deleted VM's design:
- 128-byte int instructions (`int[32]`), opcode in word 0.
- Per-SM instruction streams indexed by `%smid`.
- A 2-stage instruction pipeline.
- Fixed warp roles: controller, loader, storer, MMA launcher, plus 16 consumer warps.
- Linear `if/else` opcode dispatch over a variadic op pack.
- 13 × 16 KB smem "pages" whose logical→physical order is passed from instruction to instruction by each op's `release_lid` hook.
- mbarrier-based page/instruction semaphores.

### Cited Findings — deleted `prototype/vm/` (as of OLD = `50ee1f0a`)
- **Deleted files.** TK 2.0 deleted `prototype/vm/{config,consumer,launcher,loader,noop,storer,templates,util,vm}.cuh` and `prototype/vm/controller/{controller,instruction_fetch,page_allocator,semaphore_constructor,timings_store}.cuh`, plus `prototype/interpreter/{Makefile,test.cu}`, `demos/kvm-runner/*` and `tests/vm/*` (local `git show e463ed89 --name-status`). Commit totals: 732 files, +11,543 / −53,639 — [commit e463ed89](https://github.com/HazyResearch/ThunderKittens/commit/e463ed89).
- **History.** VM history runs from "hello kvm" (`206f7b42`, 2025-04-09) through a burst of perf commits in May 2025: "2 stage pipe is a bit faster", "slightly faster page barriers", "use sm id", "gating all timing behind a constexpr flag", "*this* is the fixed shared paging". The last change was "Port multigpu into megakernel prototype" (`ac9d8d4f`, 2025-06-03) (local `git log -- prototype/vm`).
- **Default config** [OLD prototype/vm/config.cuh:11-52](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/config.cuh#L11-L52):
  - `INSTRUCTION_PIPELINE_STAGES = 2`.
  - `INSTRUCTION_WIDTH = 32` ("128 bytes per instruction"), `using instruction_t = int[32]`.
  - `TIMING_WIDTH = 128` and `DYNAMIC_SEMAPHORES = 32`.
  - `NUM_CONSUMER_WARPS = 16; NUM_WARPS = 4 + 16` ("One controller warp, one load warp, one store warp, and one mma warp").
  - `PAGE_SIZE = 16384`, with `NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE` and `static_assert(NUM_PAGES == 13)`.
  - `SCRATCH_BYTES = 1024`.
  - `CONSUMER_REGISTERS = 104; NON_CONSUMER_REGISTERS = 64`.
  - `TIMING_RECORD_ENABLED = false`.
  - `GMEM_SPIN_LOOP_SLEEP_NANOS = 20`, declared as `bool`, a latent bug.
  - Instructions are a `gl<int, 1, -1, -1, INSTRUCTION_WIDTH>`, i.e. [worker][instr_idx][32].
- **Per-slot smem state.** `instruction_state_t` holds the instruction words, timings, `pid_order[NUM_PAGES]` (logical→physical page map), 32 dynamic semaphores and 1 KB scratch. Two of these form a ring — [OLD prototype/vm/util.cuh:15-22](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/util.cuh#L15-L22).
- **Worker id.** `get_worker_id()` returns `%smid`, not `blockIdx.x` (the latter is commented out), so each physical SM reads its own instruction row. `vm_clock()` reads `%globaltimer` — [OLD util.cuh:25-40](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/util.cuh#L25-L40).
- **Dispatch.** `dispatch_op<op_dispatcher, op, ops...>::run(opcode, ...)` is a compile-time recursive `if(opcode == op::opcode) ... else <rest>`, i.e. a linear compare chain. The base case is `asm volatile("trap;")` — [OLD util.cuh:43-58](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/util.cuh#L43-L58).
- **Op registration.** An op is a struct with `static constexpr int opcode` and nested `controller { release_lid(); init_semaphores(); }`, `loader::run`, `launcher::run`, `consumer::run` and `storer::run`. The op list is a variadic template pack: `kvm<config, globals, ops...>`. `NoOp` (opcode 0) is always prepended — [OLD prototype/vm/noop.cuh:10-54](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/noop.cuh#L10-L54); [OLD prototype/vm/vm.cuh:153-166](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/vm.cuh#L153-L166).
- **Worker main loops.** `MAKE_WORKER(name, ...)` generates each role's `main_loop`: `for each instruction: kvms.await_instruction(); dispatch_op<name_dispatcher>(opcode); kvms.next_instruction()`. `next_instruction` does `__syncwarp` + lane-0 `arrive(instruction_finished[ring])` — [OLD util.cuh:126-140,257-298](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/util.cuh#L126-L298).
- **Role assignment** [OLD vm.cuh:112-135](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/vm.cuh#L112-L135):
  - Warps `< NUM_CONSUMER_WARPS` run `increase_registers<CONSUMER_REGISTERS>` and the consumer loop.
  - The last warpgroup runs `decrease_registers<NON_CONSUMER_REGISTERS>` and switches on `warpgroup::warpid()`: 0 = loader, 1 = storer, 2 = launcher (MMA), 3 = controller, default = `trap`.
  - The kernel is `__launch_bounds__(NUM_THREADS, 1) __cluster_dims__(CLUSTER_BLOCKS)` with `__grid_constant__ globals` — [OLD vm.cuh:161-166](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/vm.cuh#L161-L166).
- **Controller warp, per instruction** [OLD prototype/vm/controller/controller.cuh:16-134](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/controller/controller.cuh#L16-L134):
  - (0) If the ring slot was used, wait `instruction_finished[ring]` and `invalidate_semaphore` the previous instruction's dynamic semaphores (lines 30-48).
  - (1) `load_instructions`: each lane `< 32` loads one int from global (plain LD, not TMA) (lines 54-55; [instruction_fetch.cuh:12-28](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/controller/instruction_fetch.cuh#L12-L28)).
  - (2) Page allocation: for lane `< NUM_PAGES`, call the *previous* op's `controller::release_lid(g, prev_instruction, lane)` and set `pid_order[lane] = prev.pid_order[lid]` (lines 61-79).
  - (3) Lane 0 calls the op's `controller::init_semaphores`, which returns a count and is followed by `fence.proxy.async.shared::cta`. The count is broadcast via `__shfl_sync` (lines 85-101; [semaphore_constructor.cuh:12-22](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/controller/semaphore_constructor.cuh#L12-L22)).
  - (4) `arrive(instruction_arrived[ring])` (line 106).
- **Smem paging.** Pages are `page<config>{ int data[PAGE_SIZE/4]; }` with `as_st<T>()` reinterpret helpers (fp8 128×128 or float 64×64) — [OLD util.cuh:60-76](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/util.cuh#L60-L76).
  - `pid(lid)` reads the slot's `pid_order` with a raw `lds` from a cached shared address — [OLD util.cuh:148-152](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/util.cuh#L148-L152).
  - Each page has `INSTRUCTION_PIPELINE_STAGES_BITS` semaphores. `wait_page_ready(pid)` waits on each bit of `instruction_index`. `finish_page(pid, count)` arrives on all of them — [OLD util.cuh:145-171](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/util.cuh#L145-L171).
  - At init, page semaphore *i* gets count `NUM_CONSUMER_WARPS * (1<<i)` and is immediately arrived, so all pages start "free" — [OLD vm.cuh:84-90](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/vm.cuh#L84-L90).
  - The page array lives at the first 1024-aligned address of dynamic smem — [OLD vm.cuh:35-37](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/vm.cuh#L35-L37).
  - The NoOp loader releases every page at once (`wait_page_ready` then `finish_page(pid, NUM_CONSUMER_WARPS)`) — [OLD noop.cuh:24-33](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/noop.cuh#L24-L33).
- **Static sync objects** [OLD vm.cuh:27-34,80-97](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/vm.cuh#L27-L97):
  - `instruction_arrived[2]` (count 1) and `instruction_finished[2]` (count `NUM_WARPS-1`).
  - `page_finished[NUM_PAGES][bits]`.
  - On Blackwell, one `tensor_finished` (TMEM handoff, initialized then arrived so it starts available).
  - `semaphores_ready`.
- **Profiling.** Fixed event slots (`TEVENT_CONTROLLER_START=0 ... TEVENT_IFETCH_DONE, PAGE_ALLOC_DONE, SEMS_SETUP, LOADER/LAUNCHER/STORER/CONSUMER_START`, gmem wait/store events, first/last load/use/store), recorded as `globaltimer` deltas into the per-instruction timing array. Recording is gated by `if constexpr(config::TIMING_RECORD_ENABLED)` — [OLD util.cuh:186-242](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/util.cuh#L186-L242).
- **LCSF-style args inside the VM.** The VM `templates.cuh` defined `persistent_state` and LCSF-style `*_args` structs, a carryover from the interpreter design — [OLD prototype/vm/templates.cuh:9-163](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/templates.cuh).

### Cited Findings — surviving `prototype/interpreter/` (HEAD)
- **Fixed geometry.** `NUM_PRODUCER_WARPS=4, NUM_CONSUMER_WARPS=8`, one block — [prototype/interpreter/interpreter.cuh:11-14](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/interpreter/interpreter.cuh#L11-L14).
- **Ops are whole LCSF templates.** Each op must satisfy the same `kernel_template` concept (layout + producer/consumer callbacks). `run_op_producer<Op>` and `run_op_consumer<Op>` re-run the LCSF pipeline for one instruction and re-carve smem per op with `shared_allocator` — [interpreter.cuh:16-29,31-164](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/interpreter/interpreter.cuh#L16-L164).
- **Cross-op smem hazard.** The "danger_stage" logic makes the producer wait on the previous op's `finish_finished` before overwriting smem that overlaps the previous op's finish block (`max_finish_offset`) — [interpreter.cuh:68-69,91](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/interpreter/interpreter.cuh#L68-L91).
- **Dispatch.** `dispatch_producer` / `dispatch_consumer` use the same linear recursive `if(opcode == op::opcode)` chain. Here the base case is a silent no-op, not a trap — [interpreter.cuh:290-310](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/interpreter/interpreter.cuh#L290-L310).
- **Instruction fetch is TMA-style bulk copy.** Lane 0 issues `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` into a double-buffered `instructions[2][cols]` smem array, keyed by `blockIdx.x` — [interpreter.cuh:312-327,346](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/interpreter/interpreter.cuh#L312-L346).
- **Main loop.** Consumers run `increase_registers<232>`, wait `instruction_arrived[i%2]`, stop on opcode 0, dispatch, and arrive `instruction_finished` — [interpreter.cuh:383-405](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/interpreter/interpreter.cuh#L383-L405). Producers run `decrease_registers<40>`, and producer warp 2 prefetches the next instruction ("warp #2 doesn't get enough love") — [interpreter.cuh:406-428](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/interpreter/interpreter.cuh#L406-L428).
- **Usage.** Nothing on main includes it except the aggregate header [prototype/prototype.cuh:14](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/prototype.cuh#L14). Its test (`prototype/interpreter/test.cu`) was deleted in TK 2.0.

### Cited Findings — branches
- **Remote megakernel branches, all stale (last commits 2025).** Local `git log -1 origin/<b>`:
  - `origin/megakernels` (`cce72c2f`, 2025-09-21; still contains `prototype/vm`)
  - `bvm-llama-unity` (2025-05-27)
  - `bvm-unity-pipeline` (2025-05-18; contains `demos/kvm-runner/kvm_runner/{instructions,scheduler,python_vm}.py` and a `kvm_batch_runner`)
  - `bvm-separate-globals` (2025-04-27)
  - `gang-kernel-examples` (2025-05-13)
- **README.** The only megakernel reference is a blog link: "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B, May 2025" — [README.md:390](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/README.md#L390).

### Inferences
- **Where the megakernel work went.** The VM runtime was moved out of the TK library, most likely into the separate HazyResearch Megakernels repo (covered by another researcher). TK main keeps only primitives the VM needed: `smid()` helper (`bd9dcce0`, 2026-03-28), `invalidate_semaphore`, `elect_leader_from_active` (`af0623cf`), PDL/CLC.
- **Dispatch cost.** VM dispatch was a linear opcode compare chain repeated in each of the five roles. Every op's code for each role is inlined into one giant kernel, so binary size grows as (#ops × #roles). The base-case `trap` in the VM adds divergent trap paths. That matches the megabake memory note about C7519 / divergent traps; this is my connection, not something the TK repo states.
- **No ring-buffer allocator.** The VM's page scheme is a *permutation* passed between consecutive instructions. Each op says, via `release_lid(query)`, which of its logical pages are freed earliest, and the next instruction's logical page *i* maps to that physical page. This lets the next op's loads start on pages the previous op releases first, without a general allocator.

### Gaps
- The deleted `timings_store.cuh`, `loader/storer/launcher/consumer.cuh` (6-line `MAKE_WORKER` invocations) and the kvm-runner Python scheduler and instruction encodings were not read in detail. The Megakernels repo is the canonical successor and is out of scope.
- No commit message explains *why* the VM was removed. The TK 2.0 commit body is just "ThunderKittens 2.0 (#161)". The README says only that the repo structure changed and that Python packaging and Ampere support were dropped — [README.md:26-30](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/README.md#L26-L30).

## Q3. What changed recently (2025-2026), deletions and simplifications, latest commit and branches

### Takeaway
2026 has been about breadth and hardening of primitives, not new runtime frameworks:
- TK 2.0 (Jan 11, 2026) deleted the VM, the kvm-runner, Python packaging and many demos, and flattened the include tree.
- Feb-Mar 2026: Blackwell GEMMs gained CLC/PDL scheduling, an educational series, and NVRTC/JIT readiness.
- Apr-May 2026: INT8, integer types and SM120.
- Jul-Sep 2026: B300/SM103 NVFP4, Vera Rubin (SM107), and mbarrier memory-semantics fixes.

Latest commit: 2026-09-12.

### Cited Findings
- **HEAD** is `67845f5f`, 2026-09-12 — [commit](https://github.com/HazyResearch/ThunderKittens/commit/67845f5f). Recent ones:
  - "Add SM107 support and Vera Rubin FP8/NVFP4 GEMMs" (`f6e16dcc`, 2026-09-10)
  - "Add FP8-output and FP4-output Vera Rubin GEMMs" (`5e4d4a5e`)
  - The README lists Sept 10, 2026 as Vera Rubin support — [README.md:32](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/README.md#L32).
- **TK 2.0** (`e463ed89`, 2026-01-11) — [commit](https://github.com/HazyResearch/ThunderKittens/commit/e463ed89):
  - 313 files deleted, 45 added, 84 modified, ~300 renamed.
  - Deleted: `prototype/vm/*`, `demos/kvm-runner/*`, `tests/vm/*`, `config.py`, `thunderkittens.cpp` (the Python package), many `kernels/attn/demo/*` decode schedulers (gqa/mha/mla_decode, `scheduler_v2.py`), `include/common/debug.cuh`, `include/ops/device/device.cuh`, `include/types/shared/st_descriptor.cuh` (replaced by `types/shared/descriptor.cuh`).
  - Flattened: `ops/group/mma/warpgroup/base/*` → `ops/group/mma/base/*`; `mma/tensor/tensor.cuh` → `mma/tcgen05.cuh`; `memory/util/tma.cuh` → `util/tma.cuh`; `types/device/*` → `types/system/*` (pgl, ipc).
  - Added: `ops/thread/util/sync.cuh` and `ops/thread/memory/tile/shared_to_tensor.cuh`.
  - Source: local `git show e463ed89 --name-status`.
- **README TK 2.0 notes.** "full support for Blackwell GPUs along with MXFP8 and NVFP4"; "We no longer support the repo as a Python package ... Kernels under /kernels must now be compiled individually"; "no longer actively support Ampere" — [README.md:26-30](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/README.md#L26-L30).
- **Pre-TK2 2025 work.**
  - "Add load-compute-store-communicate template" (`173c6e8c`, 2025-10-18) created `prototype/lcsc/` for multi-GPU work.
  - "Add inter-SM/inter-GPU synchronization primitives" (`e1cedaad`).
  - "Release ParallelKittens kernels" (`b12a9150`, 2025-11-07).
  - "Only enable cluster_dims on blackwell for 2-CTA matmul; on Hopper this is unneeded and kills perf (up to -100 TFLOPs)" (`909d64be`, 2025-10-09).
  - Source: local `git log`.
- **Jan-Feb 2026 Blackwell GEMM tuning** (Stuart Sul). Local `git log`:
  - "Add PDL utilities" / "Use PDL" (`a494eeb0`, `2c699997`, 2026-01-26)
  - "Add grouped CLC / PDL utils" (`db5959b2`)
  - "Separate TMEM provisioning" (`e3d0b200`)
  - "Use dynamic grid size"
  - "Remove NUM_BLOCKS_PER_SM parameter" (`42226f05`)
  - "Use 2 producer warpgroups" (`2b8ca709`)
  - "Pipeline tcgen05.cp and tcgen05.mma within same thread" (`9f6483f4`)
  - "Remove unnecessary fences" (`0907814c`)
  - "Prefetch TMA descriptors" (`7c60e091`)
  - "Allow 5-stage pipeline"
  - "Add epi overlap parameter"
- **"Remove register allocation"** (`261018c6`, 2026-01-29) deleted `PRODUCER_REGISTERS = 256; CONSUMER_REGISTERS = 256` and both `warpgroup::increase_registers` calls from the NVFP4 B200 GEMM — [commit 261018c6](https://github.com/HazyResearch/ThunderKittens/commit/261018c6).
- **NVRTC / JIT readiness (Mar 2026).**
  - "Add KITTENS_NO_HOST flag for JIT compilation use case" (`d6553cbd`, 2026-03-17), which guards host code in [include/kittens.cuh:33,54](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/kittens.cuh#L33-L54).
  - "Access static members via type to avoid nvrtc errors" (`2b86a196`, `00ea8a24`).
  - "Fix alignment for NVRTC compilation on CUDA 13" (`0230013a`, 2026-08-18).
  - "Remove deprecated dsmem unit tests" (`ddc75b55`, 2026-03-16).
  - "Add kittens::smid() device helper" (`bd9dcce0`, 2026-03-28).
- **Datatypes and arches (Apr-Jul 2026).**
  - INT8/UINT8 across RT/ST/TMA/tcgen05/wgmma (`25cf0da1`..`c4bd6b4c`, 2026-05-07..09).
  - "Add initial support for SM120" (`d60dc325`).
  - "Rename GPU/GPU_TARGET build variable to ARCH with SM-based values" (`9681df68`).
  - "Use XOR shuffles for register tile reductions (#198)" (`34b15f7e`, 2026-05-28).
  - "Add SM103 NVFP4 K96 tcgen05 support and B300 GEMM sample (#201)" (`1c3920d9`, 2026-07-12).
- **Memory-semantics fixes (Aug-Sep 2026).**
  - "Support relaxed semantics for mbarrier operations" (`e6a5b870`)
  - "Support all memory semantics for multimem instructions" (`2656c389`)
  - "Use generic fence" (`72f14517`)
  - "Fix compiler ordering around mbarrier waits" (`b62aa152`, 2026-09-10): "Add memory clobbers to wait, careful_wait, and test_wait so NVCC cannot move ordinary memory accesses across their acquire operations" — [commit b62aa152](https://github.com/HazyResearch/ThunderKittens/commit/b62aa152).
- **External write-ups.** README links a "ThunderKittens 2.0: Even Faster Kernels for Your GPUs, Feb 2026" blog and a Together AI Vera Rubin NVL72 post (Sept 2026) — [README.md:394-395](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/README.md#L394-L395).

### Inferences
- I found no 2026 "Kittens VM 2" or "retire the abstractions" code in this repo. The "retiring" visible here is threefold: (a) removing the VM/kvm-runner/interpreter test from TK main in TK 2.0; (b) new flagship Blackwell kernels not using LCF/LCSF; (c) dropping `setmaxnreg` from some Blackwell GEMMs. Any 2026 VM redesign would live in the Megakernels repo or the blog, not here.

### Gaps
- I did not fetch the TK 2.0 blog post (Feb 2026) or the Hamza Elshafie deep-dive, so their stated rationale for the changes is unverified here.
- Some 2026 history may be squashed into PR merges.

## Q4. Pipelining (smem rings, barriers, warp specialization) and register allocation

### Takeaway
Pipelines are N-stage smem rings, each stage with an `*_arrived` / `*_finished` mbarrier pair. Phase parity is tracked in a single `uint32_t` bitfield, with arrived bits in the low 16 and finished bits in the high 16, initialized to `0xFFFF0000`. The ring index advances with a compile-time modulo. Warp specialization is a hand-coded branch on `warpgroup::groupid()` / `warpid()` with `elect_leader()` single-thread issuers. Register reallocation uses `setmaxnreg` wrappers with explicit template counts. Blackwell kernels use it less: the accumulator lives in TMEM, and one NVFP4 kernel dropped it entirely.

### Cited Findings
- **Phase bits and rings.** `get_phasebit<half>(bitfield, ring)` reads bit `ring` (half 0) or `ring+16` (half 1), and `update_phasebit` XORs it. `ring_advance<N>(r) = (r+d)%N`. A bad `half` hits `brkpt` — [include/common/util.cuh:431-453](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/common/util.cuh#L431-L453). The initializer comment reads "***_finished phase bits start as 1s, ***_arrived phase bits start as 0s" (`0xFFFF0000`) — [lcsf.cuh:116](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/lcsf/lcsf.cuh#L116); [bf16_b200_gemm.cu:91](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/bf16_b200/bf16_b200_gemm.cu#L91).
- **B200 GEMM pipelines** [bf16_b200_gemm.cu:88-114](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/bf16_b200/bf16_b200_gemm.cu#L88-L114):
  - A load ring (`inputs_arrived/finished[LOAD_PIPE_DEPTH]`).
  - An MMA→epilogue TMEM double buffer (`outputs_arrived[NUM_CONSUMERS]`, `outputs_finished[MMA_PIPE_DEPTH]`, where `MMA_PIPE_DEPTH = OVERLAP_MMA_EPI ? 2 : 1`).
  - A CLC schedule ring (`schedule_arrived/finished[CLC_PIPE_DEPTH]`).
  - TMEM lifecycle semaphores (`tmem_provisioned`, `tmem_finished`).
  - All initialized by `threadIdx.x == 32` with explicit arrival counts, e.g. `schedule_finished` count `(2+NUM_CONSUMERS)*CLUSTER_SIZE+NUM_CONSUMERS`.
- **TMA expect-bytes and 2-CTA multicast.** The MMA issuer calls `tma::expect_bytes(inputs_arrived[...], bytes/NUM_CONSUMERS)` before waiting. The loader uses `tma::cluster::load_async(..., mask=1<<cta_rank)` — [bf16_b200_gemm.cu:126-134,170-177](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/bf16_b200/bf16_b200_gemm.cu#L126-L177).
- **MMA completion is signalled by hardware.** `mm2_ABt(d_tt, a, b, inputs_finished[ring])` passes the stage's `finished` semaphore into the MMA, so the tensor core frees the smem stage on completion. `detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived[...])` signals the epilogue — [bf16_b200_gemm.cu:173-178](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/bf16_b200/bf16_b200_gemm.cu#L173-L178).
- **Epilogue pipelining.** The epilogue double-buffers smem D tiles with `tma::store_async_read_wait<NUM_D_TILES-1>()` before overwriting. It releases TMEM early (`outputs_finished` arrive right after `tensor_load_wait()`) so the next tile's MMA can start — [bf16_b200_gemm.cu:209-242](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/bf16_b200/bf16_b200_gemm.cu#L209-L242).
- **LCSF pipelining.** The producer polls output-store and input-load readiness in one loop with `test_wait` and `__nanosleep(10)` — [lcsf.cuh:163-178](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/lcsf/lcsf.cuh#L163-L178). `SAFE_STAGES_BETWEEN_BLOCKS` bounds how many next-task loads may start before the previous task's finish block (which aliases the tail of smem) is released — [lcsf.cuh:66-71,150-162](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/lcsf/lcsf.cuh#L66-L162).
- **`setmaxnreg` wrappers.** `increase_registers<n>()` / `decrease_registers<n>()` emit `setmaxnreg.{inc,dec}.sync.aligned.u32` and `static_assert(n % 8 == 0)`. Presets: `producer_registers()` = 24, and `consumer_registers<NCWG>()` = `480/NCWG - 8*(NCWG>3) - 224*(NCWG==1)` — [group.cuh:52-61](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/group/group.cuh#L52-L61).
- **Register splits observed:**
  - README H100 GEMM: producer 40 / consumer 232 — [README.md:107,122](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/README.md#L107-L122).
  - Interpreter: 40 / 232 — [interpreter.cuh:384,407](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/interpreter/interpreter.cuh#L384-L407).
  - B200 bf16: 56 / 224, the consumer bump only when `!OVERLAP_MMA_EPI` — [bf16_b200_gemm.cu:118,185-186](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/bf16_b200/bf16_b200_gemm.cu#L118-L186).
  - Deleted VM: 64 non-consumer / 104 consumer across 20 warps — [OLD config.cuh:50-51](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/config.cuh#L50-L51).
  - NVFP4 B200: removed (`261018c6`).

### Inferences
- **Megakernel register budget.** The VM's 104-register consumer budget, versus 224-232 in single-op GEMMs, shows the cost of keeping 16 consumer warps resident for all ops in one megakernel. Every op's consumer code must fit the smallest common register budget.
- **One-thread issuers.** TK 2.0 kernels drive TMA, CLC and MMA from single elected threads in otherwise idle warps (`warp::elect_leader()`). The rest of the producer warpgroup stays idle at low register counts.

### Gaps
- I did not verify whether the H100 kernels still use `producer_registers()` / `consumer_registers()` presets or explicit counts everywhere.

## Q5. Performance-critical design choices visible in the code

### Takeaway
TK pushes everything to compile time: tile shapes, pipe depths, warp roles, op lists and timing flags. It uses opaque-but-thin PTX wrappers with explicit memory clobbers. It avoids runtime polymorphism except for the (now removed) opcode compare chains. It aligns smem to 1024 B for swizzle/TMA. It uses `bra.uni` spin loops and elected single-thread issue to keep control flow warp-uniform.

### Cited Findings
- **Compile-time configuration.**
  - Prototype knobs are `constexpr` via concept detection (`FLAG_GETTER`) — [prototype/common/templates.cuh:32-58](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/common/templates.cuh#L32-L58).
  - The B200 GEMM exposes every tuning parameter as a template argument with `static_assert` bounds — [bf16_b200_gemm.cu:6-34](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/bf16_b200/bf16_b200_gemm.cu#L6-L34).
  - Smem budgets are `static_assert`ed at compile time — [bf16_b200_gemm.cu:52-57](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/bf16_b200/bf16_b200_gemm.cu#L52-L57); [lcsf.cuh:46-51](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/lcsf/lcsf.cuh#L46-L51).
- **Timing compiled out.** VM timing was compiled out with `if constexpr(config::TIMING_RECORD_ENABLED)` (commit "gating all timing behind a constexpr flag", `0cf89492`, 2025-05-03) — [OLD util.cuh:188-194](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/util.cuh#L188-L194).
- **Uniform spin loop.** `wait()` is `mbarrier.try_wait.parity` in a `bra.uni` loop with a `"memory"` clobber. Pre-Hopper adds `nanosleep 5` "to save instruction issue slots" — [sync.cuh:106-140](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/thread/util/sync.cuh#L106-L140). Missing clobbers on some wait variants were a real compiler-reordering bug fixed in Sept 2026 (`b62aa152`).
- **Smem alignment and swizzle.** Smem is aligned to 1024 B by default on SM90+ — [util.cuh:251](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/common/util.cuh#L251); [OLD vm.cuh:36](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/vm.cuh#L36). Swizzle is XOR-based in `st::idx` — [st.cuh:108-129](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/types/shared/st.cuh#L108-L129).
- **Cluster dims cost.** Commit `909d64be` (2025-10-09) found that enabling `cluster_dims` on Hopper when it isn't needed "kills perf (up to -100 TFLOPs)". It is now enabled only for Blackwell 2-CTA.
- **Opcode dispatch.** Dispatch is a compile-time-unrolled linear `if/else` chain (VM base case `trap`, interpreter base case no-op). There is no jump table or function pointer, and every role inlines every op's body — [OLD util.cuh:43-58](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/util.cuh#L43-L58); [interpreter.cuh:290-310](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/interpreter/interpreter.cuh#L290-L310).
- **Persistent-kernel scheduling.**
  - Old: static `task_iter*gridDim.x + blockIdx.x` — [prototype/common/util.cuh:8-10](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/common/util.cuh#L8-L10).
  - New on Blackwell: dynamic Cluster Launch Control work stealing — [include/ops/thread/util/util.cuh:237-290](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/include/ops/thread/util/util.cuh#L237-L290).
  - Supergroup tile swizzling via `get_swizzled_2d_idx<SUPERGROUP_SIZE>` is used for L2 reuse — [bf16_b200_gemm.cu:122,138](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/bf16_b200/bf16_b200_gemm.cu#L122-L138).
- **Inter-kernel overlap.** PDL hides launch gaps: `pdl::wait()` before the first load, and `warpgroup::pdl::arrive()` on the last tile — [bf16_b200_gemm.cu:123,217,233](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/kernels/gemm/bf16_b200/bf16_b200_gemm.cu#L123-L233). PDL arrival was moved earlier in Jan 2026 ("Arrive PDL earlier", `dd65359e`).
- **Instruction-cache awareness (only indirect).** The VM loader for instructions used a per-lane plain LD of 32 ints (one 128-byte line), while the interpreter used a bulk async copy — [OLD instruction_fetch.cuh:12-28](https://github.com/HazyResearch/ThunderKittens/blob/50ee1f0a/prototype/vm/controller/instruction_fetch.cuh#L12-L28); [interpreter.cuh:312-327](https://github.com/HazyResearch/ThunderKittens/blob/67845f5f/prototype/interpreter/interpreter.cuh#L312-L327). I found no explicit I-cache or code-size handling in TK code or comments.

### Inferences
- **Lessons for megabake.**
  - (1) TK's own production path in 2026 hand-writes one warp-specialized body per kernel rather than going through a generic op-dispatch framework.
  - (2) Where TK did build a megakernel VM, it used fixed role warps, a 2-deep instruction ring, a fixed count of equal-size smem pages, and per-op compile-time hooks (`release_lid`, `init_semaphores`, per-role `run`).
  - (3) The linear opcode chain and per-role inlining mean binary size and register pressure are bounded by the union of all ops, which is consistent with the low 104-register consumer budget.
- **Phase-bit approach.** The single 32-bit phase bitfield plus compile-time ring modulo is a cheap way to keep pipeline state in registers and avoid smem traffic.

### Gaps
- I found no comments or commits in TK quantifying I-cache misses or code size for the VM. Any such data would be in the Megakernels repo or blog posts, which are out of scope here.
