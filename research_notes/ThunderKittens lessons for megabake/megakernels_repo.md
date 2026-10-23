# HazyResearch Megakernels repo: architecture notes (kernel bodies, instruction IR, scheduler)

Repo: `/workspace/megabake/agent_space/Megakernels` (call it `MK/`). Unless a branch is named, citations refer to `main` at `7309cec` (2025-06-02). Citations to other branches use the form `origin/<branch>:path:line`. For mk-v2 I extracted the files with `git archive origin/mk-v2` and read them there; the line numbers are file-relative.

Branch and date map, from `git log --all`:
- `main`: last commit 7309cec on 2025-06-02 ("Update README.md"). It holds the low-latency Llama-1B demo, the Python scheduler for both settings, and the throughput Python code, but not the throughput CUDA code.
- `origin/throughput`: last commit 2025-09-28 (B. Spector, "initial copy-paste" plus a merge). It adds `demos/cross-gpu-llama/*`, `demos/thunder-attention/mla/*`, `demos/workstealing_example/workstealing.cu`, a global-work-queue option and multi-GPU `pgl` instruction layouts. It is 105 files and +21k lines over main.
- `origin/mk-v2`: last commit 2026-04-26 (Stuart Sul). This is **"MegaKittens"**, a rewrite with a new `csrc/` and a `megakittens/` Python package that traces PyTorch graphs into a single persistent kernel. It targets Blackwell only.
- `origin/mk-v2-gptfast` (2026-04-29), `origin/mk-v2-llama-ablations` (2026-04-26: "no barrier option", "no-virtual-smem", "compile per layer"), and **`origin/mk-v2-llama` (2026-05-26, nashbrown, "Llama1b cleanup"). The latest commit in the repo is on mk-v2-llama.** That branch ports the Llama-1B ops to `csrc/itypes/llama1b/*` and adds Llama-70B itypes plus "manual kernels" ablations.

---

## Q1. How an instruction is defined (opcode, argument layout, size), how op bodies are written (worker roles), and how a new op is added

### Takeaway
On main, an instruction is a fixed 32×int32 (128 B) row: word 0 is the opcode and words 1–31 are op-specific integers serialized from a Python dataclass. An op is a C++ struct with a static `opcode` and five nested structs (`controller`, `loader`, `launcher`, `consumer`, `storer`). Each warp role runs its own nested struct, and dispatch is a compile-time if-chain over a variadic op list. MegaKittens (mk-v2) widens the instruction to a 256 B struct. The new fields are explicit src/dst tensor ids, 16 indices, and per-instruction global-barrier lists. Its dispatch switch is JIT-generated.

### Cited Findings
**Instruction format (main)**
- The config sets `INSTRUCTION_WIDTH = 32; // 128 bytes per instruction` with `instruction_t = int[32]`. It also sets `TIMING_WIDTH = 128` ints per instruction, `DYNAMIC_SEMAPHORES = 32` and `INSTRUCTION_PIPELINE_STAGES = 2`. — [include/config.cuh:9-22](MK/include/config.cuh)
- The instruction tensor is `gl<int, 1, -1, -1, 32>`, indexed `[worker_id(=SM id), instruction_index, word]`. Timings use the same shape with 128 ints. — [include/config.cuh:53-56](MK/include/config.cuh); [include/controller/instruction_fetch.cuh:16-26](MK/include/controller/instruction_fetch.cuh)
- In Python, `Instruction.serialize()` emits `[opcode] + fields`. Each int field is one word. A list or tuple field is emitted as `len` followed by its elements, and `None` becomes 0. A field named `global_idx` is skipped. — [megakernels/instructions.py:83-119](MK/megakernels/instructions.py)
- `serialize_and_pad` zero-pads each instruction to `INTS_PER_INSTRUCTION = 32`, with `assert num_padding >= 0`. — [megakernels/scheduler.py:13,274-278](MK/megakernels/scheduler.py)
- On the device, each op parses words by position in its own `parsed_instruction` constructor.
  - `MatVecAddOp` reads `layer=instr[1]`, `start_block_idx=[2]`, `end_block_idx=[3]` and `reduction_block_idx=[4]`. — [demos/low-latency-llama/matvec_adds.cu:21-38](MK/demos/low-latency-llama/matvec_adds.cu)
  - `rms_upgate_silu` reads `iters = 2*instruction[2]` (the list length) and `block_idxs = instruction + 3`. This matches the Python len-prefixed list encoding of `LayerNormDoubleMatVecSiLU.block_idxs`. — [demos/low-latency-llama/upgate.cu:21-26](MK/demos/low-latency-llama/upgate.cu); [megakernels/demos/latency/instructions.py:135-157](MK/megakernels/demos/latency/instructions.py)
- Latency opcodes:
  - 1 `RMS_QKV_MatVecRopeAppend`
  - 2 `PartialAttention`
  - 3 `AttentionReduction`
  - 4 `O_ProjResidual`
  - 5 `RMS_DoubleMatVecSiLU`
  - 6 `DownProjResidual`
  - 7 `RMS_LM_Head`
  
  Model dimensions are hard-coded macros: `LLAMA_1B_HIDDEN_DIM 2048`, SM counts 132 (H100) and 148 (B200), and so on. — [demos/low-latency-llama/llama.cuh:7-26](MK/demos/low-latency-llama/llama.cuh)
- Opcode 0 is a `NoOp` that the wrapper prepends automatically (`mk_internal<config, globals, NoOp<config>, ops...>`), "to ensure that the VM can support zeros". Its loader releases all pages immediately. — [include/megakernel.cuh:158-164](MK/include/megakernel.cuh); [include/noop.cuh:8-54](MK/include/noop.cuh)

**Dispatch**
- `dispatch_op<dispatcher, ops...>::run(opcode, ...)` is a recursive variadic template. It compiles to a linear `if (opcode == op::opcode) ... else ...` chain and hits `trap` on an unknown opcode. — [include/util.cuh:32-55](MK/include/util.cuh)
- `MAKE_WORKER(name, start_event, is_consumer)` generates each role's `main_loop`. The loop runs `for each instruction: await_instruction(); record(start); dispatch_op(... op::name::run(g, mks)); record(end); next_instruction()`. The loader, consumer, launcher and storer headers are one-line instantiations of this macro. — [include/util.cuh:260-304](MK/include/util.cuh); [include/consumer.cuh:6](MK/include/consumer.cuh); [include/loader.cuh:7](MK/include/loader.cuh)
- The kernel is bound to Python with `kittens::py::bind_kernel<mk<default_config, llama_1b_globals, attention_partial_op, ..., rms_lm_head_op>>(m, "mk_llama", &globals::Bar, &globals::instructions, &globals::timings, ...weights..., ...)`. The op set is fixed at compile time. — [demos/low-latency-llama/llama.cu:15-53](MK/demos/low-latency-llama/llama.cu)

**Warp roles (main)**
- There are 16 consumer warps plus 4 others, so 20 warps or 640 threads per block. The block uses `__launch_bounds__(NUM_THREADS, 1)` and one block per SM (`grid = sm_count`). — [include/config.cuh:25-28](MK/include/config.cuh); [include/megakernel.cuh:166-171](MK/include/megakernel.cuh); [demos/low-latency-llama/llama.cuh:144-146](MK/demos/low-latency-llama/llama.cuh)
- Role assignment:
  - Warps 0–15 are consumers, set to `increase_registers<104>`.
  - The last warpgroup is set to `decrease_registers<64>`. Its warp 0 is the loader, warp 1 the storer, warp 2 the launcher and warp 3 the controller.
  
  The config comment says "One controller warp, one load warp, one store warp, and one mma warp". — [include/megakernel.cuh:118-140](MK/include/megakernel.cuh); [include/config.cuh:24,50-51](MK/include/config.cuh)
- Required op interface, as exemplified by NoOp and every op:
  - `controller::release_lid(g, instruction, query)` returns the logical page id.
  - `controller::init_semaphores(g, s)` returns the number of semaphores it initialized.
  - `loader::run`, `launcher::run`, `consumer::run` and `storer::run`.
  
  — [include/noop.cuh:11-53](MK/include/noop.cuh); [demos/low-latency-llama/matvec_adds.cu:89-178](MK/demos/low-latency-llama/matvec_adds.cu)
- Role usage in practice for the latency matvec ops:
  - The loader's lane 0 issues TMA weight loads.
  - The other loader lanes release pages the op does not use.
  - The consumers compute partial matvecs into scratch.
  - The storer reduces the 16 per-warp partials, does the epilogue, issues the TMA store, then runs `atomicAdd` on a global barrier.
  - The launcher does nothing except, on Blackwell, `wait_tensor_ready()` / `arrive(tensor_finished)` to cycle TMEM ownership.
  
  — [demos/low-latency-llama/matvec_pipeline.cuh:117-243,293-301](MK/demos/low-latency-llama/matvec_pipeline.cuh)
- Ops are factored as shared "pipelines" with op-specific hooks. `matvec_pipeline<Config, Globals, parsed_instruction, pipeline_specifics>` calls `pipeline_specifics::load_iter`, `::store` and `::gmem_wait`. `rms_matvec_pipeline` adds an RMSNorm prologue through non-type template params `ActPtr` / `RmsPtr` (pointer-to-member into globals). `MatVecAddOp<EXPECTED_ARRIVAL_COUNT, WeightsPtr, InPtr, OutPtr, opcode, prev_opcode>` is instantiated as both `downproj` and `o_proj`. — [demos/low-latency-llama/matvec_pipeline.cuh:5-12,246-349](MK/demos/low-latency-llama/matvec_pipeline.cuh); [demos/low-latency-llama/matvec_adds.cu:10-19,181-192](MK/demos/low-latency-llama/matvec_adds.cu)
- Per-op pipelines use the 32 dynamic `kittens::semaphore`s that live in the instruction slot. The matvec pipeline uses 3 input stages × 4 pages (16×512 bf16 tiles), with page 0 for activations and RMS scale. Its semaphore count is `SEM_COUNT = 1 + (3+3)*2`, and per-warp output partials go in the 4 KB scratch. — [demos/low-latency-llama/matvec_pipeline.cuh:8-26,38-57,104-115](MK/demos/low-latency-llama/matvec_pipeline.cuh); [include/util.cuh:11-19](MK/include/util.cuh)
- `attention_partial` (Flash-decode per KV head, GQA 4) uses page 0 for Q/O/L and page 1 for K/V stages (NUM_STAGES=3). The launcher warp issues the K/V TMA loads, consumer warp 0 does the compute, and the loader only frees pages 2..12. Most tiles are 16-row with "only 4 rows are used", which wastes 75% of the MMA. — [demos/low-latency-llama/attention_partial.cu:6-37,297-385](MK/demos/low-latency-llama/attention_partial.cu)

**Adding an op (main)**
- There is no registry. To add an op you:
  1. Pick an opcode `#define` in `llama.cuh`.
  2. Write a struct with `opcode` plus the five nested role structs, and add it to the `mk<...>` template list in `llama.cu`.
  3. Add fields to the `globals_t` struct and to the `bind_kernel` member list, then rebuild with `make`.
  4. Add a Python `Instruction` dataclass with `opcode()`, `prev_opcode()`, `cost()` and fields in serialization order.
  5. Add a PyVM solver for diff testing.
  
  — [demos/low-latency-llama/llama.cuh:7-13,51-147](MK/demos/low-latency-llama/llama.cuh); [demos/low-latency-llama/llama.cu:26-53](MK/demos/low-latency-llama/llama.cu); [megakernels/demos/latency/instructions.py:32-196](MK/megakernels/demos/latency/instructions.py); [megakernels/dispatch.py:21-42](MK/megakernels/dispatch.py)

**mk-v2 / MegaKittens instruction format**
- `struct instruction_t` has `static_assert(sizeof == 256)`. It contains:
  - `int icode`
  - `uint8_t src_tensors[16]`, `dst_tensors[8]`
  - `int indices[16]`
  - `uint32_t src_barriers[16]`, `int src_barrier_targets[16]`
  - 4 uint8 counts (`num_src_input/reuse_barriers`, `num_dst_input/reuse_barriers`)
  - `uint32_t dst_barriers[8]`
  
  The Python mirror is a pydantic model that says it "Must match `struct instruction_t` in csrc/schema.cuh". — [origin/mk-v2:csrc/schema.cuh:25-44](MK/csrc/schema.cuh); [origin/mk-v2:megakittens/schema/instruction.py:17-50](MK/megakittens/schema/instruction.py)
- An ITYPE is enforced by a C++20 concept requiring nested `controller/loader/launcher/consumer/storer`. Dispatch uses `if constexpr` on a `WorkerType` enum (`page_manager, semaphore_manager, loader, launcher, consumer, storer`). — [origin/mk-v2:csrc/schema.cuh:7-23](MK/csrc/schema.cuh); [origin/mk-v2:csrc/utils.cuh:8-17](MK/csrc/utils.cuh)
- The `icode → op` switch is JIT-generated per graph: `case {icode}: return dispatch_instruction<{op}, worker_type, T>(args...);`. Each unique `(itype, src_tensors, dst_tensors)` gets a fresh icode, and tensor indices are compile-time template params. — [origin/mk-v2:megakittens/dispatcher.py:321-347](MK/megakittens/dispatcher.py); [origin/mk-v2:megakittens/scheduler.py:345-352](MK/megakittens/scheduler.py); [origin/mk-v2:.claude/CLAUDE.md "Architecture notes"](MK/.claude/CLAUDE.md)
- Adding an op in mk-v2 takes two files: `csrc/itypes/<name>.cuh` and a Python `IType` subclass. The subclass defines a torch custom op, a fake op, `inputs`/`outputs` TensorSpecs, `block_indices`, `num_instructions` and `validate`, and is auto-discovered through `IType.__subclasses__()`. The CLAUDE.md also lists rules:
  - "Every page claimed must be released".
  - "`init_semaphores` must return the count ... Getting this incorrect results in undefined behaviors".
  - "`page_finish()` ... must be called by only one thread".
  
  — [origin/mk-v2:.claude/CLAUDE.md "Adding a new instruction type", "Rules for writing a CUDA instruction type"](MK/.claude/CLAUDE.md)
- mk-v2 config:
  - 8 consumer warps plus 4 others (12 warps).
  - Registers 224 for consumers and 56 for the rest.
  - `CLUSTER_SIZE = 2`.
  - `PAGE_SIZE = 32768`, described as "the only knob and everything else is derived". `NUM_PAGES` is computed (7 on Blackwell per CLAUDE.md).
  - Scratch is whatever smem is left.
  
  — [origin/mk-v2:csrc/schema.cuh:46-74](MK/csrc/schema.cuh)
- mk-v2 role order differs from main: warpgroup warp 0 is the controller, 1 the loader, 2 the launcher and 3 the storer. — [origin/mk-v2:csrc/megakittens.cuh:54-75](MK/csrc/megakittens.cuh)

### Inferences
- Positional int packing with hand-matched Python field order and C++ `instruction[k]` reads is brittle, because nothing cross-checks the two sides. mk-v2 fixes this by moving to a typed struct with a pydantic mirror and generic `indices[]` / tensor-id / barrier fields, so ops no longer encode their own dependency metadata.
- The 32-word cap limits list-valued args. For example, upgate's `block_idxs` holds at most 29 entries, which is fine for 512 blocks over 132 SMs (~4 each).

### Gaps
- I did not read every mk-v2 itype body (attention.cuh is 721 lines), only gemm.cuh paging and barrier usage.

---

## Q2. How a model is lowered into instructions, and how instructions are assigned to SMs (host side)

### Takeaway
Lowering is hand-written per model and per setting in Python. A `ScheduleBuilder.make_dag()` emits per-layer `DAG_Node(instruction, dependencies)` lists with hand-picked work partitions (for example one QKV instruction per SM). An `assign_to_sms(mode)` pass maps the topologically ordered list onto per-SM queues using `rr`, `zz`, `wave`, `dag` (greedy list scheduling on a FLOP/byte cost model) or `pool`. The default is round-robin. Queues are NoOp-padded to equal length and tensorized once, then reused for every decode step. In mk-v2, lowering comes instead from a traced torch graph (aot_autograd): each node emits one instruction per output tile, and instructions are dealt round-robin across SMs in waves.

### Cited Findings
**Main: DAG construction (latency)**
- `ScheduleBuilder.build(model)` calls `make_globals(model)` and then `make_dag(globs)` and returns a `Schedule(globs, dag_nodes, end_node)`. `get_linear_instructions()` "assumes this is in topological order". — [megakernels/scheduler.py:47-91](MK/megakernels/scheduler.py)
- Per layer, the latency `make_dag_layer` creates:
  - QKV: one instruction per SM, each a contiguous range of 16-row output blocks (`blocks_per_sm = num_qkv_blocks / sm_count`).
  - PartialAttention: one per KV head (8), with deps on the exact QKV nodes that produced that head's K and V blocks.
  - O-proj: 128 instructions of one 16-row block each, all depending on all partial nodes.
  - Up/gate: one per SM with *strided* block indices `range(sm_idx, num_blocks, sm_count)`.
  - Down-proj: jobs over (4 column splits of 2048) × 128 blocks, sliced so each SM gets a contiguous run within one column split.
  - LM head: one per SM.
  
  — [megakernels/demos/latency/scheduler.py:118-232,270-386](MK/megakernels/demos/latency/scheduler.py)
- `make_dag_layer` hard-codes `assert globs.skip_attn_reduction` and `num_attention_partitions = 1`. The `AttentionReduction` op (opcode 3) is therefore never scheduled in the latency demo. `pick_num_attention_partitions` exists ("TODO limitation until we have a better reduction tree", capped at 24), but `make_dag_layer` does not use it. — [megakernels/demos/latency/scheduler.py:21-35,276-277](MK/megakernels/demos/latency/scheduler.py)
- Block sizes are fixed at 16 for all matvecs and KV blocks, with `matvec_reduction_size=2048` and `attn_reduction_size=4`. — [megakernels/demos/latency/scheduler.py:103-111](MK/megakernels/demos/latency/scheduler.py)
- Cost model: cost is output elements × hidden size, for example `(end-start)*qkv_block_size*hidden_size`. Up/gate is ×2. PartialAttention costs `seq_len/num_partials*head_dim*2`. — [megakernels/demos/latency/instructions.py:53-58,76-81,126-131,151-157](MK/megakernels/demos/latency/instructions.py)

**Main: SM assignment**
- `round_robin_assign_to_sms`: instruction i goes to SM i % sm_count. — [megakernels/scheduler.py:154-161](MK/megakernels/scheduler.py)
- `zig_zag_assign_to_sms`: the order alternates forward and backward across SMs every `sm_count` instructions. — [megakernels/scheduler.py:164-175](MK/megakernels/scheduler.py)
- `wave_assign_to_sms`: consecutive same-opcode instructions are grouped into "waves". Within each wave, instructions are sorted largest-cost first and assigned to the least-loaded SM through a min-heap of accumulated cost. — [megakernels/scheduler.py:178-217](MK/megakernels/scheduler.py)
- `assign_dag_to_sms`: greedy list scheduling. Ready nodes come off a max-cost-first heap and go to the SM with the earliest finish time.
  - It sets `start_time = sm_time`. The dependency-aware `max(ready_time, sm_time)` is commented out.
  - The critical-path priority (`calc_priority`) is defined but its call is commented out.
  
  — [megakernels/scheduler.py:41-46,94-151 (esp. 104, 133-134)](MK/megakernels/scheduler.py)
- `pool_assign_to_sms` splits instructions by `tags()["pool"]` (memory vs compute) and round-robins each set over a `memory_fraction` partition of SMs. It is used by the throughput setting, whose instructions are tagged `MemoryInstruction` / `ComputeInstruction`. — [megakernels/scheduler.py:220-242](MK/megakernels/scheduler.py); [megakernels/demos/throughput/instructions.py:43-54](MK/megakernels/demos/throughput/instructions.py)
- `generate.py` defaults to `sched: str = "rr"` and `setting: str = "latency"`. It builds the schedule once, assigns it, and tensorizes it. — [megakernels/scripts/generate.py:47-48,139-144](MK/megakernels/scripts/generate.py)
- `tensorize_instructions` pads every queue to `max_queue_len` with `NoOp()`. It builds an `int32 [num_sms, max_len, 32]` instruction tensor and a zeroed `[num_sms, max_len, 128]` timings tensor. — [megakernels/scheduler.py:281-308](MK/megakernels/scheduler.py)
- On the device, a block finds its queue through `get_worker_id()`, which returns `%smid` (`// return blockIdx.x;` is commented out). This assumes exactly one resident block per SM. — [include/util.cuh:21-30](MK/include/util.cuh)
- The schedule is static across decode steps. Per token, `MK_Generator.run` writes the embedding into `hidden_states`, sets `pos_id` and calls `barriers.fill_(barrier_fill_val)` before launching. Attention partitioning over sequence length is recomputed on the device from `g.pos_id`. — [megakernels/generators.py:115-143](MK/megakernels/generators.py); [demos/low-latency-llama/attention_partial.cu:340-348](MK/demos/low-latency-llama/attention_partial.cu)

**Throughput setting (main Python)**
- Throughput ops are 1 PreAttnLayerNorm, 2 QKV_MatMulRopeAppend, 3 AttentionDecode, 4 O_ProjResidual, 5 PreMLP_Norm, 6 GateSilu, 7 UpMatMul, 8 DownProjResidual, 9 PreLMHeadRMS and 10 LM_Head. Each uses `batch_start_idx` / `output_block_idx` tiles with 128×128 matmul blocks. — [megakernels/demos/throughput/instructions.py:57-223](MK/megakernels/demos/throughput/instructions.py); [megakernels/demos/throughput/scheduler.py:40-42](MK/megakernels/demos/throughput/scheduler.py)
- The throughput DAG is effectively empty of edges. Every node is created as `DAG_Node(i, prev_layer_outputs)`, but `make_dag_layer` returns `(new_nodes, [])`, so `prev_layer_outputs` is always `[]`. There is also a TODO: "these dependencies are broken bc the lm head norm and matmul is split into two instructions". Ordering and correctness rely on the linear order plus device-side barriers. — [megakernels/demos/throughput/scheduler.py:317-322,339-398](MK/megakernels/demos/throughput/scheduler.py)
- Throughput instructions define no `cost()` methods, so the `dag` and `wave` modes cannot be used with them. — [megakernels/demos/throughput/instructions.py:57-223](MK/megakernels/demos/throughput/instructions.py)
- On main, the throughput `mk_dir` points to `tests/batch-vm/llama_official`, which does not exist on main. The script also asserts `batch_size == 1024`: "must recompile the kernel with new BATCH_SIZE". — [megakernels/scripts/generate.py:59-75](MK/megakernels/scripts/generate.py)

**mk-v2 lowering and scheduling**
- `megakittens.compile()` traces a torch function. aot_autograd flattens inputs, and the DAG nodes are `IType`s mapped from torch ops through `torch_functions_map`. — [origin/mk-v2:README.md](MK/README.md); [origin/mk-v2:.claude/CLAUDE.md "Architecture notes"](MK/.claude/CLAUDE.md)
- `schedule(dag)` runs four phases:
  1. Count instructions per node, with cluster-aligned offsets ("CTA pairs must get same instruction type").
  2. Assign tensors with memory reuse. A buffer is reused only if the last consumer is at least `sm_count` instructions earlier; otherwise the scheduler emits a "release barrier".
  3. Assign fine-grained barriers by tile-region overlap.
  4. Generate the flat instruction list, padded with noops to a multiple of the cluster size.
  
  — [origin/mk-v2:megakittens/scheduler.py:26-59,62-151,154-288,291-418](MK/megakittens/scheduler.py)
- SM assignment in mk-v2 static mode is pure round-robin in waves: `wave = i // sm_count; sm = i % sm_count`, packed as `[max_per_sm_insts, sm_count, 64]`. The device indexes `g.instructions[{iter, blockIdx.x, 0}]`. — [origin/mk-v2:megakittens/dispatcher.py:79-90](MK/megakittens/dispatcher.py); [origin/mk-v2:csrc/controller.cuh:50-58](MK/csrc/controller.cuh)
- mk-v2 also has `GLOBAL_WORK_QUEUE` mode, off by default. Each block takes the next instruction index from Blackwell Cluster Launch Control (`kittens::clc::schedule/query`), with grid = number of instructions, so this is dynamic work stealing. — [origin/mk-v2:csrc/schema.cuh:47](MK/csrc/schema.cuh); [origin/mk-v2:csrc/controller.cuh:27-49](MK/csrc/controller.cuh); [origin/mk-v2:megakittens/dispatcher.py:379-382](MK/megakittens/dispatcher.py)
- The throughput branch had already added a config switch `ENABLE_GLOBAL_WORK_QUEUE` ("explicit SM scheduling or work stealing?") with `GLOBAL_WORK_QUEUE_PARTITIONS`. It also added multi-GPU `pgl` instruction layouts and an `instruction[0] != -1` stop signal. — [origin/throughput:include/config.cuh (diff vs main)](MK/include/config.cuh); [origin/throughput:include/controller/instruction_fetch.cuh](MK/include/controller/instruction_fetch.cuh); [origin/throughput:demos/workstealing_example/workstealing.cu](MK/demos/workstealing_example/workstealing.cu)

### Inferences
- The "DAG" scheduler on main is a cost-balanced list scheduler that ignores dependency timing. Because round-robin is the default, the latency demo's real schedule is "one instruction per SM per op, in op order". This works because most ops are partitioned as exactly `sm_count` instructions.
- Partitioning, the choice of how much work goes into each instruction, carries most of the scheduling intelligence, and it is hand-coded per op. Examples are contiguous vs strided block lists and down-proj jobs cut at column-split boundaries.

### Gaps
- I did not inspect the 1,076-line `megakernels/demos/tp_throughput/scheduler.py` on the throughput branch, so its assignment policy is unknown. Its file list includes `test_cpp_scheduling.py`, which suggests a C++ scheduler port.
- I found no benchmark numbers comparing the rr, zz, wave, dag and pool modes in the repo.

---

## Q3. How dependencies are encoded, waited on and signalled

### Takeaway
Cross-SM dependencies use global-memory int counters: `Bar[layer, opcode-1, idx]` on main, and a flat `barriers[]` with per-instruction (id, target) lists in mk-v2. A producer increments the counter with `atomicAdd` after a full `tma::store_async_wait()`. A consumer spin-waits on a `volatile` load with `__nanosleep` until the count reaches an expected value. On main, the expected counts and barrier indices are hard-coded inside each op's C++ body, and the host zeroes the counters before every launch. mk-v2 computes barrier ids and targets automatically from tile-region overlap and embeds them in the instruction. It also adds "reuse" barriers for buffer write-after-read hazards.

### Cited Findings
**Main**
- Barriers are an `int32 [num_layers, 10 ("more than the number of opcodes"), num_heads + 2*kv_heads = 48]` tensor. On the C++ side they are `gl<uint, 1, -1, -1, 48>`. — [megakernels/demos/latency/scheduler.py:53-61](MK/megakernels/demos/latency/scheduler.py); [demos/low-latency-llama/llama.cuh:103-108](MK/demos/low-latency-llama/llama.cuh)
- Wait pattern: `while (*(volatile int*)&g.Bar[{layer, prev_opcode-1, idx}] < EXPECTED_ARRIVAL_COUNT) __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);`. It runs on one thread (consumer warp 0 lane 0), followed by `group<16>::sync`. — [demos/low-latency-llama/matvec_adds.cu:123-140](MK/demos/low-latency-llama/matvec_adds.cu)
- Signal pattern: `tma::store_async_wait(); // not just read wait! full wait! must be visible in global!` then `atomicAdd(&g.Bar[{layer, opcode-1, 0}], inst.iters)`. There is no explicit fence: `// asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.` — [demos/low-latency-llama/matvec_adds.cu:165-176](MK/demos/low-latency-llama/matvec_adds.cu)
- Granularity varies by op:
  - QKV signals `Bar[layer, 0, block_idx/4]`, which is per head because 4×16 = 64 = head_dim.
  - Attention waits for count 4 on each of its 4 Q heads and on its K and V head slots.
  - Upgate signals per 2048-wide chunk `Bar[layer, 4, block_idx*16/2048]`, which lets down-proj wait only on its own `reduction_block_idx` column.
  - O-proj waits for 32 arrivals (`num_attention_heads`).
  - The next layer's QKV, and the LM head, wait for `Bar[layer-1, DownProj-1, 0] >= 512`, which is 4 col splits × 128 blocks.
  
  — [demos/low-latency-llama/rms_matvec_rope_append.cu:52-58,161-162](MK/demos/low-latency-llama/rms_matvec_rope_append.cu); [demos/low-latency-llama/attention_partial.cu:314-326,396-405](MK/demos/low-latency-llama/attention_partial.cu); [demos/low-latency-llama/upgate.cu:15-16,117-120](MK/demos/low-latency-llama/upgate.cu); [demos/low-latency-llama/matvec_adds.cu:129-131,181-192](MK/demos/low-latency-llama/matvec_adds.cu); [demos/low-latency-llama/rms_lm_head.cu:14,32-36](MK/demos/low-latency-llama/rms_lm_head.cu)
- Fine-grained overlap trick: in `attention_partial`, only the *last* KV block of the last partial waits on the barrier for the newly appended K/V. Earlier KV-cache blocks stream in before the current token's QKV op has finished. — [demos/low-latency-llama/attention_partial.cu:357-383](MK/demos/low-latency-llama/attention_partial.cu)
- Residual adds use a TMA reduce-add store (`tma::store_add_async` into `hidden_states`), so several SMs accumulate into one residual vector without a separate reduction op. — [demos/low-latency-llama/matvec_adds.cu:75-80](MK/demos/low-latency-llama/matvec_adds.cu)
- `skip_attn_reduction` mode: partial attention writes `attn_out` directly and bumps `Bar[layer, AttentionReduction-1, 0]`, impersonating the reduction op's barrier. — [demos/low-latency-llama/attention_partial.cu:609-667](MK/demos/low-latency-llama/attention_partial.cu)
- Python DAG dependencies are only used for host-side ordering and assignment. The device never sees them, because barrier slots and expected counts are duplicated by hand in C++. — [megakernels/demos/latency/scheduler.py:287-336](MK/megakernels/demos/latency/scheduler.py); [demos/low-latency-llama/matvec_adds.cu:181-192](MK/demos/low-latency-llama/matvec_adds.cu)
- Counters are monotonic within a launch. The host resets them each step with `barriers.fill_(barrier_fill_val)`, which is an extra torch kernel per token. — [megakernels/generators.py:115-116,132](MK/megakernels/generators.py)

**mk-v2**
- `barrier_wait` spins on `ld.relaxed.gpu.global.u32` until `barrier_val == target` (the comment says it "should not spin-loop with acquire") and then issues `fence.acquire.gpu`. `barrier_arrive` is `red.relaxed.gpu.global.add.u32`. Both carry a "TODO: change scope to `sys` for multi-gpu". — [origin/mk-v2:csrc/utils.cuh:26-39](MK/csrc/utils.cuh)
- Each instruction carries `src_barriers[i]` / `src_barrier_targets[i]`. The first `num_src_input_barriers` are data dependencies (RAW) and the next `num_src_reuse_barriers` are buffer-reuse (WAR). `dst_barriers` follow the same split. Helpers `all_input_barrier_wait`, `all_reuse_barrier_wait` and `all_barrier_arrive` loop over them. — [origin/mk-v2:csrc/utils.cuh:42-69](MK/csrc/utils.cuh); [origin/mk-v2:csrc/schema.cuh:32-42](MK/csrc/schema.cuh)
- In gemm, the loader waits on input barriers before issuing TMA. The consumer leader waits on reuse barriers before writing output. The consumer releases its page and calls `all_barrier_arrive` at the end. — [origin/mk-v2:csrc/itypes/gemm.cuh:73,149,165-166](MK/csrc/itypes/gemm.cuh)
- Barrier synthesis works on unit regions. It computes a GCD unit region per edge across all producer and consumer tile boxes, maps each unit to producer instruction indices, and creates **one barrier per unique (producer node, set of producer instructions)**. The target is the size of that set. Reuse barriers target `num_consumer_insts`. Per-instruction limits are 16 src and 8 dst barriers, and exceeding them raises an error. — [origin/mk-v2:megakittens/scheduler.py:186-288](MK/megakittens/scheduler.py)
- The dispatcher zeroes `barrier_tensor` before each launch. — [origin/mk-v2:megakittens/dispatcher.py:365-367](MK/megakittens/dispatcher.py)
- `origin/mk-v2-llama-ablations` has a commit "Add no barrier option", meaning barriers can be disabled for ablation. — `git log origin/mk-v2-llama-ablations` (873dc3b, 2026-04-25)

### Inferences
- The main-branch design, with hard-coded counts like `512`, `4` and `num_attention_heads` in op bodies, couples kernels to one model shape and one partitioning. mk-v2's automatic barrier derivation decouples them, at the cost of host-side region analysis and a 256 B instruction.
- Equality spin (`!= target`) in mk-v2 versus `< expected` on main means mk-v2 counters must hit an exact value. Over-arrival would deadlock, so the targets must be exact.

### Gaps
- No memory-model justification is given for omitting a release fence between `store_async_wait` and `atomicAdd` on main. The code comment itself expresses doubt.

---

## Q4. The device interpreter: controller loop, shared-memory paging, next-instruction overlap, timing

### Takeaway
A single controller warp runs ahead of the workers by up to `INSTRUCTION_PIPELINE_STAGES = 2` instructions. For each slot it:
1. waits for the instruction that last used the slot to finish, and invalidates that instruction's semaphores;
2. fetches the next 128 B instruction from global memory;
3. computes the new instruction's physical page order by asking the *previous* op which logical pages it frees first (`release_lid`);
4. calls the new op's `init_semaphores`;
5. arrives on `instruction_arrived`.

Shared memory is 13 × 16 KB pages, each with a per-page mbarrier. Pages pass from instruction N to N+1 as soon as N's warps `finish_page`, so N+1's loader starts prefetching while N is still computing. Timing is an optional per-instruction 128-slot `clock64` record, bulk-copied to global memory by the controller.

### Cited Findings
**Controller loop (main)**
- The steps:
  - **Step 0:** if the slot has been used before, `wait(instruction_finished[ring])`, then lanes `< num_semaphores` call `invalidate_semaphore` (with a `// TODO needed?` before the warp sync), and the slot's timings are stored.
  - **Step 1:** `load_instructions`, where 32 lanes each copy one int.
  - **Step 2:** page order.
  - **Step 3:** lane 0 calls `init_semaphores` through dispatch and broadcasts the count via `__shfl_sync`.
  - **Step 4:** `arrive(instruction_arrived[ring])`.
  
  — [include/controller/controller.cuh:24-132](MK/include/controller/controller.cuh)
- `instruction_finished` is initialised with count `NUM_WARPS - 1` (every non-controller warp arrives) and `instruction_arrived` with count 1. — [include/megakernel.cuh:82-86](MK/include/megakernel.cuh)
- Workers `await_instruction()` with parity `(instruction_index / STAGES) & 1`. In `next_instruction()`, lane 0 arrives on `instruction_finished` and the ring advances. — [include/util.cuh:122-140](MK/include/util.cuh)
- Dead or alternate code: `instruction_fetch_loop`, `page_allocator_loop` and `semaphore_constructor_loop` are separate-warp versions of steps 1–3, but nothing calls them. `instruction_fetch_loop` even calls `load_instructions` with 4 args against a 3-arg signature, so it only compiles because it is an uninstantiated template. `semaphore_constructor_loop` has `static_assert(STAGES == 2, "Need to be changed.")`. — [include/controller/instruction_fetch.cuh:10-51](MK/include/controller/instruction_fetch.cuh); [include/controller/page_allocator.cuh:21-71](MK/include/controller/page_allocator.cuh); [include/controller/semaphore_constructor.cuh:22-81](MK/include/controller/semaphore_constructor.cuh)

**Paging (main)**
- `PAGE_SIZE = 16384` gives `NUM_PAGES = (MAX_SMEM - STATIC)/16384`, with `static_assert(NUM_PAGES == 13)`. Static smem per stage holds a 4 KB scratch, the instruction, the timings and 32 semaphores. — [include/config.cuh:32-44](MK/include/config.cuh)
- `page_finished[pid][STAGES_BITS]` semaphores are initialised with count `NUM_CONSUMER_WARPS` and pre-arrived, so every page starts free. — [include/megakernel.cuh:87-93](MK/include/megakernel.cuh)
- `wait_page_ready(pid)` waits on parity `instruction_index` bit. `finish_page(pid, count)` arrives `count` times, so 16 arrivals are needed per page per instruction. Consumers usually arrive 1 per warp, and the loader arrives `NUM_CONSUMER_WARPS` at once for pages it does not use. — [include/util.cuh:155-174](MK/include/util.cuh); [include/noop.cuh:27-31](MK/include/noop.cuh)
- Logical-to-physical mapping: `new.pid_order[lane] = prev.pid_order[prev_op::release_lid(prev_instr, lane)]`. `release_lid` gives the order in which the previous instruction frees its logical pages, so the new op's logical page 0 is mapped to the first physical page the previous op frees. Ops read `s.pid(lid)` with a `lds` from the cached shared address. — [include/controller/controller.cuh:74-99](MK/include/controller/controller.cuh); [include/util.cuh:150-154](MK/include/util.cuh)
- `release_lid` is a hand-coded 13-entry permutation per op. Examples:
  - matvec: special cases for `iters == 1`, `iters == 2` and `iters % 3`, commented "NOTE: assumes a three stage pipeline".
  - attention_partial: `{2..12, 0, 1}`.
  - attention_reduction: `{1..12, 0}`.
  
  — [demos/low-latency-llama/matvec_pipeline.cuh:70-102](MK/demos/low-latency-llama/matvec_pipeline.cuh); [demos/low-latency-llama/attention_partial.cu:278-281](MK/demos/low-latency-llama/attention_partial.cu); [demos/low-latency-llama/attention_reduction.cu:138-141](MK/demos/low-latency-llama/attention_reduction.cu)
- Early release: the matvec loader's lanes `>= needed_pages` release unused pages immediately. Consumers release weight pages during the last `INPUT_PIPELINE_STAGES` iterations, and the activation page right after the RMS norm. — [demos/low-latency-llama/matvec_pipeline.cuh:155-159,201-207,344-345](MK/demos/low-latency-llama/matvec_pipeline.cuh)
- mk-v2 keeps the same scheme with one page semaphore per page (count 1, `page_wait` parity `iter&1`, single-thread `page_finish`). gemm's `lid_release_order` has three variants keyed on `num_iters % LOAD_PIPE_DEPTH`, with `static_assert(NUM_PAGES == 7 && LOAD_PIPE_DEPTH == 4)`. — [origin/mk-v2:csrc/schema.cuh:142-147](MK/csrc/schema.cuh); [origin/mk-v2:csrc/megakittens.cuh:36-48](MK/csrc/megakittens.cuh); [origin/mk-v2:csrc/itypes/gemm.cuh:37-45](MK/csrc/itypes/gemm.cuh)

**Overlap of the next instruction with the current one**
- With 2 instruction slots, the controller prepares instruction N+1 (fetch, page order, semaphores) while workers run N. Because each worker role loops independently, the loader can start N+1's TMA loads into pages that N has already freed while N's consumers or storer are still working. — [include/controller/controller.cuh:24-132](MK/include/controller/controller.cuh); [include/util.cuh:260-304](MK/include/util.cuh)
- Blackwell TMEM is handed off similarly. The launcher waits on `tensor_finished` (parity `instruction_index % 2`), and NoOp and matvec ops just pass it through. — [include/util.cuh:176-181](MK/include/util.cuh); [include/noop.cuh:37-42](MK/include/noop.cuh)
- In mk-v2, the controller broadcasts `instruction_arrived` across the CTA pair with `tensor_commit<CLUSTER_SIZE>`, commented "hack: use tcgen05.commit for mbarrier broadcast". Workers stop when `icode == -1`. The kernel starts with `kittens::pdl::wait()` (programmatic dependent launch). — [origin/mk-v2:csrc/controller.cuh:45,54,89](MK/csrc/controller.cuh); [origin/mk-v2:csrc/workers.cuh:7-21](MK/csrc/workers.cuh); [origin/mk-v2:csrc/megakittens.cuh:51](MK/csrc/megakittens.cuh)

**Timing and profiling (main)**
- `state::record(event_id)` writes `clock64() - start_clock` into `timing()[event_id]` only `if constexpr (TIMING_RECORD_ENABLED)`, which defaults to false. — [include/util.cuh:190-196](MK/include/util.cuh); [include/config.cuh:46](MK/include/config.cuh)
- Fixed event slots:
  - controller start, ifetch, page alloc, sems setup and end: 0–4
  - loader, launcher and storer start: 5, 7, 9
  - consumer per-warp: from 11
  - gmem wait / store: 44–47
  - first/last load, use and store: 48–53
  - output ready: 54
  - op-defined events: from 55
  
  The demo defines extra op slots (`ATOMIC_ADD_START`, `WEIGHT_WAIT_*`, `RMS_*`). — [include/util.cuh:214-246](MK/include/util.cuh); [demos/low-latency-llama/llama.cuh:28-44](MK/demos/low-latency-llama/llama.cuh)
- When the controller recycles a slot, it bulk-copies the 512 B timing row to `g.timings[smid, instruction_index]` with `cp.async.bulk.global.shared::cta` and then zeroes it. Blackwell uses `st.bulk.weak` to zero. — [include/controller/timings_store.cuh:10-46](MK/include/controller/timings_store.cuh); [include/controller/controller.cuh:51-58](MK/include/controller/controller.cuh)
- The throughput branch adds `megakernels/timings.py`, `scripts/plot_timings.py`, `scripts/debug_stall.py` and an HTML profiler at `util/profiler/index.html` (2,187 lines). — [origin/throughput diffstat vs main](MK/)

### Inferences
- Paging is a software-managed, double-buffered smem ownership protocol. The hard part is that each op must know, at controller time, the *previous* op's release order. That order often depends on the iteration count modulo the pipeline depth, which is why `release_lid` tables are hand-maintained and asserted against `NUM_PAGES` and pipeline depth.
- Step 2 runs one warp-wide dispatch per lane (one lane per page), and steps 1–3 are serialised in one warp. The main-branch per-instruction controller overhead is therefore a global fetch, a dispatch chain, semaphore init and an mbarrier arrive.

### Gaps
- There is no measured controller overhead in the repo on main.

---

## Q5. Low-latency vs throughput variants, and the newer "VM" / MegaKittens code

### Takeaway
Main ships only the low-latency Llama-1B (batch 1, matvec) CUDA demo and the shared VM headers. The throughput variant (batched 128×128 matmuls, Llama-8B/70B-style ops, memory/compute SM pools, multi-GPU) lives on `origin/throughput` (Sept 2025). The newest work (Apr–May 2026) is MegaKittens on the `mk-v2*` branches. It is a Blackwell-only, torch-compile-style runtime with NVRTC JIT, auto-derived barriers, 2-CTA clusters, CLC work stealing, and ported Llama-1B and Llama-70B itypes.

### Cited Findings
- Latency path: all ops are matvec-based with 16-row blocks. The fusions are RMSNorm+QKV+RoPE+KV-append, RMSNorm+up+gate+SiLU, and the residual add folded into o_proj/down_proj via TMA add-store. The Blackwell matvec path abuses `mma_ABt` with broadcast activations and then `row_max` to extract the result; the Hopper path does elementwise `mul` + `row_sum`. — [demos/low-latency-llama/utils.cuh:39-100](MK/demos/low-latency-llama/utils.cuh); [demos/low-latency-llama/matvec_adds.cu:78-79](MK/demos/low-latency-llama/matvec_adds.cu)
- Throughput path (Python on main): it splits norms into separate "memory" instructions (`PreAttnLayerNorm`, `PreMLP_Norm`, `PreLMHeadRMS`) and keeps the matmuls as "compute" instructions, for use with `pool` SM partitioning. Barriers gain a batch dimension `[layers, 10, bs, max(48, intermediate/128)]`. — [megakernels/demos/throughput/instructions.py:43-223](MK/megakernels/demos/throughput/instructions.py); [megakernels/demos/throughput/scheduler.py:44-56](MK/megakernels/demos/throughput/scheduler.py)
- The throughput branch README says "open-sourcing the code for research reference, as opposed to a product that we intend to support", and its Low-Latency section says "Please use the other branch!". It adds `demos/cross-gpu-llama/*`, including `all_device_barrier.cu`, `inc_barriers.cu`, `attention_prefill.cu`, `matmul_pipeline.cuh` and others, plus `demos/thunder-attention/mla/*`. — [origin/throughput:README.md](MK/README.md); [origin/throughput diffstat](MK/)
- MegaKittens README: "GPU megakernel runtime that fuses PyTorch operator graphs into a single persistent CUDA kernel". It requires Blackwell (sm100a/sm103a), CUDA 13+ and PyTorch 2.9+, and its usage is `@megakittens.compile()`. — [origin/mk-v2:README.md](MK/README.md)
- MegaKittens itypes on mk-v2 are generic: attention, elementwise_binary, elementwise_unary, gemm, noop and rmsnorm. mk-v2-llama adds `llama1b/*` ports of the main-branch ops, `llama70b/*` (matmul_pipeline, half-TMEM variants, attention_decode and others), `copy`, `reduce` and `pos_id_increment`, plus `examples/llama70b/manual_kernels/*` for comparison against standalone kernels. — `git ls-tree origin/mk-v2`; `git ls-tree origin/mk-v2-llama`
- The latest commits are "Llama1b cleanup" (2026-05-26), "finished manual kernels ablation" and "cuda graph for manual decode" (2026-05-17). Ablation branch commits include "Add no-virtual-smem", "Add no barrier option" and "Add compile per layer". — `git log origin/mk-v2-llama`, `git log origin/mk-v2-llama-ablations`

### Inferences
- The project moved from a hand-written, model-specific VM (fixed opcodes, hard-coded barrier counts) toward a compiler (trace → DAG → tiles → auto barriers → JIT dispatch). The "no-virtual-smem" and "no barrier" ablations suggest the authors are quantifying the cost of the paging and barrier machinery itself.

### Gaps
- I did not diff the mk-v2-llama `llama1b/*` ports against main's demo, so I don't know whether the release orders and barrier counts changed. I also did not read the throughput branch's CUDA ops.
- The repo contains no performance results for MegaKittens vs the original megakernel.

---

## Q6. Known limitations, TODOs, hacks and comments that suggest design regrets

### Takeaway
The code is research-grade and specialised. Model dimensions, GQA ratio, page count, pipeline depth and barrier counts are all baked in with `static_assert`s or literals. The scheduler's dependency modelling is partial: the timing in the latency DAG is ignored, and the throughput DAG is broken. There is also a latent config bug in the spin-sleep constant.

### Cited Findings
- **Config bug:** `static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;` is declared `bool`, so the value is `true` (1), and every `__nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS)` sleeps about 1 ns rather than 20. mk-v2 fixes the type: `static constexpr int SPIN_LOOP_SLEEP_NS = 20`. — [include/config.cuh:48](MK/include/config.cuh); [origin/mk-v2:csrc/schema.cuh:73](MK/csrc/schema.cuh)
- Hard-coded shapes:
  - `static_assert(NUM_PAGES == 13)`
  - `static_assert(GQA_RATIO == 4)`
  - `static_assert(INPUT_PIPELINE_STAGES == 3)` inside `release_lid`
  - RMS variance computed as `full_sum / 2048.0f`
  - `K_BLK_START = 2048/16`, `V_BLK_START = 2560/16`
  - `EXPECTED_ARRIVAL_COUNT = 512`
  
  — [include/config.cuh:44](MK/include/config.cuh); [demos/low-latency-llama/attention_partial.cu:14-15](MK/demos/low-latency-llama/attention_partial.cu); [demos/low-latency-llama/matvec_pipeline.cuh:73-79](MK/demos/low-latency-llama/matvec_pipeline.cuh); [demos/low-latency-llama/utils.cuh:29](MK/demos/low-latency-llama/utils.cuh); [demos/low-latency-llama/rms_matvec_rope_append.cu:15-17](MK/demos/low-latency-llama/rms_matvec_rope_append.cu)
- TODOs in the code:
  - "TODO we can do better here and reduce up before gate is ready" — [upgate.cu:83](MK/demos/low-latency-llama/upgate.cu)
  - "NOTE: hardcoding to 3 output stages for now" — [upgate.cu:67](MK/demos/low-latency-llama/upgate.cu)
  - "TODO we can do better - we can start a reduction col's work once that fraction of the upgate work is done" — [latency/scheduler.py:373](MK/megakernels/demos/latency/scheduler.py). The device already supports this via per-column barriers (upgate.cu:117-120), but the host DAG does not model it.
  - "TODO: make sure reduction_list can't go beyond instruction" — [latency/instructions.py:91](MK/megakernels/demos/latency/instructions.py)
  - "TODO limitation until we have a better reduction tree" (partitions ≤ 24) — [latency/scheduler.py:27](MK/megakernels/demos/latency/scheduler.py)
  - "simple signalling strat for now" (attention reduction bumps one counter by 4), with a per-head atomic commented out — [attention_reduction.cu:323-335](MK/demos/low-latency-llama/attention_reduction.cu)
  - "TODO needed?" (warp sync after semaphore invalidation) and "technically don't need to reset, whatevs?" — [controller.cuh:48,160](MK/include/controller/controller.cuh)
  - "TODO: for now, deliberately not using sizeof(sv_t) here because we've had alignment issues before." — [utils.cuh:112-113](MK/demos/low-latency-llama/utils.cuh)
- Activation loads bypass TMA. The `tma::load_async` of activations is commented out, and consumers use `warp::load` from global memory after the spin-wait. — [matvec_pipeline.cuh:322-334](MK/demos/low-latency-llama/matvec_pipeline.cuh)
- Fences are unresolved. `// asm volatile("fence.acq_rel.gpu;")` is commented out at every signal site. — [matvec_adds.cu:172-173](MK/demos/low-latency-llama/matvec_adds.cu); [upgate.cu:114](MK/demos/low-latency-llama/upgate.cu); [attention_partial.cu:642](MK/demos/low-latency-llama/attention_partial.cu)
- Scheduler gaps:
  - The DAG scheduler ignores ready time and priority (scheduler.py:104, 133-134).
  - The throughput DAG has no edges, plus a "TODO: these dependencies are broken" (throughput/scheduler.py:317-318).
  - AttentionReduction is unused in the latency schedule (latency/scheduler.py:276-277).
- `get_worker_id()` uses `%smid` rather than `blockIdx.x`. This silently assumes 1 block per SM and grid == SM count. — [include/util.cuh:27-30](MK/include/util.cuh)
- mk-v2 hacks and TODOs:
  - "hack: use tcgen05.commit for mbarrier broadcast" (controller.cuh:45,54,89)
  - "TODO: is this really needed?" on `fence.proxy.async` (controller.cuh:84)
  - "TODO: change scope to `sys` for multi-gpu setting" (utils.cuh:29-37)
  - "TODO: eventually, we want these to be set during runtime" on src/dst tensors (instruction.py:13)
  - "TODO: handle multi-GPU case" (dispatcher.py:196,314)
  - "TODO: if input dims change, all other dims + gls should change" (dispatcher.py:311)
  
  — [origin/mk-v2:csrc/controller.cuh](MK/csrc/controller.cuh); [origin/mk-v2:csrc/utils.cuh](MK/csrc/utils.cuh); [origin/mk-v2:megakittens/dispatcher.py](MK/megakittens/dispatcher.py)
- The mk-v2 README and CLAUDE.md point to `csrc/itypes/add.cuh` / `megakittens/itypes/add.py` as references, but neither file exists in the mk-v2 tree, so those docs are stale. — [origin/mk-v2:README.md](MK/README.md); `git ls-tree origin/mk-v2`
- On main, the `generate.py th()` path references the `tests/batch-vm/llama_official` directory, which is absent. — [megakernels/scripts/generate.py:59-66](MK/megakernels/scripts/generate.py)

### Inferences
- The repeated "hand-maintained release order plus hard-coded barrier counts" pattern is the main maintainability cost. mk-v2 automates barrier derivation but still requires a hand-written `lid_release_order` per itype ("Getting this right is essential to pipelining instructions", per CLAUDE.md).
- For megabake, the cleanest lessons to carry over:
  - a typed instruction struct shared by host and device;
  - barrier ids and targets computed on the host and embedded per instruction;
  - separate RAW and WAR (reuse) barrier lists;
  - a type-correct spin-sleep constant;
  - deciding the release-fence question deliberately rather than by comment.

### Gaps
- I did not verify that the `bool` sleep constant actually changes measured behaviour. That is a C++ semantics inference with no benchmark behind it.
