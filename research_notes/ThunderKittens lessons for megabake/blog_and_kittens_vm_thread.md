# "Retire the Abstractions" blog + Flapping Airplanes "megakernel framework" thread (+ the Kittens VM / Megakernels lineage)

Retrieval status, read first:
- **Blog** (Hazy Research, 2026-08-05): fetched in full as raw HTML. Every quote below is verbatim.
- **X thread** (`flappyairplanes/status/2112955212612919318`, 2026-10-21): x.com returns HTTP 402. xcancel is suspended and nitter.net is offline. The text of the root tweet and the author's self-reply chain **was retrieved** through `api.fxtwitter.com` (`/2/conversation/<id>`), plus the first page of replies. **The thread never uses the phrase "Kittens VM".** It is a $1B Series A announcement from Flapping Airplanes, a lab co-founded by ThunderKittens lead author Ben Spector. It has six technical or marketing tweets and gives no architecture details: no instruction format, no scheduler, no numbers. Most replies are congratulations. Only the first page of replies could be scanned, because cursor paging returned 404.
- **"Kittens VM"**: the only concrete artifact I found under that name is in the open-source HazyResearch Megakernels code. Its per-SM interpreter state object is named `kvms` and the interpreter is called "the VM" (e.g. `"Overall VM execution time"`, `"Forward a NoOp to the VM"`). My inference is that "Kittens VM" means this ThunderKittens-based on-GPU instruction interpreter, and that is how it is documented below. Web searches for "Kittens VM" (with "GPU" and with "megakernel") turned up nothing else. The results were kitty-terminal pages and the unrelated Sandia "Kitten" OS.
- The blog's one directly linked new artifact is Cursor's **Mixture-of-Kittens (MoK)** post (2026-08-04). It is the megakernel that was "built without the abstraction", so it is covered in depth.
- Material from 2025 is labeled **[2025]**. Everything else is from 2026.

---

## Q1. What does "Retire the Abstractions" argue? Which abstractions are retired, and what replaces them?

### Takeaway
Sul & Ré argue that the C++ abstraction layer they built in 2025 to make megakernels tractable (the Megakernels interpreter framework) was a "cognitive offloader". Coding agents now do that job: the "abstraction" lives in a messy prompt, and target-specific code is regenerated from scratch. They predict CUDA DSLs, ThunderKittens included, are "next on the retirement list". What survives is intent, invariants, tests/oracles and domain knowledge.

### Cited Findings
- Authors and date: Stuart Sul and Chris Ré, Aug 5, 2026, "6 min read". TL;DR: "We think CUDA DSLs are heading toward retirement". [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- Why megakernels needed an abstraction in the first place: "implementing megakernels can be quite miserable. It requires complex data structures, extensive inter-thread/SM/GPU synchronization, deeply nested control flow, etc. We couldn't hold it in our heads, so we did what computer science has done for 70 years: we built a layer of abstraction" (links to github.com/HazyResearch/Megakernels). [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- **Regret / failure mode of the abstraction era:** "Even with it, we had to fight through race conditions and deadlocks for a couple months before the Llamas ran blazingly fast." [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- **The abstraction they deleted:** "This year, we built an MoE megakernel [Cursor MoK], but we deleted the abstraction. With agents, we could work through the complexity directly and build target-optimized code from scratch. No intermediate layers of C++ abstraction were needed." [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- How far automation goes today: "The tasks we could already do without abstractions (e.g., writing an optimized GEMM kernel) are now nearly automated given the right prompt. For now, we still have to tell the agent which PTX instructions to use, the warp-specialization design, and so on, but it gets to the state-of-the-art pretty darn quickly." [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- What is not yet automated: "we certainly can't one-shot a megakernel today. But agents let us put the abstraction in the prompt, in an incomplete, messy form, rather than in carefully designed C++ templates. ... The ideas that went into the megakernel still mattered just as much, but it was much easier to express." [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- Thesis line: "The job of abstraction as a cognitive offloader is starting to retire. Agents are taking that job." [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- Prediction: "By inductive reasoning, CUDA DSLs are next on the retirement list, including our beloved ThunderKittens. Probably next year, or maybe sooner. We don't know how far down this will go. Somewhere below us is a floor where an abstraction is purely the contract, rather than an offloader, and it stops." Footnote: "We are still going to update, maintain, and love our ThunderKittens. Vera Rubin kernels coming soon!" [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- DSL framing: "there's now a competition between an intelligent executor reading underspecified instructions and doing the right thing, versus specifying every gap so a dumb executor doesn't need judgment. DSLs and frameworks are that specification, expressed as a codebase. If the executor stops being dumb, the DSL loses its ground." [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- On codebases: they must be "precise enough for a machine, readable enough for a human, and flexible enough for the future", which is "a nearly impossible set of constraints for a string of tokens to satisfy". A codebase "appears portable, but barely". [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- Compiler analogy: "Intelligence already sits between intent and machine. The question is whether letting it sit higher is a difference in degree or in kind, and we don't think it's in kind." [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- **Eligibility caveats (when NOT to retire):**
  1. An abstraction is "a shared surface where application, reuse, and review attach. ... Ten teams on ThunderKittens test the same tile semantics, and that compounds over time. Ten teams generating bespoke megakernels have ten disjoint sets of problems, and they don't stratify."
  2. "There is no shared oracle. For a megakernel, the tests are the contract and the work. ... When we deleted the framework, we kept the oracle: reference implementations, numerics tolerances, our intuition of what the profiled Gantt chart should look like, etc. ... We can only retire a layer once we have an oracle that outlives it."
  3. "We're a single, biased sample. We deleted abstractions in a domain we knew deeply. Abstractions also exist to transmit knowledge to people who don't have it yet". [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- What replaces the abstraction: "The intent, the invariants, the tests, and the hard-earned domain knowledge that currently lives inside ThunderKittens's abstractions ... Trust moves up a level. We scrutinize the spec and the oracle, not the diff. The implementation becomes disposable: a cache of one particular compilation, not the source of truth." Closing line: "Abstractions retire. Ideas stay." [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- The companion Cursor post makes the same claim from the MoK side: "Last year, writing a megakernel required creating a layer of simplified abstractions like this one [Megakernels]. This year, we did not need a framework. ... Agents automated the simpler tasks (e.g., single-operator kernels) and helped us get through the harder ones (e.g., writing a distributed MoE megakernel) much faster and with a very small team." Also: "Given the right design directions (e.g., how to specialize warps, which PTX instructions to use), agents can one-shot state-of-the-art kernels." [Cursor MoK](https://cursor.com/blog/mixture-of-kittens)
- Earlier stance, showing the arc (Feb 2026): TK 2.0 said the way to make megakernels accessible was "Rather than another DSL or standalone compiler, we believe what's needed is a fundamental change in ML infrastructure ... [PyTorch's] 10+-year-old backend architecture may not be the best fit". TK 2.0 also advertised a "Much simpler build structure ... so you (or your agent) can easily adapt them". [TK 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)
- Earlier stance [2025]: the throughput post's stated next step was the opposite of retirement: "we'd like to design a more general megakernel instruction set and abstract these decisions into the host-side scheduler, in order to simplify the process of designing high-performance megakernels." [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)

### Inferences
- The concrete "retired abstraction" is the Megakernels interpreter/VM template: per-SM instruction queues, the opcode-dispatch `ops...` pack, paged smem, and controller-managed semaphores. It is not ThunderKittens tiles, which they still maintain. Within roughly 10 months (Sep 2025 to Aug 2026) the group went from "build a more general VM/instruction set" to "delete the VM and hand-generate target-specific megakernels with agents".
- For MegaBake, the "oracle" list is the transferable part: reference implementation, numeric tolerances, and an expected profiler Gantt chart. A compiler-generated megakernel needs the same oracle and gets the "shared surface" benefit that bespoke agent-written kernels lose.

### Gaps
- The blog gives no quantitative comparison (dev time, LOC, bug count) between the abstraction-based and abstraction-free megakernels. The claim is purely qualitative.
- The blog does not say which agent or model was used, or what the prompts looked like.

---

## Q2. What is the "Kittens VM": instruction format, opcode/dispatch, SM scheduling, synchronization, smem paging, cross-instruction pipelining?

### Takeaway
The "VM" is the HazyResearch Megakernels on-GPU interpreter **[2025 code, last pushed 2026-05-26]**. Each SM runs one persistent CTA. A controller warp fetches 128-byte (32×int32) instructions from a per-SM queue in global memory, dispatches on opcode word 0 through a compile-time variadic `ops...` pack, allocates 13×16 KiB smem pages, and builds per-instruction mbarrier semaphores. Loader, consumer, storer and launcher warps then run the instruction's role-specific functions with a 2-deep instruction pipeline. Cross-SM dependencies use global-memory counters. Scheduling was static per-SM queues for latency, and a dynamic atomic global work queue for throughput.

### Cited Findings
**Instruction format and encoding**
- `INSTRUCTION_WIDTH = 32; // 128 bytes per instruction`, `using instruction_t = int[INSTRUCTION_WIDTH]`; `INSTRUCTION_PIPELINE_STAGES = 2`; `TIMING_WIDTH = 128`; `DYNAMIC_SEMAPHORES = 32`. [config.cuh](https://github.com/HazyResearch/Megakernels/blob/main/include/config.cuh)
- Python side: `Instruction.serialize()` emits `words = [self.opcode()]`, followed by each dataclass field. Ints go in as-is. Tuples and lists are written as length then elements, and `None` becomes 0. `NoOp` is opcode 0. [instructions.py](https://github.com/HazyResearch/Megakernels/blob/main/megakernels/instructions.py)
- Instructions live in a global tensor laid out `gl<int, 1, -1, -1, INSTRUCTION_WIDTH>`, indexed `{worker_id (SM), instruction_index, 0}`. The fetch is one warp-wide copy: `if (laneid < INSTRUCTION_WIDTH) instruction[laneid] = src_ptr[laneid]`. [config.cuh](https://github.com/HazyResearch/Megakernels/blob/main/include/config.cuh), [instruction_fetch.cuh](https://github.com/HazyResearch/Megakernels/blob/main/include/controller/instruction_fetch.cuh)

**Opcode dispatch / kernel body authoring**
- The kernel is `mk<config, globals, ops...>`. The wrapper prepends `NoOp` ("Forward a NoOp to the VM, to ensure that the VM can support zeros"). `dispatch_op<..., ops...>::run(opcode, ...)` selects the op struct whose `opcode()` matches. Each op provides per-role hooks such as `op::controller::release_lid(...)` and `op::controller::init_semaphores(...)`, plus loader/consumer/storer/launcher main-loop bodies. [megakernel.cuh](https://github.com/HazyResearch/Megakernels/blob/main/include/megakernel.cuh), [page_allocator.cuh](https://github.com/HazyResearch/Megakernels/blob/main/include/controller/page_allocator.cuh), [semaphore_constructor.cuh](https://github.com/HazyResearch/Megakernels/blob/main/include/controller/semaphore_constructor.cuh)
- [2025] "each streaming multiprocessor (SM) within the GPU receives a sequence of instructions (each implemented using the same CUDA template) and executes them." "We implement each of these instructions using a common CUDA template (with load, store, compute boilerplate functions)". [No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- [2025] "Each instruction specifies a unit of work that would traditionally be performed by a thread block, e.g. compute an output tile ... each instruction is organized into dedicated sections, e.g. a load function ..., a compute function, and a store function". [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- Llama-1B instruction set [2025]: "A fused RMS norm & QKV & RoPE instruction", attention partial and "An attention reduction instruction", "O-projection + residual", "A fused RMS norm & up-gate & SiLU instruction", "A down-projection + residual instruction", and "An RMS norm & language modeling head instruction". [No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles). The demo files match: `rms_matvec_rope_append.cu`, `attention_partial.cu`, `attention_reduction.cu`, `matvec_adds.cu`, `upgate.cu`, `rms_lm_head.cu`. [repo tree](https://github.com/HazyResearch/Megakernels/tree/main/demos/low-latency-llama)

**Warp roles**
- Config: "One controller warp, one load warp, one store warp, and one mma warp." `NUM_CONSUMER_WARPS = 16`, `NUM_WARPS = 4 + 16` (640 threads), `NUM_BLOCKS = 1`, `CONSUMER_REGISTERS = 104`, `NON_CONSUMER_REGISTERS = 64`. Consumers call `increase_registers`. Non-consumer warpgroup warp 0 is the loader, 1 the storer, 2 the launcher, 3 the controller. [config.cuh](https://github.com/HazyResearch/Megakernels/blob/main/include/config.cuh), [megakernel.cuh](https://github.com/HazyResearch/Megakernels/blob/main/include/megakernel.cuh)
- [2025] "each of the load, compute, and store functions within the instruction template is executed by its own, independent set of threads ... in the background are controller threads that help coordinate the interpreter, although they're not programmed by the user." [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)

**Controller loop (per instruction)**, from [controller.cuh](https://github.com/HazyResearch/Megakernels/blob/main/include/controller/controller.cuh):
- Step 0: if the ring slot was used before, `wait(instruction_finished[ring], phasebit)` and then `invalidate_semaphore` on the old instruction's semaphores.
- Step 1: `load_instructions`.
- Step 2: "Establish physical page order". It calls the *previous* instruction's `release_lid(laneid)` to map the order in which the previous instruction releases its logical pages onto physical page ids (`pid_order`).
- Step 3: "Construct semaphores" via the op's `init_semaphores` (returns a count, at most 32), followed by `fence.proxy.async.shared::cta`.
- Step 4: `arrive(instruction_arrived[ring])`, "Let the rest of the world know that next instruction is ready to roll!"
- `instruction_finished` is initialized with count `NUM_WARPS - 1`, so all non-controller warps must arrive.

**Shared-memory paging**
- `PAGE_SIZE = 16384`; `NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE`; `static_assert(NUM_PAGES == 13)`. Static smem is `512 + STAGES*(SCRATCH_BYTES(4096) + (32+128)*4 + 32*8)`. Each page has `page_finished[NUM_PAGES][STAGES_BITS]` semaphores counted in `NUM_CONSUMER_WARPS`. [config.cuh](https://github.com/HazyResearch/Megakernels/blob/main/include/config.cuh), [megakernel.cuh](https://github.com/HazyResearch/Megakernels/blob/main/include/megakernel.cuh)
- [2025] "We first divide the first 213kB of shared memory on an H100 into 13 16KiB pages ... instructions have to explicitly request and release them from the interpreter. The interpreter automatically passes released pages to the next instruction, allowing them to start issuing memory loads as early as shared memory becomes available." [No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- On Blackwell builds the VM also manages a `tensor_allocator_t` (TMEM) and a `tensor_finished` semaphore. [megakernel.cuh](https://github.com/HazyResearch/Megakernels/blob/main/include/megakernel.cuh)

**Cross-SM synchronization**
- [2025] "we initialize an array of counters (i.e. integers) in GPU global memory with a starting value of zero. Whenever an instruction completes, it increments one of these counters. Similarly, whenever a new instruction starts, it must wait for some of these counters to reach a target value". Finer granularity for down-proj: "produce and consume the intermediate state in four chunks, each with their own counter." [No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- The globals carry a `barriers: Tensor`. The spin-wait sleep is configured by `GMEM_SPIN_LOOP_SLEEP_NANOS = 20`. Note that it is declared `static constexpr bool` in the source, apparently a typo for int. [instructions.py](https://github.com/HazyResearch/Megakernels/blob/main/megakernels/instructions.py), [config.cuh](https://github.com/HazyResearch/Megakernels/blob/main/include/config.cuh)
- [2025] Throughput variant: "synchronization is at the granularity of 128 output rows of a matrix; other times it's at the granularity of an individual attention head for an individual token." [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- TK 2.0 lesson: "tightening the acquire/release pattern usage was one of the last optimizations required for our Megakernel implementation to surpass SGLang: we observed that a few loose fence instructions caused over 10% loss in performance." [TK 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)

**Scheduling (static vs dynamic)**
- [2025] Latency megakernel: "We schedule each SM's instruction sequence ahead of time on the Python side, and notably we can reuse each schedule for hundreds of forward passes!" [No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- Host schedulers in the repo: `round_robin_assign_to_sms`, `zig_zag_assign_to_sms`, `wave_assign_to_sms` (sorts each wave by cost and assigns to the least-loaded SM via a heap), `pool_assign_to_sms(memory_fraction)`, and DAG-based `assign_dag_to_sms` (list scheduling: pops the max-cost ready node and assigns it to the earliest-free SM using `instruction.cost(globs)`). [scheduler.py](https://github.com/HazyResearch/Megakernels/blob/main/megakernels/scheduler.py)
- [2025] Throughput megakernel switched to dynamic: "we create a global work queue ... When an SM needs to fetch a new instruction to run, it atomically increments a global instruction counter". Why it was not used for latency: "the runtime of each instruction was so fast that the latency of this atomic increment would be prohibitive. But with a throughput-oriented megakernel -- where individual instructions frequently take 100 microseconds or more -- this cost can be entirely hidden". [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- [2025] The interpreter is only affordable because "our instructions are really big, operating at the granularity of dozens of microseconds, which helps amortize the modest overhead of the interpreter." "Without this, we couldn't afford the overhead of an interpreter, and we'd need to compile in advance!" [TP-Llama intro](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-intro)
- [2025] Interleaving: different op types are mixed into the global queue, producing "quilts" with no explicit SM grouping as in NanoFlow. [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- [2025] CPU-side: "with 64 threads generating instructions in C++, we generally find >90% CPU idle time." [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)

**Pipelining across instruction boundaries**
- [2025] "our interpreter will start loading the model weights for an instruction as soon as it can, even if a previous instruction is still finishing up". [No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- [2025] Profiled transition (Gate SiLU into Up matmul): the controller is signaled "after fourth-to-last load". The next instruction "Resolves dependency check, and issues loads for first 2.5 stages, before stalling due to lack of available shared memory pages". Then "Final store finishes reading from shared memory, releases last page", it "Atomically increments flag in global memory signifying instruction completion", and teardown invalidates semaphores and writes timings. [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- [2025] Multi-GPU traffic is issued from storer threads ("We perform all communication from our dedicated storer threads"). It uses TK's Parallel Global Layout and NVSwitch acceleration, and "removed our reliance on NCCL entirely". [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)

**Profiling**
- Timing slots: `TIMING_WIDTH=128` ints per instruction, with events such as `TEVENT_CONTROLLER_START`, `TEVENT_IFETCH_DONE`, `TEVENT_PAGE_ALLOC_DONE` and `TEVENT_SEMS_SETUP`. [controller.cuh](https://github.com/HazyResearch/Megakernels/blob/main/include/controller/controller.cuh). Instructions "only report events from the first 8 and last 4 stages, since we have limited timing slots". Timing overhead averaged "0.39%, with a maximum of 1.07%". [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)

### Inferences
- The design splits control (fetch/decode, page mapping, semaphore construction: the controller warp) from data roles (loader/consumer/storer). The page hand-off is decided by the previous instruction's `release_lid` order, which is what makes "start loading the next weights while the last store drains" possible without a general allocator. This maps directly onto a MegaBake-style compiler: page release order can be computed statically.
- `kvms` (in `controller::main_loop(const globals &g, ::megakernel::state<config> &kvms)`) is the only literal "Kittens VM" artifact I could locate.

### Gaps
- I did not read the loader/consumer/storer/launcher main loops (`loader.cuh` etc. are 5–6-line shims that include elsewhere) or the `throughput` branch's global-queue code. The atomic work-queue claims come from the blog only.
- No source describes a "Kittens VM" as a distinct named product from Flapping Airplanes. Whether their "megakernel framework" descends from this VM is unconfirmed.

---

## Q3. Performance numbers, models and hardware

### Takeaway
2025 (VM era): Llama-3.2-1B bs=1 under 1 ms per forward pass on H100 (78% of HBM BW, ~2.5× vLLM, >1.5× SGLang) and under 680 µs on B200. Llama-70B TP8 on H100 was +22% throughput vs SGLang. 2026 (post-abstraction): MoK on GB300 NVL72 is up to 2.37× vs the best public MoE baseline and 1.41× end-to-end training tokens/s. Flapping Airplanes published no numbers.

### Cited Findings
- [2025] Llama-1B, H100: "we use 78% of memory bandwidth and outperform existing systems by over 1.5x"; "almost 2.5x faster than vLLM and over 1.5x faster than SGLang". B200: "under 680 microseconds per forward pass", ">3.5x" vs vLLM. The B200 theoretical limit is "~3,000 forward passes per second". [No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- [2025] B200 600 µs breakdown: "250 microseconds are spent storing activations, awaiting consistency, and loading them" (two load and two store latencies per instruction at ~500 ns each, about 200 µs, plus about 50 µs suspected to be atomics). "200 microseconds ... RMS norm and matrix-vector" (95% matvec; on Hopper "better to simply run on the CUDA cores", on Blackwell tensor cores are "marginally helpful"). "80 microseconds are on setup and various other overheads (e.g. passing instruction barriers, marking pages as complete, etc.)" [No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- [2025] Launch costs (H100 dummy kernel): "about 2.1 microseconds" on a stream and "around 1.3 microseconds" with CUDA graphs. [No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- [2025] Llama-70B TP8 H100 in Tokasaurus, ShareGPT 65,536 prompts: total 23,468 tok/s vs SGLang 19,170 (input 14,425 vs 11,783; output 9,043 vs 7,387). Pipelining ablation: gap 3.4 µs vs 10.2 µs, ">7%" on those instructions, 31,516 vs 29,607 TPS, "around 2-6% end to end MFU". Global queue vs round robin: "14.2% end-to-end reduction" without it at bs 8192, and "At very small batch sizes, the overhead in the global work queue actually outweighs its benefit." Interleaving: +6.4% at bs 8192, ~0% at bs ≤2048. O-proj replication costs "an additional 9 GB of memory per GPU" and "reduces the maximum batch size by about 15%". [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- 2026 MoK (GB300 NVL72, EP64, 2,048 tokens/GPU): "up to 2.37x faster for the MXFP8 forward, 1.78x for the MXFP8 backward, 1.92x for the BF16 forward, and 1.58x for the BF16 backward, compared to the fastest baseline". Baselines were NCCL+PyTorch, DeepEP+PyTorch, DeepEP+TransformerEngine and HybridEP+Megatron. Shapes: Kimi K2.7 Code, GLM-5.2, Qwen3.5-397B-A17B and DeepSeek-V4-Pro. End-to-end on 512 GB300 GPUs: 760.9 to 1,070.2 tokens/s/GPU (1.41×). [Cursor MoK](https://cursor.com/blog/mixture-of-kittens)
- MoK micro-numbers: the schedule kernel is "<3% of the total MoE runtime". Pull NVLink is "up to 29% higher" utilization under imbalance. Push-dispatch signalling costs "103 µs versus 18 µs" (5.8×). The minibatch sweep for Kimi shapes bottoms out at T≈2560 (3.425 ms) vs 5.981 ms at T=512, consistent with the heuristic `T ≥ 2C·128·256 / min(2I,H)` giving T≥2368 at C=148. [Cursor MoK](https://cursor.com/blog/mixture-of-kittens)
- Flapping Airplanes thread: no performance numbers. The claims are qualitative ("world's most complete megakernel framework"; "evidence of this at trillion-parameter pretraining scale"). [FA tweet 2](https://x.com/flappyairplanes/status/2112955825367269788), [FA tweet 5](https://x.com/flappyairplanes/status/2112956524939342206)

### Inferences
- The 2025 latency breakdown (≈42% of time in store→flag→wait→load round trips) is the clearest published evidence that cross-SM global-memory synchronization, not the interpreter, dominates bs=1 megakernel time once bubbles are gone.

### Gaps
- The MoK paper (arXiv:2609.36070) was not read. The numbers here are from the blog only.
- The per-batch-size tables for the global work queue (Table 2) and pipelining were only partially captured.

---

## Q4. Lessons, failure modes, regrets; what works badly

### Takeaway
The recurring pain points are synchronization bugs (races, deadlocks, loose fences) and the difficulty of designing custom instruction sets. Interpreter overhead is acceptable only for coarse instructions. Static schedules can't absorb jitter, while dynamic queues cost too much at tiny instruction sizes. Hardware SM partitioning (green contexts) was unreliable. The 2026 answer is to drop the framework and use agents plus a strong oracle, with the caveat that verification no longer "stratifies".

### Cited Findings
- "fight through race conditions and deadlocks for a couple months". [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- [2025] "a key challenge of writing megakernels is that there is tremendous complexity in both designing these custom instruction sets, and coordinating (and especially debugging) synchronization patterns across GPUs." [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- [2025] "designing these interpreters is really hard, and ... more of the work ought to be offloaded to the CPU-side scheduler." [TP-Llama intro](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-intro)
- [2025] Code fragility: "it is sensitive to compiler versions, GPU setup, and sometimes even being looked at the wrong way, and we have no intention whatsoever of supporting it." Remaining issues: "Our scheduling heuristics are quite simple, our instructions frequently stall on synchronizations that could likely be hidden better, and our megakernel still has register spills and other low-level problems." [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- [2025] Recompute vs communicate tradeoff: in the latency kernel, "our QKV matrix-vector product fused the RMS norm into its beginning, saving an instruction boundary, but performing identical RMS computations over a hundred times. When focusing on throughput, this recomputation is not worth it." [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- Fences: "a few loose fence instructions caused over 10% loss in performance." [TK 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)
- MoK: "for inter-SM overlapping, we found multiple streams with green contexts unreliable at partitioning SMs exactly as intended, while software partitioning gives exact allocation guarantees." Another reason to use a megakernel: minibatching without fusion "would require multiple kernel launches". [Cursor MoK](https://cursor.com/blog/mixture-of-kittens)
- MoK: comm/compute granularity has an interior optimum. Too fine and "the tensor cores never fully saturate", too coarse and they "wait too long for the first round of tokens". The recommendation is "at least two full waves" per grouped GEMM. [Cursor MoK](https://cursor.com/blog/mixture-of-kittens)
- MoK: Grace CPUs are slow, so CPU-GPU syncs hurt ("Our traces often show GPU kernels bottlenecked by CPU work such as logging"). This motivated device-side scheduling and ring buffers. [Cursor MoK](https://cursor.com/blog/mixture-of-kittens)
- Retirement caveats (verification fragmentation, need for an oracle, onboarding) are covered in Q1. [Blog](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- Secondary, unverified: a Spanish summary of a Latent Space discussion reports claims that "ningún proveedor serio de inferencia usa un kernel de 67k líneas de código fusionado a mano en producción" ("no serious inference provider runs a 67k-line hand-fused kernel in production"). It also reports NVIDIA's Kyle Kranen saying Rubin's tile-level dependency triggers "mata los megakernels" ("kills megakernels"). [ecosistemastartup](https://ecosistemastartup.com/megakernels-2026-cursor-logra-2-37x-mas-velocidad-con-mixture-of-kittens/). I did not verify these against the original podcast.

### Inferences
- The hardest problems they name are dependency/sync correctness and per-model instruction-set design. These are exactly what a compiler that derives instructions and counters from a graph (MegaBake) automates. "Retire the abstractions" argues agents now cover that ground for experts. It does not argue that the ideas (paging, counters, cross-instruction pipelining) are wrong.

### Gaps
- No explicit list of deadlock root causes is published.
- The Latent Space source was not checked directly.

---

## Q5. Relation to earlier HazyResearch megakernels work, MoK, and Flapping Airplanes

### Takeaway
Lineage: ThunderMLA's on-device scheduler, then the Llama-1B low-latency VM (May 2025, static per-SM queues), then the Llama-70B TP throughput VM (Sep 2025, global atomic work queue, in-kernel comms), then TK 2.0 (Feb 2026), then MoK (Aug 2026, abstraction deleted and agent-built; software SM partitioning into comp/comms SMs with counters, CLC scheduling), and finally the "Retire" essay. Flapping Airplanes (Oct 2026), co-founded by TK lead author Ben Spector, now claims a general megakernel framework for training, inference and games in one kernel, without technical disclosure.

### Cited Findings
- [2025] "Our approach is built on an on-GPU interpreter – essentially a more sophisticated version of our infrastructure underlying ThunderMLA". [No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- [2025] "our core megakernel abstraction (a pipelined instruction interpreter that runs on each SM) is highly transferable across both domains." Authors: Benjamin Spector*, Jordan Juravsky*, Stuart Sul*, et al. [TP-Llama main](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main), [TP-Llama intro](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-intro)
- MoK design (abstraction-free): "We built MoK using the megakernel technique, where instructions overlap at the SM task level instead of being separated by kernel launch boundaries." The ideas are carried over from the VM, but the implementation is bespoke:
  - Comp SMs and comms SMs "signal each other through a local counter".
  - NVLink can be saturated "with less than a third of the SMs" using TMA.
  - The number of comms SMs is tunable separately for forward and backward.
  - A device-side schedule table `{src_rank, src_index}` is built once and reused for all four all-to-alls.
  - Pull dispatch and push combine remove cross-GPU signalling.
  - A fixed-size ring buffer ("macrobatch", "a few hundred megabytes") is cycled at minibatch granularity with dispatch-combine interleaving and a reversed ring to minimize backward replay.
  - Fully deterministic ("bitwise-identical output regardless of hardware scheduling").
  - Tasks are scheduled through Cluster Launch Control (CLC), "a hardware-native work-stealing feature introduced in Blackwell", so the kernel "can naturally yield to kernels on a higher-priority stream" (e.g. FSDP all-gather).
  - MXFP8, with quantization fused into dispatch, grouped GEMMs and SwiGLU.
  - Authors: Stuart Sul, Nash Brown, Henry Wildermuth, William Lin, Federico Cassano. Code at github.com/cursor/mixture-of-kittens. Paper arXiv:2609.36070.
  [Cursor MoK](https://cursor.com/blog/mixture-of-kittens)
- Flapping Airplanes thread, verbatim (all 2026-10-21, retrieved via fxtwitter API):
  1. "We've spent the past year developing new primitives for GPUs. We believe this enables new algorithms to change how machines learn, open new markets, and dramatically reshape AI's cost structure. We have raised a $1 billion Series A, co-led by Index and Lux." [tweet](https://x.com/flappyairplanes/status/2112955212612919318)
  2. "We've developed the world's most complete megakernel framework. It can perform entire training runs, serve inference, play video games, and even do all three simultaneously in a single kernel." [tweet](https://x.com/flappyairplanes/status/2112955825367269788)
  3. "Our systems work makes training & inference cheaper—but it also makes GPUs more expressive. The same framework that allows us to overlap memory-bound inference with compute-bound training also enables odd architectures, objectives, and optimizers." [tweet](https://x.com/flappyairplanes/status/2112956283284488483)
  4. "...unlock new markets in AI, including robotics, scientific discovery ... We imagine a world where models can be intelligent without ingesting half the internet." [tweet](https://x.com/flappyairplanes/status/2112956439086125194)
  5. "We have barely scratched the surface of GPUs, and we think GPUs are more flexible and valuable than most believe. Internally, we already see evidence of this at trillion-parameter pretraining scale." [tweet](https://x.com/flappyairplanes/status/2112956524939342206)
  6. "If you are interested in helping us abuse GPUs in new and exciting ways, please reach out." [tweet](https://x.com/flappyairplanes/status/2112956603502911718)
  The remaining tweets are investor thank-yous (NVIDIA among them).
- Context: the founders are Ben Spector (CEO, Stanford CS PhD student, "lead author on ThunderKittens"), Asher Spector and Aidan Smith. The round is $1B at a reported $6B valuation, with $1.18B raised in total. [SiliconANGLE](https://siliconangle.com/2026/10/21/year-old-ai-research-lab-flapping-airplanes-raises-1b-at-a-reported-6b-valuation/), [TechCrunch](https://techcrunch.com/2026/10/21/year-old-ai-lab-flapping-airplanes-raises-1b-at-a-6b-valuation/). Index: "Flappy's megakernels, which run an entire program on a GPU as a single kernel, eliminate the wasted time between steps"; the team works "from the kernels that run on GPUs through pre-training, post-training, and inference". [Index Ventures](https://www.indexventures.com/perspectives/toward-more-efficient-ai-leading-flapping-airplanes-1b-series-a/). Press notes the claims are unverified: "its capabilities are company claims rather than demonstrated results". [Superpower Daily](https://superpowerdaily.com/posts/flapping-airplanes-raises-1-billion-to-pursue-more-efficient-ai-training). One outlet reports a $60B valuation, which conflicts with the $6B figure in the others. [KuCoin flash](https://www.kucoin.com/news/flash/ai-startup-flapping-airplanes-raises-1-billion-at-60-billion-valuation-in-9-months)

### Inferences
- There are now two divergent branches from the same 2025 VM:
  - Hazy/Cursor (Sul): retire the general VM and build bespoke, agent-written, target-specific megakernels, keeping the oracle.
  - Flapping Airplanes (Spector): push toward a *more* general framework, multi-workload in one kernel (memory-bound inference co-scheduled with compute-bound training), which fits the 2025 "more general instruction set + host-side scheduler" direction.
  Which branch Flapping Airplanes' framework actually follows (interpreter vs compiled) is undisclosed.
- MoK's use of CLC for yielding, and software SM partitioning instead of green contexts, are 2026 Blackwell-era choices absent from the 2025 VM. Both are relevant if MegaBake targets B200/GB300 or needs to coexist with comms kernels.

### Gaps
- **The X thread contains no "Kittens VM" description and no architecture details.** If the requester saw "Kittens VM" in a quote-tweet, a reply deeper than the first page, or the attached GIF (`video.twimg.com/tweet_video/HVK5hvPa4AAuD2s.mp4`, not inspected), it was not retrievable here.
- Flapping Airplanes has published no blog, paper or code describing its framework's instruction format, scheduling or numbers, based on its site (flappingairplanes.com is essentially empty), the thread, and press coverage.
- The MoK GitHub repo and the arXiv paper were not read, so MoK's internal task and instruction representation (e.g. whether it keeps any opcode/queue structure) is unknown beyond the blog's description.
