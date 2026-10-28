<p align="center">
  <img src="assets/megabake.png" width="200" />
</p>

# MegaBake

MegaBake turns a captured PyTorch inference workload into a planned, composable CUDA megakernel. PyTorch and Hugging Face keep the model and runtime interface. MegaBake owns region fusion, CuTe DSL device bodies, scheduling, and execution planning.

- **Website:** [megabake.github.io](https://megabake.github.io)
- **Megakernels:** [megabake/megakernels](https://github.com/megabake/megakernels)

## Documents

| Document | What it holds |
| --- | --- |
| [docs/north-star.md](docs/north-star.md) | The fixed scope |
| [docs/architecture.md](docs/architecture.md) | The compiler as built and its target design |
| [docs/roadmap.md](docs/roadmap.md) | **What to do next**, what is missing, and the measured state |

The compilation path is:

1. Capture the invocation with Dynamo and preserve the PyTorch runtime contract.
2. Receive live FX after the pinned AOT/Inductor preparation and post-grad passes, before lowering and scheduling.
3. Recover tensor semantics and regions with explicit layouts, dependencies, effects, and numerical requirements.
4. Select CuTe device bodies (tuned where available, generated otherwise) and plan their execution in one persistent kernel.
5. Check the complete invocation and measure it against the strongest equivalent baseline.

`mode=prefill|decode` selects policy defaults and validates the captured state contract; actual dimensions select the bodies. Strict megakernel compilation is the default (`fallback=error`); `fallback=inductor` delegates the whole invocation to Inductor and labels the result `inductor_fallback`. Pass `compile_kernel=False` to keep the FX reference path for CPU-only capture work.

## Current status

Complete Hugging Face decoders compile into one cooperative CuTe kernel, in prefill and stateful decode: SmolLM-135M and 1.7B, Qwen2-0.5B, the DeepSeek-V3 0.6B draft, the Beam DFlash2 draft, gemma-2b, Gemma-3-270M, and Qwen3.5-0.8B (decode only; its prefill needs `cumsum`). The kernel covers the embedding, every layer, the final norm, the vocabulary projection, and the KV-cache updates.

- **Projections:** the Hopper TMA/WGMMA GEMM. Launches whose GEMMs have at most 64 rows use swapAB; `addmm` biases are added in the epilogue.
- **Attention:** declared through transformers' `AttentionInterface` (`megabake::attention`, `megabake::kv_attention`). Decode is flash decoding: each row's context splits across CTAs, and at long contexts a row packs the query heads that share a KV head, so each cached key and value is read once.
- **Small operators** (norms, RoPE, residual adds, gated activations) run generated bodies and, at decode sizes, fold into their consumer's barrier interval.
- **Synchronization:** a counting grid barrier (~0.7 µs at 60 CTAs). Full models do not use events yet.

Limits: SM90 and bf16 only; static shapes served through size buckets; decode KV capacity ≤ 8,192; prefill is slow above ~64 tokens; the compiler is still tuned to Llama-shaped models (see [docs/roadmap.md](docs/roadmap.md)).

## Current performance

On a 60-SM H200 MIG slice, bf16 (full table in [docs/roadmap.md](docs/roadmap.md#measured-state-2026-10-21)):

- **Decode, pipelined calls:** 5–55% faster than a CUDA-graphed Inductor decode on 6 of 8 models; 1.3x slower on Gemma 3 and 2.2x slower on Qwen3.5.
- **Long-context decode** (2026-10-27, flash-decoding attention): 6–29% faster than before at 4,096 cached keys on the head_dim-64 models. There SmolLM-135M, Qwen2-0.5B and the DeepSeek draft run 1.8–2.1x faster than graphed Inductor; SmolLM-1.7B is still 1.5x slower ([docs/roadmap.md](docs/roadmap.md#done-part-of-step-11-flash-decoding-attention-2026-10-27)).
- **Decode, synchronized calls:** roughly ties graphed Inductor; ~600 µs of host work per call loses four models.
- **Prefill** is slower than Inductor + CUDA graphs on every model, and 3–21x slower at 128–300 tokens.

Against hand-tuned baselines on a full H100 (external run, Llama-3.2-1B decode), MegaBake takes 1,784 µs/token. Hand-written CUDA graphs + PDL take 980 and HazyResearch's megakernel 966. Both are hand-written for this one model. With the real checkpoint on the 60-SM slice, the kernel reaches ~58% of read bandwidth. It loses most to decode attention that splits short contexts across CTAs, GEMMs that leave SMs idle, RoPE in its own barrier interval, and ~630 µs of host time per call; the LM head runs near peak. [docs/roadmap.md](docs/roadmap.md#done-external-baseline-llama-321b-2026-10-23) has the breakdown.

Reproduce on an SM90 device:

```bash
.venv/bin/python benchmarks/full_model.py --model HuggingFaceTB/SmolLM-135M HuggingFaceTB/SmolLM-1.7B
.venv/bin/python benchmarks/full_model.py --variants inductor megabake --modes decode   # a subset
MEGABAKE_PHASE_TIMER=1 .venv/bin/python tools/phase_profile.py --mode decode            # time per barrier interval
.venv/bin/python tools/dump_megakernels.py --checkpoint                                   # correctness check plus generated source
```

`benchmarks/run.py --workload mlp|transformer-block` and `benchmarks/isolated.py` measure the smaller fixtures.

## Current frontend

The Dynamo backend runs the pinned AOT, pre-grad, and post-grad FX transformations, retains the result as an FX `GraphModule`, and compiles it on SM90 when the complete plan is supported. It does not invoke Inductor lowering or code generation. It preserves the PyTorch wrappers around arguments, outputs, aliases, and mutations, and separates its capture cache from compiled-plan reuse.

```python
import torch
import megabake

model = model.eval()
compiled = torch.compile(model, backend=megabake.make_backend(), fullgraph=True)
with torch.no_grad():
    output = compiled(example_input)
```

The development CLI also exercises the capture path. CPU commands use the reference forward; CUDA commands follow the same backend policy and require a supported graph:

```bash
python src/main.py demo --device cpu
python src/main.py transformers --model-name HuggingFaceTB/SmolLM-135M
python src/main.py export --callable module:callable --arg '[[1.0, 2.0]]'
```

## Environment setup

The checked-in requirements target Python 3.13, PyTorch with CUDA 13.0, and CuTe DSL's CUDA 13 package. Use an NVIDIA GPU and driver supported by that PyTorch wheel. Kernel compilation needs the CUDA 13.0 development toolkit and its `nvcc`; installing Python requirements does not install `nvcc`. The CUDA version reported by `nvidia-smi` describes driver support, not the installed compiler toolkit.

From the repository root, create and activate a virtual environment, then install the project and its pinned direct dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Point `CUDA_HOME` at the CUDA 13.0 toolkit before compiling kernels, especially if `/usr/local/cuda` selects another version:

```bash
export CUDA_HOME=/path/to/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
```

Check the selected Python, PyTorch CUDA build, GPU visibility, and CUDA compiler:

```bash
python --version
python -c 'import torch; print("torch:", torch.__version__); print("CUDA runtime:", torch.version.cuda); print("GPU available:", torch.cuda.is_available())'
nvcc --version  # should report release 13.0
```

Activate the environment again in new shells with `source .venv/bin/activate`.

```bash
.venv/bin/python -m megabake.main transformers \
  --model-name HuggingFaceTB/SmolLM-135M \
  --device cuda \
  --dump-dir fx_traces/smollm \
  --dump-compute-graph \
  --dump-lowering-graph
```
