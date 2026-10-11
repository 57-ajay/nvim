# tinyVLLM.md — tinyvllm: PagedAttention and an LLM serving engine, from scratch

> **What this is:** the plan **and** the save file for `tinyvllm`, a small, readable LLM inference engine that rebuilds vLLM's core ideas from first principles. The centrepiece is **PagedAttention**: a paged KV-cache memory manager *and* the attention kernels that read it, written by you (PyTorch reference → Triton).
>
> **To any AI tutor or harness reading this:** this file is the single source of truth for the project. Before your first reply, read §0 Quick Status → §3 Laws → §4 How we work → the active phase in §9 → the latest §16 Session Log entry. Follow the same tutoring rules as `PROGRESS.md` §3 (one step per turn, predict before run, hint ladder, explain-back, ★ interview hooks).

| Field | Value |
|---|---|
| File version | v1 |
| Last updated | 2026-10-11 · plan written · next: Phase 0, step 0.1 |
| Companion files | `PROGRESS.md` (Zero-to-Hero tracker) · `INTERVIEWGUIDE.md` (gets a new Part R, §13) |
| Repo | `~/projects/py/tinyvllm` → GitHub `57Ajay/tinyvllm` (public) |
| Prior work it builds on | `karpathynn/py/l7.py`, `l9.py`, `model_runner.py` (GPT-2 + top-k/top-p sampling) · `karpathynn/nnrs/gpt2rs` (Rust GPT-2 with KV cache, HF logit parity 1.94e-7) · HF `57Ajay/tinystories-gpt-51m` |
| Reference codebases | nano-vllm (`GeeeekExplorer/nano-vllm`, ~1.2K LOC) · vLLM (`vllm-project/vllm`, V1 engine; read at commit `c85616e`, 2026-10-11) |

---

## 0. Quick Status — the save slot (update every session)

| | |
|---|---|
| Current phase | **Phase 0: Setup & motivation** |
| Current step | **0.1 Repo scaffold** |
| Next action | Create the repo, `pyproject.toml` with extras, CPU test run green |
| Milestones | M1 CPU engine `[ ]` · M2 Triton PagedAttention `[ ]` · M3 Serving + perf `[ ]` · M4 Results + write-up `[ ]` · M5 First vLLM PR `[ ]` |
| Phases | P0 `[ ]` · P1 `[ ]` · P2 `[ ]` · P3 `[ ]` · P4 `[ ]` · P5 `[ ]` · P6 `[ ]` · P7 `[ ]` · P8 `[ ]` · P9 `[ ]` · P10 `[ ]` · P11 `[ ]` · P12 `[ ]` |
| Sessions completed | 0 |
| Interview Part R | 0 / 30 answers written |

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done (code works, tests green, **and** you explained it back) · `[d]` deferred (reason in §16) · `[s]` stretch, optional

---

## 1. The pitch

**One line:** *tinyvllm is a from-scratch LLM inference engine with hand-written PagedAttention: a block-based KV cache with prefix caching and copy-on-write, a vLLM-V1-style continuous-batching scheduler, Triton paged-attention kernels, and an OpenAI-compatible streaming server. It is developed on a CPU and benchmarked on GPUs.*

**Why build it (your goals):**
1. **Understand vLLM** by rebuilding its two load-bearing ideas: *paged KV memory* (PagedAttention, Kwon et al., SOSP '23) and *iteration-level scheduling* (continuous batching, Orca, OSDI '22).
2. **A standout resume project.** Most "mini-vLLM" repos either call FlashAttention's paged kernel and never touch it, or skip paging entirely. You will write the kernel, prove it correct against an oracle, measure it against FlashAttention, and reproduce the paper's memory-waste result.
3. **A runway into vLLM contributions.** tinyvllm uses vLLM's names and structure (`BlockPool`, `KVCacheManager`, `SchedulerOutput`, `slot_mapping`, `query_start_loc`, `unified_attention`), so every file you write maps onto a vLLM file you can then read, test and patch (§12).

---

## 2. What makes tinyvllm different

| | nano-vllm | **tinyvllm (this plan)** | vLLM V1 (for reference) |
|---|---|---|---|
| PagedAttention **kernel** | Delegated to `flash-attn` (`flash_attn_with_kvcache(block_table=…)`); you never see it | **Written by you:** PyTorch oracle → Triton decode v1 → GQA-packed → split-KV v2 → unified varlen prefill+decode | FlashAttention / FlashInfer / Triton `kernel_unified_attention` (vLLM's own legacy CUDA `paged_attention_v1/v2` kernels are gone from `csrc/`) |
| KV block size | Locked to multiples of **256** (flash-attn paged-KV constraint) | Any size (8–256); **block-size ablation** is one of the experiments | `CacheConfig.DEFAULT_BLOCK_SIZE = 16` (a backend may override it) |
| Hardware | CUDA + flash-attn (Ampere or newer) only | **CPU** (dev + correctness) · **T4** (Turing, fp16, where flash-attn 2 cannot run) · **L4/A100** (bf16, headline numbers) | Many backends |
| Scheduler | Prefill-first, separate prefill/decode steps; chunked prefill only for the first sequence | **Unified token budget** `{req_id: num_tokens}`: mixed prefill+decode in one batch, chunked prefill for every request | Unified token budget |
| Preemption | Recompute | Recompute; swap-to-CPU as a stretch (the paper's other option) | Recompute |
| Prefix cache | xxhash chain; free list is a `deque` (O(n) `remove` on a cache hit) | Hash chain + **O(1) doubly linked LRU free queue**, tail-first freeing, hit-rate metrics | Same idea (`FreeKVCacheBlockQueue`) |
| Parallel sampling `n>1` | No | **Shared prompt blocks + copy-on-write** (paper §4.4) | Via prefix caching (`parallel_sampling.py`) |
| Sampling | Temperature only; **greedy is forbidden** (`assert temperature > 1e-10`) | Greedy, temperature, top-k, top-p, min-p, penalties, per-request seeds, logprobs, stop strings | Full |
| Models | Qwen3 | **GPT-2 family (including your TinyStories-51M)** + Llama/Qwen3 family (SmolLM2-135M, Qwen3-0.6B) | Hundreds |
| Serving | Offline `generate()` only | Offline + **OpenAI-compatible SSE server** + Prometheus `/metrics`; benchmarkable with `vllm bench serve` | Full |
| Tests | None in the repo | Kernel-vs-oracle (parametrized, as in vLLM's `tests/kernels`), block-manager invariants (hypothesis), HF logit parity, batch invariance, CPU CI | Huge suite |
| Experiments | Throughput table | Throughput **+ paper memory-waste reproduction + block-size ablation + kernel roofline + serving latency curves** | — |

**Design principle that ties it together: oracle-first development.** Every fast path has a slow, obviously correct twin, and a test that compares them:
`HF transformers` ⟷ `tinyvllm model` · `contiguous SDPA` ⟷ `torch_ref paged attention` ⟷ `Triton kernels` · `one request alone` ⟷ `same request in a busy batch` · `big KV pool` ⟷ `tiny pool with preemptions`.

---

## 3. The Laws (non-negotiable for this project)

1. **CPU-first, GPU-complete.** Everything is developed and tested on the CPU VM (8 cores, 32 GB, no GPU). Every module is device-agnostic, and nothing assumes CUDA unless it sits behind `device.type == "cuda"` or a `gpu` test marker. Speed claims come only from GPU runs.
2. **The learner writes the engine.** You write all engine logic: block pool, KV manager, scheduler, model runner, models, sampler, kernels, async engine. **The tutor supplies** plotting code, the Colab runner notebook, benchmark-harness plumbing (argparse, JSON dumping), HTTP/SSE boilerplate, and the GitHub Actions YAML.
3. **Oracle before optimization.** No kernel, cache or scheduler optimization lands without a test comparing it with the slow reference (§2).
4. **vLLM names.** Use vLLM's vocabulary for every concept (§6.3), so knowledge transfers 1:1.
5. **Numbers are earned.** Every number in the README or resume has a JSON file in `results/` and the exact command that produced it. No estimated or remembered numbers.
6. **Small, readable, typed.** Python 3.12, type hints, `@dataclass(slots=True)`, ruff. Target ≤ ~3,000 LOC of library code (excluding tests and benchmarks). Comments explain *why*, not *what*.
7. **Commit per step.** One commit per finished step (`P2.4: torch_ref paged attention + tests`). The learner pushes; the tutor only pulls (same as `PROGRESS.md` Law 10).
8. **Update the interview bank as you go.** Every step's ★ hook becomes an answer in `INTERVIEWGUIDE.md` Part R once the code works.

---

## 4. How we work (tutor protocol, same spirit as PROGRESS.md §3)

**Step message format:**
```
P<phase>.<step> — <title>                          [CPU] / [GPU] / [CPU→GPU]
Concept:    intuition first, then shapes/data structures (3–8 lines)
Your task:  what to write, in which file, with the function signature
Test:       the test that proves it (you write it, or the tutor gives the skeleton)
Expected:   values/shapes/output to check against
Checkpoint: paste back test output + ONE question (predict / explain-back)
vLLM link:  the matching vLLM file/function to skim afterwards
Interview hook: ★ <question this unlocks>
```

**Commands:** `resume tinyvllm` (start: read §0, recap, ≤3 warm-up questions) · `that's it for today` (update §0, tick checklists, add a §16 log entry, update the §16 Results Board, write Part R answers, bump version) · `hint` (H1 nudge → H2 pseudocode → H3 code with blanks → H4 full solution) · `deeper` · `review my code` · `quiz me` · `interview mode` · `park it` · `status`.

**GPU session protocol:** the tutor gives a ready-to-run Colab cell block. You run it and paste back `nvidia-smi`, the pytest summary and `results/*.json`. Keep each GPU session short: Colab disconnects, and GCP bills by the minute.

---

## 5. Hardware & environment plan

### 5.1 Where each thing runs

| Machine | GPU arch | dtype | What we use it for | Attention backends |
|---|---|---|---|---|
| **VM (CPU)**, 8 cores / 32 GB | — | fp32 (bf16 optional) | All development; all correctness tests; Triton kernels under `TRITON_INTERPRET=1`; serving demo with SmolLM2-135M or your TinyStories-51M | `torch_ref`, `triton` (interpreted, tiny shapes only) |
| **Colab T4** (free) | Turing sm_75, 16 GB, ~320 GB/s | **fp16** (no bf16, no flash-attn 2) | First real Triton runs, GPU test suite, small benchmarks. A selling point: tinyvllm runs on T4, where nano-vllm (flash-attn 2) can't | `torch_ref`, `triton` |
| **GCP L4** (g2-standard, spot) | Ada sm_89, 24 GB, ~300 GB/s | bf16 | Headline benchmarks vs nano-vllm, vLLM and HF; FlashAttention comparison; serving benchmark | `torch_ref`, `triton`, `flash_attn` (comparison only) |
| A100 (Colab Pro / GCP), optional | Ampere sm_80, 40/80 GB, ~1.5–2.0 TB/s | bf16 | One "big GPU" data point; kernel roofline | same |

GCP tips: use a Deep Learning VM image (drivers preinstalled), use spot instances, `gcloud compute instances stop` the moment a run finishes, and check current pricing before you start.

### 5.2 Triton on a CPU-only machine
- `TRITON_INTERPRET=1` runs `@triton.jit` kernels with a NumPy-based interpreter on CPU tensors. It is slow and not representative of speed, but it executes the **same kernel source**, so you can write, step through (with `print`) and unit-test kernels on the VM. Use tiny shapes (e.g. 2 sequences, head_dim 16 or 32, block_size 4 or 16).
- The CPU torch wheel does not pull in Triton; install it explicitly (`uv add --optional gpu triton`, plus a CPU dev group that also includes it).
- Some features behave differently or are unsupported in the interpreter (no performance meaning, limited autotune). Keep kernels simple, and verify every kernel on a real GPU before claiming it works.

### 5.3 Repo bootstrap (step 0.1 does this together)
```bash
mkdir -p ~/projects/py/tinyvllm && cd ~/projects/py/tinyvllm
uv init --package --python 3.12 .
# CPU torch from the explicit CPU index (same trick as PROGRESS.md §4.3).
# pip ignores [tool.uv.*], so `pip install -e .[gpu]` on Colab keeps Colab's CUDA torch.
uv add torch numpy safetensors tokenizers tiktoken huggingface-hub
uv add --optional gpu triton
uv add --optional serve fastapi uvicorn
uv add --dev pytest hypothesis transformers ruff matplotlib triton
```
`pyproject.toml` gets: pytest markers `gpu` (auto-skip when CUDA is missing) and `slow`; a ruff config; and a console script `tinyvllm = "tinyvllm.entrypoints.cli:main"`.

**Verify (paste to the tutor):**
```bash
uv run python -c "import torch, triton; print(torch.__version__, torch.cuda.is_available(), triton.__version__)"
TRITON_INTERPRET=1 uv run pytest -q
```

### 5.4 CI
GitHub Actions, CPU-only: `ruff check` + `pytest -m "not gpu"` (including the interpreted Triton tests at tiny shapes). A green badge on a GPU project developed without a GPU is a story in itself.

---

## 6. Architecture

### 6.1 The big picture
```
              ┌──────────────────────────── entrypoints ────────────────────────────┐
 prompts ───► │  LLM.generate()  (offline batch)       OpenAI server (/v1/*, SSE)   │
              └───────────────┬──────────────────────────────────┬──────────────────┘
                              ▼                                  ▼
              ┌──────────────── LLMEngine  /  AsyncLLM (background loop) ────────────┐
              │ tokenize → Request → scheduler.add_request()                         │
              │ while work:  out    = scheduler.schedule()          # what to run    │
              │              result = model_runner.execute(out)     # run it         │
              │              scheduler.update_from_output(out, result) → detokenize  │
              └───────────────┬──────────────────────────────────┬───────────────────┘
                              ▼                                  ▼
      ┌───────────── Scheduler (CPU) ─────────────┐   ┌──────────── ModelRunner (device) ───────────┐
      │ token budget · chunked prefill · preempt  │   │ SchedulerOutput → flat batch tensors        │
      │   └─ KVCacheManager                       │   │   input_ids, positions, slot_mapping,       │
      │        allocate_slots / free / prefix hit │   │   block_table, query_start_loc, seq_lens    │
      │        └─ BlockPool                       │   │ model.forward(...)                          │
      │             blocks[], FreeBlockQueue(LRU),│   │   └─ Attention backend                      │
      │             hash → block (prefix cache)   │   │        torch_ref │ triton │ (flash_attn)    │
      └───────────────────────────────────────────┘   │ Sampler → token ids (one sync per step)     │
          bookkeeping only: block IDs, no tensors     └─────────────────────────────────────────────┘
                                                        owns the KV cache tensors
```
The split matters: **the scheduler and KV manager never touch tensors.** They move block *IDs* around. The model runner turns those IDs into index tensors, and the kernels follow them. That's the page-table idea from operating systems.

### 6.2 KV cache layout & memory math
- Per layer: `kv_cache[layer]` has shape `[2, num_blocks, block_size, num_kv_heads, head_dim]` (K and V; "NHD" token-major layout, like vLLM's flash layout).
- **Bytes per token** = `2 × n_layers × n_kv_heads × head_dim × dtype_bytes`.
- `num_blocks = kv_budget_bytes // (block_size × bytes_per_token)`. On GPU the budget comes from profiling (`gpu_memory_utilization`); on CPU it's a config value (`kv_cache_memory_gb`).

| Model (verify each against its `config.json`) | Layers | KV heads × head_dim | dtype | Bytes/token | 1,024-token sequence |
|---|---|---|---|---|---|
| GPT-2 124M | 12 | 12 × 64 | fp16 | 36,864 (36 KiB) | 36 MiB |
| SmolLM2-135M (GQA 9:3) | 30 | 3 × 64 | fp32 (CPU) | 46,080 (45 KiB) | 45 MiB |
| Qwen3-0.6B (GQA 16:8) | 28 | 8 × 128 | bf16 | 114,688 (112 KiB) | 112 MiB |
| Your TinyStories-51M | ? | ? | ? | **exercise in P2.1** | ? |

Fragmentation, the reason paging exists: a system that reserves `max_model_len` slots per request wastes `1 − actual_len/max_len` of its reservation (*internal* fragmentation, plus *external* fragmentation between differently sized reservations). The paper reports that only **20.4–38.2%** of KV memory held actual token states in the systems it measured. With paging, waste is at most `block_size − 1` slots per request, at the tail of its last block.

### 6.3 Vocabulary (vLLM names; use them everywhere)

| Term | Meaning |
|---|---|
| `block_size` | Tokens per KV block (page). Default 16. |
| `KVCacheBlock` | `block_id`, `ref_cnt`, `block_hash` (set when full and cached), free-list prev/next pointers |
| `block_table` | Per request: list of physical `block_id`s; logical block *i* → physical block `block_table[i]` |
| `slot_mapping` | Per **token** in the batch: the flat slot `block_table[pos // B] * B + pos % B` where its K/V gets written; `-1` = padding, skip |
| `num_computed_tokens` | How many of a request's tokens already have K/V in the cache. **The single source of truth for progress.** |
| `num_scheduled_tokens` | `{req_id: n}` for this step. Prefill chunk, decode (1), or spec tokens; the scheduler doesn't care which. |
| `query_start_loc` | Prefix sums of per-request query lengths (a.k.a. `cu_seqlens_q`), shape `[num_reqs + 1]` |
| `seq_lens` | Per request: KV length *after* this step = `num_computed_tokens + num_scheduled_tokens` |
| `logits_indices` | Which flat token rows need logits (the last token of each request that will sample) |
| null block | Block 0 is reserved and never handed out; padding entries point to it |

### 6.4 Worked example of one flat batch (block_size B = 16)
Three requests run in the **same** forward pass:
- **A** is decoding: 36 tokens already cached, feeding token #37 → 1 query token at position 36.
- **B** is a new prompt of 5 tokens, prefilled fully.
- **C** is a 36-token prompt whose first 32 tokens hit the prefix cache → compute 4 tokens at positions 32–35.

```
input_ids        = [a36 | b0 b1 b2 b3 b4 | c32 c33 c34 c35]          # N = 10 tokens, no padding, no batch dim
positions        = [ 36 |  0  1  2  3  4 |  32  33  34  35]
query_start_loc  = [0, 1, 6, 10]
seq_lens         = [37, 5, 36]                                       # KV length after this step
block_table      = A:[7, 2, 9]   B:[4]   C:[11, 12, 5]               # padded to a [3, 3] int32 tensor
slot_mapping     = [9*16+4=148 | 64 65 66 67 68 | 80 81 82 83]
logits_indices   = [0, 5, 9]                                         # last token of each request
```
If you can produce these tensors by hand and in code, you understand the engine. (This exact example becomes a unit test in P3.4.)

### 6.5 Invariants (asserted in debug mode, property-tested in P2)
1. `sum(block.ref_cnt) == total number of (request, block) references`; the free queue holds exactly the blocks with `ref_cnt == 0` (except the null block).
2. A block becomes immutable once it is full and hashed. New K/V is written only into a request's **last** block, so copy-on-write only ever applies to that block.
3. The scheduler never schedules a request without its slots being allocated, and never exceeds `max_num_batched_tokens` or `max_num_seqs`.
4. Positions, slots and which tokens to feed are all derived from `num_computed_tokens`; there is no separate `is_prefill` flag that can drift.
5. Greedy outputs on CPU fp32 are identical whether a request runs alone or in a busy batch, and with or without preemptions or prefix hits.

### 6.6 Repo layout (target)
```
tinyvllm/
├── pyproject.toml · README.md · tinyVLLM.md (this file) · .github/workflows/ci.yml
├── tinyvllm/
│   ├── __init__.py                 # LLM, SamplingParams
│   ├── config.py                   # ModelConfig, CacheConfig, SchedulerConfig, EngineConfig
│   ├── sampling_params.py · request.py · outputs.py
│   ├── core/
│   │   ├── block_pool.py           # KVCacheBlock, FreeBlockQueue, BlockPool (+ prefix-hash map)
│   │   ├── kv_cache_manager.py     # allocate_slots, get_computed_blocks, free, fork (CoW)
│   │   ├── hashing.py              # block hash chain
│   │   └── scheduler.py            # unified token-budget scheduler → SchedulerOutput
│   ├── worker/
│   │   ├── model_runner.py         # flat batch build, forward, sample; profile_run
│   │   ├── input_batch.py          # persistent batch (P8)
│   │   └── cuda_graphs.py          # (P8)
│   ├── attention/
│   │   ├── metadata.py             # AttentionMetadata
│   │   ├── backends/               # torch_ref.py, triton.py, flash_attn.py (comparison only)
│   │   └── ops/                    # reshape_and_cache.py, paged_decode.py, split_kv.py, unified_attention.py
│   ├── models/                     # gpt2.py, llama.py (Llama/Qwen3/SmolLM2), registry.py, loader.py
│   ├── layers/                     # rotary.py, norms.py, sampler.py
│   ├── engine/                     # llm_engine.py, async_llm.py, detokenizer.py, metrics.py
│   ├── entrypoints/                # llm.py, openai_server.py, cli.py
│   └── utils/                      # device.py, timing.py
├── tests/                          # unit + kernel + parity + e2e (markers: gpu, slow)
├── benchmarks/                     # bench_naive, bench_throughput, bench_kernels, bench_serving, bench_memory_waste
├── notebooks/colab_runner.ipynb    # tutor-provided
├── results/                        # JSON from every benchmark run (Law 5)
└── docs/                           # design.md, paged_attention.md, notes_paper.md, results.md
```

### 6.7 Public API (what the README will show)
```python
from tinyvllm import LLM, SamplingParams

llm = LLM("HuggingFaceTB/SmolLM2-135M", device="cpu", block_size=16,
          kv_cache_memory_gb=1.0, attn_backend="torch_ref", enable_prefix_caching=True)
outs = llm.generate(["The capital of France is", "Once upon a time"],
                    SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64, n=2, seed=0))
```
```bash
tinyvllm serve Qwen/Qwen3-0.6B --attn-backend triton --port 8000
curl localhost:8000/v1/completions -d '{"model":"qwen3","prompt":"Hello","stream":true}'
```

---

## 7. Milestones

| Milestone | Phases | Where | "Done" means | What it's worth |
|---|---|---|---|---|
| **M1: CPU engine** | P0–P5 | VM | Paged KV cache + continuous batching + chunked prefill + preemption + full sampling + prefix caching + CoW, all oracle-tested on CPU | Interview-ready: you can whiteboard vLLM's core and show working code |
| **M2: Triton PagedAttention** | P6–P7 | VM (interpreter) → Colab → L4 | Your kernels pass the oracle tests on GPU; benchmarked vs torch_ref and FlashAttention; roofline and block-size plots | The standout resume line |
| **M3: Serving + perf** | P8–P9 | VM + GPU | OpenAI-compatible streaming server, metrics, CUDA graphs, serving benchmark | "Production-shaped" |
| **M4: Results + write-up** | P10 | L4 | README results tables + plots, design doc, blog post, resume bullets with real numbers | Portfolio-ready |
| **M5: Upstream** | P12 (in parallel from M1 onward) | any | A merged (or under review) vLLM PR | "vLLM contributor" |

If an interview is only days away, **M1 is the cut-off**: an engine you can explain end to end beats half-finished kernels. M2 is the differentiator to finish right after.

---

## 8. Phase map (one line each)

| Phase | Name | Where | Key output |
|---|---|---|---|
| P0 | Setup & motivation | CPU | Repo, CI, naive-generation baseline numbers |
| P1 | Models in flat-batch form | CPU | GPT-2 + Llama-family models, HF logit parity |
| P2 | **PagedAttention memory system** | CPU | BlockPool, slot mapping, `torch_ref` paged attention, KVCacheManager |
| P3 | **Continuous batching** | CPU | Unified scheduler, chunked prefill, preemption, ModelRunner, `LLM.generate` |
| P4 | Sampling, stopping, parallel sampling | CPU | Vectorized sampler, stop strings, `n>1` with copy-on-write |
| P5 | Automatic prefix caching | CPU | Hash chain, LRU eviction, hit-rate metrics → **M1** |
| P6 | GPU bring-up | Colab/GCP | Colab runner, memory profiling, GPU baseline |
| P7 | **PagedAttention kernels in Triton** | CPU→GPU | reshape_and_cache, decode v1, GQA packing, split-KV v2, unified varlen → **M2** |
| P8 | Performance engineering | GPU | Profiling, CUDA graphs, persistent batch, compile |
| P9 | Serving | CPU→GPU | AsyncLLM, OpenAI server, metrics, serving bench → **M3** |
| P10 | Results & write-up | GPU | Benchmarks, plots, README, blog → **M4** |
| P11 | Stretch menu | any | Spec decoding, TP, KV quantization, swap, … |
| P12 | vLLM contribution track | any | Read-map, first PRs → **M5** |

---

## 9. Phases & steps

> Each step: tag · what you build · **Done when** · ★ interview hook. A step may take several turns. Tick `[x]` only after the code works, the tests pass, and you've explained it back.

### P0: Setup & motivation · `[ ]`
- [ ] **P0.1 Repo scaffold** `[CPU]`: §5.3 bootstrap, package skeleton, `gpu`/`slow` markers, ruff. **Done when:** `uv run pytest` passes (1 smoke test) and the verify command prints versions.
- [ ] **P0.2 Device & dtype policy** `[CPU]`: `utils/device.py` with `pick_device()`, `pick_dtype(device)` (CPU→fp32, sm_75→fp16, ≥sm_80→bf16), `triton_available()`, `is_interpreter()`. CPU CI workflow (tutor gives the YAML). **Done when:** CI is green on GitHub.
- [ ] **P0.3 Read the paper** `[CPU]`: Kwon et al. 2023 §1–4 + §6 (eval) and the vLLM V1 blog post. Write `docs/notes_paper.md` (1 page, your words): the 3 kinds of KV waste, the block table, CoW, recompute vs swap. **Done when:** you can explain Fig. 2 (memory waste) without looking. ★ *What problem does PagedAttention solve?*
- [ ] **P0.4 Motivation baseline** `[CPU]`: `benchmarks/bench_naive.py` uses HF `model.generate` with static padded batches on SmolLM2-135M and your TinyStories-51M: tok/s, plus KV slots reserved vs actually used. **Done when:** "before" numbers are in `results/` and on the §16 Results Board. ★ *Static vs continuous batching: what does static batching waste?*

### P1: Models in flat-batch form · `[ ]`
- [ ] **P1.1 Model contract** `[CPU]`: `forward(input_ids[N], positions[N], attn_metadata) -> hidden[N, d]`, plus `compute_logits(hidden[logits_indices])`. A packed batch has **no batch dimension and no padding**. Bring-up backend: per-request SDPA over the packed rows, no cache. **Done when:** a packed batch of 3 prompts gives the same last-token logits as 3 separate forwards. ★ *Why do inference engines flatten the batch instead of padding to `[B, T]`?*
- [ ] **P1.2 GPT-2 family** `[CPU]`: load HF `gpt2` (safetensors; Conv1D weights transposed; tied `wte`/`lm_head`) and **your TinyStories-51M** `.pt` (`{"config","model"}` checkpoint; vocab padded to 50304, so mask logits ≥ 50257; tiktoken `gpt2`; prompts start with `<|endoftext|>`, as in your `model_runner.py`). Learned `wpe` ⇒ `max_model_len = 1024`. **Done when:** both models load and generate sensible text greedily.
- [ ] **P1.3 Llama family** `[CPU]`: RMSNorm, RoPE applied **from the `positions` tensor** (essential for chunked prefill and prefix hits), GQA (`num_kv_heads < num_heads`), SwiGLU, tied embeddings, Qwen3's per-head q/k RMSNorm. Models: SmolLM2-135M (CPU), Qwen3-0.6B (GPU benchmarks, and apples-to-apples with nano-vllm). ★ *How does GQA shrink the KV cache, and by how much for Qwen3-0.6B?*
- [ ] **P1.4 Weight loader** `[CPU]`: streaming safetensors, a name-mapping table, packed `qkv_proj` / `gate_up_proj` (one matmul instead of 3 or 2). **Done when:** no unexpected or missing keys, asserted.
- [ ] **P1.5 HF parity harness** `[CPU]`: `tests/test_parity_hf.py`: max |logit diff| vs transformers ≤ 1e-4 in fp32 on all 3 models, and a 32-token greedy continuation that is token-identical. **Done when:** green. (gpt2rs got 1.94e-7, so hold the same standard.)

### P2: The PagedAttention memory system (heart #1) · `[ ]`
- [ ] **P2.1 KV layout & memory math** `[CPU]`: allocate `[2, num_blocks, block_size, H_kv, D]` per layer from `kv_cache_memory_gb`. Exercise: fill in the §6.2 table row for your TinyStories model; predict `num_blocks` for 1 GB, then check. ★ *Derive the KV cache size formula. KV memory for 1,000 concurrent 2K-token Qwen3-0.6B requests?*
- [ ] **P2.2 BlockPool** `[CPU]`: `KVCacheBlock`, `FreeBlockQueue` (a hand-rolled doubly linked list: O(1) `popleft`, `append`, and **`remove` from the middle**), null block 0, `get_new_blocks(n)`, `free_blocks(blocks)`. Tests: hypothesis-driven random alloc/free/fork sequences that check the §6.5 invariants. **Done when:** 1,000 random programs pass. ★ *Why a hand-rolled linked list instead of `collections.deque`?*
- [ ] **P2.3 Write path: slot mapping + `reshape_and_cache`** `[CPU]`: compute `slot_mapping` from block tables and positions; write `k`/`v` with `kv[0].view(-1, H_kv, D)[slot_mapping] = k` (skip `-1`). **Done when:** a test writes known values and reads them back through the block table. ★ *What is `slot_mapping`, and why is it per token rather than per request?*
- [ ] **P2.4 Read path: `torch_ref` paged attention** `[CPU]`: for each request, gather its blocks via `block_table`, cut to `seq_len`, then run causal attention with **offset** (query *i* sees keys `≤ seq_len − q_len + i`) and GQA by grouping heads. Tests vs contiguous SDPA with **shuffled physical block IDs**, mixed query lengths (decode = 1, prefill chunks, prefix hits), and GQA ratios 1/2/3. (Compare with vLLM's `ref_paged_attn` in `tests/kernels/attention/test_triton_unified_attention.py` *after* writing yours.) **Done when:** max abs diff < 1e-5 in fp32. ★ *Why doesn't a non-contiguous block layout change the attention result?*
- [ ] **P2.5 KVCacheManager** `[CPU]`: `allocate_slots(req, num_new_tokens) -> list[KVCacheBlock] | None` (all-or-nothing), `free(req)`, `usage()`, `num_free_blocks`. **Done when:** unit tests cover block-boundary cases (token 16 → 17 allocates exactly one new block; 15 → 16 allocates none).
- [ ] **P2.6 End to end, one request** `[CPU]`: generate with the paged cache (prefill once, then decode token by token reading from blocks). **Done when:** greedy output == the P1 recompute output, and tok/s on CPU is measurably higher than recompute. ★ *Prefill vs decode: why is one compute-bound and the other memory-bound?*

### P3: Continuous batching (heart #2) · `[ ]`
- [ ] **P3.1 Request & lifecycle** `[CPU]`: `Request` (prompt/output token ids, `num_computed_tokens`, sampling params, arrival time), statuses `WAITING → RUNNING → FINISHED_{STOPPED, LENGTH, ABORTED}` and `PREEMPTED`. ★ *Why make `num_computed_tokens` the only progress state?*
- [ ] **P3.2 Unified token-budget scheduler** `[CPU]`: each `schedule()` (1) walks **running** requests first and gives each `min(remaining, budget)` tokens (1 for decodes, the rest of the prompt for in-flight prefills), then (2) admits **waiting** requests while there's token budget, `max_num_seqs` room and free blocks. Emits `SchedulerOutput{scheduled_new_reqs, scheduled_cached_reqs, num_scheduled_tokens: dict[str, int], total_num_scheduled_tokens, finished_req_ids}`. Chunked prefill is then just "the budget ran out mid-prompt". Tests use a fake model. ★ *What is chunked prefill, and how does it trade TTFT against inter-token latency?* ★ *Continuous (iteration-level) batching vs static batching.*
- [ ] **P3.3 Preemption by recompute** `[CPU]`: if `allocate_slots` fails for a running request, preempt the **most recently arrived** running request: free its blocks, set `num_computed_tokens = 0`, and put it at the front of `waiting`. Repeat until it fits, or preempt self. **Done when:** a test with a tiny pool forces preemptions, and the final greedy outputs are identical to a run with a big pool. ★ *Recompute vs swap preemption: when does each win?*
- [ ] **P3.4 ModelRunner** `[CPU]`: `SchedulerOutput` → `input_ids, positions, slot_mapping, block_table, query_start_loc, seq_lens, logits_indices` (§6.4 is the first test case) → forward → sample greedy for requests whose prompt is now complete; partial-prefill rows are discarded. **Done when:** the §6.4 example test passes and a mixed batch runs.
- [ ] **P3.5 Engine + offline API** `[CPU]`: `LLMEngine.step()`, `LLM.generate(prompts, params)` with outputs in input order. **Batch-invariance test:** 32 random prompts with random arrival → outputs identical to running each alone (fp32 CPU). ★ *Why can batched GPU inference differ from unbatched in bf16 even with greedy decoding?* (batch-dependent reduction order)
- [ ] **P3.6 Scheduler timeline viz** `[CPU]`: log per-step `{prefill_tokens, decode_tokens, num_running, num_waiting, kv_usage, preemptions}`; the tutor provides the stacked-area plot code. **Done when:** `plots/scheduler_timeline.png` shows chunked prefill interleaving with decodes (README GIF candidate).

### P4: Sampling, stopping & parallel sampling · `[ ]`
- [ ] **P4.1 Batched sampler** `[CPU]`: per-row params as tensors: greedy rows (`temperature == 0`), temperature, top-k, top-p, min-p, all vectorized (generalize your `model_runner.py` loop to per-row params). Per-request seeded `torch.Generator`, so a seeded request reproduces regardless of its batch-mates. ★ *top-k vs top-p vs min-p; why temperature before the truncation filters?*
- [ ] **P4.2 Penalties & logprobs** `[CPU]`: repetition / frequency / presence penalties from per-request output-token counts; logit bias; return the sampled-token logprob + top-n logprobs. ★ *How are frequency and presence penalties different?*
- [ ] **P4.3 Stopping & incremental detokenization** `[CPU]`: EOS (respect `ignore_eos`), `max_tokens`, `stop_token_ids`, **stop strings** (hold back text that could be the start of a stop string; handle multi-byte UTF-8 split across tokens, as in L8). **Done when:** streaming output concatenates to exactly the non-streaming output.
- [ ] **P4.4 `n > 1` with copy-on-write** `[CPU]`: prefill once, then fork into `n` children sharing every prompt block (`ref_cnt += 1`). Before a child writes into a shared, partially filled last block, copy that block (`copy_blocks` op) → private. Measure KV blocks used vs `n` independent requests. **Done when:** outputs equal `n` independent seeded runs, with fewer blocks used. ★ *How does PagedAttention make parallel sampling and beam search cheap?* (vLLM V1 now gets the same effect through prefix caching; know both.)

### P5: Automatic prefix caching · `[ ]`
- [ ] **P5.1 Block hash chain** `[CPU]`: `hash_i = H(hash_{i−1}, tuple(block_tokens), extra_keys)` for **full** blocks only. Because of the chain, one hash identifies the entire prefix. Decide on Python `hash` vs `hashlib.sha256` / xxhash (collision risk vs speed; vLLM offers sha256). ★ *Why hash full blocks only, and why chain the hashes?*
- [ ] **P5.2 Cache lookup on admission** `[CPU]`: find the longest cached prefix → `num_computed_tokens` jumps ahead. Never cache-hit the **whole** prompt (at least one token must run to produce logits). A hit on a free-but-cached block pulls it out of the free queue in O(1) (this is where P2.2's linked list pays off). **Done when:** a second identical prompt computes only its last token.
- [ ] **P5.3 Eviction & free order** `[CPU]`: freed blocks keep their hash and go to the queue tail (LRU). Free a request's blocks **tail-first**, so prefix heads survive longest. Reusing a block evicts its hash. Tests: eviction under pressure, outputs unchanged with caching on vs off.
- [ ] **P5.4 Prefix-cache metrics & bench** `[CPU]`: hit rate, prefill tokens saved, TTFT with vs without caching on a shared-system-prompt workload. **Done when:** numbers are in `results/`. ★ *Hash-based block cache (vLLM) vs radix tree (SGLang RadixAttention).* → **M1 complete.**

### P6: GPU bring-up · `[ ]`
- [ ] **P6.1 Colab runner** `[GPU]`: the tutor provides `notebooks/colab_runner.ipynb`: clone, `pip install -e .[gpu,dev]`, `nvidia-smi`, `pytest -m gpu`, benchmarks, then save `results/*.json` (download or copy to Drive). **Done when:** the full test suite passes on T4 with fp16.
- [ ] **P6.2 GPU memory profiling** `[GPU]`: `profile_run()` runs a dummy forward at `max_num_batched_tokens`, reads peak `torch.cuda.max_memory_allocated()`, then sets `kv_budget = total × gpu_memory_utilization − weights − peak_activations` → `num_blocks`. ★ *How does vLLM decide how many KV blocks to allocate?*
- [ ] **P6.3 Host↔device hygiene** `[GPU]`: pinned staging buffers + `non_blocking=True`; exactly **one** device→host sync per step (the sampled ids); CUDA-event timing. **Done when:** the profiler shows no stray syncs. ★ *Why is `.item()` inside a decode loop expensive?*
- [ ] **P6.4 GPU baseline** `[GPU]`: throughput with the `torch_ref` backend on T4 and L4; profile it and show that the gather copies in attention dominate. This motivates P7.

### P7: PagedAttention kernels in Triton (the standout) · `[ ]`
> **Kernel workflow:** write on the VM → `TRITON_INTERPRET=1 uv run pytest tests/kernels -m "not slow"` (tiny shapes) → push → Colab/L4: `pytest -m gpu tests/kernels` + `benchmarks/bench_kernels.py`. Every kernel is tested against `torch_ref` (P2.4) over a parametrized grid: sequence lengths 1…4K, shuffled block tables, block sizes {16, 32}, head_dim {64, 128}, GQA ratios {1, 2, 3, 8}, dtypes {fp32 (interp), fp16, bf16 (sm_80+)}.

- [ ] **P7.0 Triton warm-up** `[CPU→GPU]`: vector add → fused row softmax (one program per row), run in the interpreter first, then on T4. Concepts: `program_id`, `tl.arange`, masks, `tl.load`/`tl.store`, `constexpr`, `num_warps`, why powers of two. ★ *GPU execution model: grid, programs/blocks, warps, SRAM vs HBM.*
- [ ] **P7.1 `reshape_and_cache` kernel** `[CPU→GPU]`: one program per token: read `slot_mapping[i]`, return early if it's `-1`, store its K and V rows into the paged cache. **Done when:** bit-exact vs the torch version.
- [ ] **P7.2 Paged decode v1** `[CPU→GPU]`: grid `(num_seqs, num_q_heads)`. Each program loads its query vector, then loops over the context **one KV block at a time**: physical id = `block_table[seq, j]`; load the K tile `[block_size, D]`; scores; **online softmax** (running max `m`, running sum `l`, rescaled accumulator; the same recurrence as FlashAttention in L9); load the V tile and accumulate. **Done when:** it passes the full parametrized grid vs `torch_ref`. ★ *Walk through the online-softmax recurrence; why is it exact, not approximate?*
- [ ] **P7.3 GQA packing** `[GPU]`: grid `(num_seqs, num_kv_heads)`. One program serves all `num_q_per_kv` query heads of a group, so each K/V tile is loaded **once** and the score computation becomes a small `tl.dot`. Measure the speedup for SmolLM2 (3:1) and Qwen3 (2:1). ★ *Why is decode attention memory-bound, and how does GQA packing reduce the bytes moved?*
- [ ] **P7.4 Split-KV (PagedAttention v2 / Flash-Decoding)** `[GPU]`: add a segment axis to the grid. Each program handles one context segment and writes partial `(m, l, acc)`; a second `reduce_segments` kernel merges them with log-sum-exp rescaling. Produce a heatmap of batch × context length showing which version wins, then choose the switch threshold (vLLM's `seq_threshold_3D` is the same idea). ★ *Why does split-KV help at small batch and long context, and when does it hurt?*
- [ ] **P7.5 Unified varlen kernel** `[CPU→GPU]`: **one launch for the whole mixed batch** (prefill chunks, decodes, prefix-cache hits). Programs iterate over (query tile, KV head) using `query_start_loc`, with a causal mask offset by `seq_len − query_len`, reading all K/V from the paged cache. This is the design of vLLM's `kernel_unified_attention` (`vllm/v1/attention/ops/triton_unified_attention.py`), so read it **after** yours works. **Done when:** it replaces every other attention path in tinyvllm and the e2e tests still pass on GPU.
- [ ] **P7.6 Backend registry** `[CPU]`: `attn_backend ∈ {torch_ref, triton, flash_attn}`, also settable with `TINYVLLM_ATTN_BACKEND`. Fall back to `torch_ref` on CPU unless the interpreter is on. `flash_attn` exists only for benchmark comparison (it needs block_size 256 and sm_80+).
- [ ] **P7.7 Kernel benchmarks & roofline** `[GPU]`: latency of `torch_ref` vs your kernels vs FlashAttention's `flash_attn_with_kvcache` (L4/A100; optionally FlashInfer). Compute **achieved HBM bandwidth** = bytes of K/V (+ Q, O) read and written ÷ time, against the GPU's peak (T4 ≈ 320 GB/s, L4 ≈ 300 GB/s, A100 ≈ 1.5–2.0 TB/s). **Done when:** a roofline plot and a table are in `results/`. ★ *What is arithmetic intensity, and what's decode attention's?*
- [ ] **P7.8 Block-size ablation** `[GPU]`: block_size ∈ {8, 16, 32, 64, 128, 256} → kernel latency vs internal-fragmentation waste on a realistic length distribution. Reproduce the paper's block-size trade-off and justify your default. ★ *What does block size trade off?* → **M2 complete.**

### P8: Performance engineering · `[ ]`
- [ ] **P8.1 Profile end to end** `[GPU]`: `torch.profiler` → Chrome trace. Find GPU idle gaps (CPU overhead: Python scheduling, tensor building). **Done when:** you can state the CPU-overhead share of a decode step.
- [ ] **P8.2 CUDA graphs for decode** `[GPU]`: capture decode-only steps for bucketed batch sizes `[1, 2, 4, 8, 16, 32, …]` with static input buffers. Pad `slot_mapping` with `-1` and pad block tables; share one memory pool across graphs. Mixed prefill batches stay eager. ★ *What do CUDA graphs remove, and what constraints do they impose?*
- [ ] **P8.3 Persistent batch** `[GPU]`: keep `token_ids` and `block_table` as device tensors updated by diffs each step (added or removed requests, appended blocks) instead of rebuilding from Python lists (vLLM: `gpu_input_batch.py`, `block_table.py`).
- [ ] **P8.4 `torch.compile`** `[s]` `[GPU]`: compile the non-attention parts; register attention as a `torch.library.custom_op` so the graph doesn't break (vLLM's piecewise-compile idea).
- [ ] **P8.5 Async scheduling** `[s]` `[GPU]`: overlap scheduling step *t + 1* on the CPU with the GPU running step *t*.

### P9: Serving · `[ ]`
- [ ] **P9.1 AsyncLLM** `[CPU]`: the engine loop runs in a background thread/asyncio task; each request gets an output queue; abort on client disconnect; bounded admission (backpressure). ★ *Why does vLLM run the engine core in a separate process?*
- [ ] **P9.2 OpenAI-compatible server** `[CPU]`: FastAPI glue from the tutor; you write the engine integration. Endpoints: `/v1/completions`, `/v1/chat/completions` (tokenizer chat template), `stream: true` via SSE, `/v1/models`, `/health`. **Done when:** the `openai` Python client works against it unchanged.
- [ ] **P9.3 Metrics** `[CPU]`: `/metrics` in Prometheus text format: requests running and waiting, KV usage %, prefix hit rate, preemptions, and TTFT / ITL / end-to-end latency histograms. ★ *Define TTFT, TPOT/ITL, throughput and goodput. Which do users feel?*
- [ ] **P9.4 Serving benchmark** `[GPU]`: Poisson arrivals over a QPS sweep, reporting TTFT p50/p99, ITL, throughput. Because the server is OpenAI-compatible, **run vLLM's own client (`vllm bench serve`) against both tinyvllm and vLLM** on the same L4. → **M3 complete.**

### P10: Results & write-up · `[ ]`
- [ ] **P10.1 Offline throughput** `[GPU]`: nano-vllm's exact protocol (Qwen3-0.6B, 256 requests, input and output lengths uniform in 100–1024, `ignore_eos`). Compare HF `generate` (static batching) · tinyvllm(`torch_ref`) · tinyvllm(`triton`) · tinyvllm(`triton` + CUDA graphs) · nano-vllm · vLLM, on L4 (and T4, where nano-vllm can't run).
- [ ] **P10.2 Memory-waste reproduction** `[CPU+GPU]`: contiguous max-length reservation vs paged on the same request trace: % of KV memory holding real tokens, max concurrent requests under a fixed budget, and the resulting throughput. Plot it in the style of the paper's Fig. 2.
- [ ] **P10.3 Ablations** `[GPU]`: chunked prefill on/off (TTFT vs ITL) · prefix caching on/off (shared-prompt workload) · CUDA graphs on/off · block size (from P7.8) · `max_num_batched_tokens` sweep.
- [ ] **P10.4 Docs** `[CPU]`: README (pitch, architecture diagram, §6.4 worked example, results tables + plots, "how to run on CPU / Colab / L4"), `docs/design.md`, `docs/paged_attention.md`, and a blog post: *"Writing PagedAttention from scratch in Triton, on a laptop without a GPU."*
- [ ] **P10.5 Resume** `[CPU]`: fill the §15 bullet templates with numbers from `results/` only. → **M4 complete.**

### P11: Stretch menu (pick 1–2 after M2) · `[s]`
- [ ] **11.a Speculative decoding**: n-gram / prompt-lookup proposer (no draft model) + rejection sampling; the scheduler assigns draft tokens; the unified kernel already handles `q_len > 1`. ★ *Why does speculative decoding keep the output distribution exact?*
- [ ] **11.b Tensor parallelism**: column/row-parallel linear + all-reduce. Develop on the VM with 2 `gloo` processes (exactly like your L9 DDP run), then NCCL on 2×L4. ★ *TP vs PP for inference.*
- [ ] **11.c KV-cache quantization**: fp8/int8 with per-token-head scales → ~2× KV capacity; dequantize inside the kernel; measure perplexity drift.
- [ ] **11.d Swap preemption**: a pinned-CPU block pool with async copies; compare with recompute (the paper's §4.5 trade-off).
- [ ] **11.e Sliding-window attention**: free blocks that fall out of the window (Mistral/Gemma-style).
- [ ] **11.f Structured output**: JSON/regex via a token-mask FSM in the sampler.
- [ ] **11.g Rust core**: port `BlockPool` + scheduler to Rust via PyO3 (reusing your `gpt2rs` muscle) and measure scheduler overhead at 1K concurrent requests.

### P12: vLLM contribution track (runs in parallel from M1) · `[ ]`
- [ ] **12.1 Build & run vLLM.** On a GPU box: Python-only build, `VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto` (works when you change only Python). On the CPU VM: the CPU backend builds from source (`docs/getting_started/installation/cpu.md`), so you can read and run vLLM where you develop.
- [ ] **12.2 Read-map** (after each tinyvllm phase, read its vLLM twin and write 5 bullets on "what vLLM does that I don't, and why"):

| tinyvllm | vLLM (paths as of 2026-10-11) |
|---|---|
| `core/block_pool.py` | `vllm/v1/core/block_pool.py` (`BlockPool`), `vllm/v1/core/kv_cache_utils.py` (`KVCacheBlock`, `FreeKVCacheBlockQueue`, `hash_block_tokens`) |
| `core/kv_cache_manager.py` | `vllm/v1/core/kv_cache_manager.py`, `single_type_kv_cache_manager.py`, `kv_cache_coordinator.py` |
| `core/scheduler.py` | `vllm/v1/core/sched/scheduler.py` (`schedule()`: running first, then waiting; token budget; preemption), `sched/output.py` |
| `worker/model_runner.py`, `input_batch.py` | `vllm/v1/worker/gpu_model_runner.py`, `gpu_input_batch.py`, `block_table.py` |
| `attention/ops/reshape_and_cache.py` | `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` |
| `attention/ops/unified_attention.py` | `vllm/v1/attention/ops/triton_unified_attention.py` (`kernel_unified_attention`, `reduce_segments`, 2D vs 3D) |
| `attention/backends/*` | `vllm/v1/attention/backends/triton_attn.py`, `flash_attn.py`, `cpu_attn.py`, `registry.py` |
| `tests/kernels/test_*` | `tests/kernels/attention/test_triton_unified_attention.py` (`ref_paged_attn`) |
| `layers/sampler.py` | `vllm/v1/sample/sampler.py`, `sample/ops/topk_topp_sampler.py`, `penalties.py`, `logprobs.py` |
| `engine/*` | `vllm/v1/engine/core.py`, `async_llm.py`, `output_processor.py`, `detokenizer.py`, `parallel_sampling.py` |
| `entrypoints/openai_server.py` | `vllm/entrypoints/openai/api_server.py` |
| `worker/cuda_graphs.py` | `vllm/v1/cudagraph_dispatcher.py` |
| scheduler/prefix tests | `tests/v1/core/test_scheduler.py`, `test_prefix_caching.py`, `test_kv_cache_utils.py` |
| benchmarks | `vllm/benchmarks/serve.py`, `throughput.py` (`vllm bench serve` / `vllm bench throughput`) |

- [ ] **12.3 Contribution ladder:** (1) reproduce and triage an open issue in an area you've built (scheduler, KV cache, sampler, Triton attention); (2) a docs or test PR (edge cases you hit in tinyvllm are good candidates, e.g. block-boundary or preemption cases); (3) a small bug fix; (4) Triton-backend work: coverage, tests, or tuning for T4/L4-class GPUs, backed by benchmark tables; (5) a design discussion (RFC) once you're known.
- [ ] **12.4 Etiquette checklist:** look at the `good first issue` label and the "Selected onboarding tasks" project; join the developer Slack (slack.vllm.ai); commits **signed off** (`git commit -s`, DCO); run `pre-commit`; small focused PRs that link the issue; perf PRs include before/after benchmarks; **disclose AI assistance in the PR description** (vLLM's contributing guide requires this, and requires that you personally review and test every line). → **M5 complete** when a PR is merged.

---

## 10. Testing strategy

| Layer | Test | Oracle | Runs on |
|---|---|---|---|
| Models | logit parity, greedy continuation | HF transformers | CPU (fp32) |
| BlockPool / KV manager | property-based random programs (hypothesis) | §6.5 invariants | CPU |
| Attention | parametrized grid (lengths, shuffled block tables, GQA, dtypes, block sizes) | contiguous SDPA → `torch_ref` | CPU; Triton via interpreter; GPU for real |
| Scheduler | fake-model unit tests: budgets, chunking, preemption order, prefix hits | hand-computed `SchedulerOutput` | CPU |
| End to end | batch invariance; preemption invariance; prefix-cache invariance; streaming == non-streaming | solo run / big pool / cache off | CPU (exact); GPU (logprob tolerance) |
| Server | `openai` client round trip, SSE framing, abort | offline `LLM.generate` | CPU |

Tolerances: CPU fp32 exact tokens, logits ≤ 1e-4 · GPU fp16/bf16: attention outputs ≤ 2e-2 abs (as in vLLM's kernel tests), greedy tokens may diverge late; compare logprobs of the first *k* tokens instead.

---

## 11. Benchmarks & experiments (protocols)

| Bench | Script | Protocol | Output |
|---|---|---|---|
| Naive baseline | `bench_naive.py` | HF `generate`, static padded batches | tok/s, KV reserved vs used |
| Offline throughput | `bench_throughput.py` | nano-vllm protocol (P10.1); warm-up first; 3 runs, median | output tok/s, total tok/s |
| Kernels | `bench_kernels.py` | batch {1, 8, 32, 128} × context {128 … 16K} × backend; CUDA events; 50 iters after 10 warm-up | latency, achieved GB/s, roofline plot |
| Memory waste | `bench_memory_waste.py` | same request trace through contiguous-reservation vs paged allocators (simulation + real engine stats) | % real-token KV, max concurrency |
| Serving | `bench_serving.py` / `vllm bench serve` | Poisson arrivals, QPS sweep, fixed dataset | TTFT/ITL p50/p99, throughput |
| Prefix caching | `bench_prefix.py` | N requests sharing a long system prompt | hit rate, prefill tokens saved, TTFT |

Every run writes `results/<bench>_<gpu>_<git-sha>_<date>.json` with the full config (Law 5).

---

## 12. Contributor mindset: habits to build now
- Read vLLM's version of each component **after** you've written your own, never before. Productive confusion first, then comparison.
- Keep a `docs/vllm_diff.md` log: "vLLM does X, I do Y, because Z." It becomes PR ideas and interview stories.
- Write tests the way vLLM does (parametrized grids, reference implementations), so your first vLLM PRs look native.

---

## 13. Interview bank additions: INTERVIEWGUIDE Part R, "LLM inference & serving"
Tags: `V0`–`V12` = the tinyvllm phase where you build it. Answers are written only after the code works (same rule as the main guide).

- [ ] ★ What is a KV cache? Why does it speed up generation, and what does it cost (formula)? `V2` *(already in the bank as `SQ`; answer it here)*
- [ ] ★ Prefill vs decode: compute-bound vs memory-bound; arithmetic intensity. `V2` `V7`
- [ ] ★ What problem does PagedAttention solve? Internal, external and reservation fragmentation. `V0` `V2`
- [ ] ★ Explain the block table and slot mapping (the OS page-table analogy). `V2`
- [ ] What does block size trade off? `V7`
- [ ] ★ Static vs continuous (iteration-level) batching (Orca). `V3`
- [ ] ★ Chunked prefill: why, and the TTFT vs ITL trade-off (Sarathi-Serve). `V3`
- [ ] Preemption: recompute vs swap. `V3` `V11`
- [ ] ★ Prefix caching: hash chain, full blocks only, LRU eviction; vs RadixAttention. `V5`
- [ ] Copy-on-write for parallel sampling / beam search. `V4`
- [ ] ★ FlashAttention vs PagedAttention: orthogonal (compute tiling vs memory layout); how they combine. `V7`
- [ ] ★ Online softmax recurrence; why it's exact. `V7`
- [ ] Flash-Decoding / split-KV: when and why. `V7`
- [ ] ★ GQA/MQA: effect on KV memory and decode bandwidth. `V1` `V7`
- [ ] Why decode attention is memory-bound; roofline reasoning. `V7`
- [ ] CUDA graphs: what they remove; constraints (static shapes, padding, buckets). `V8`
- [ ] Why `.item()`/`.tolist()` syncs hurt; pinned memory and async copies. `V6`
- [ ] ★ Sampling: greedy, temperature, top-k, top-p, min-p; penalties. `V4`
- [ ] Batch invariance / nondeterminism in batched inference. `V3`
- [ ] Incremental detokenization and stop strings (UTF-8 boundaries). `V4`
- [ ] How does vLLM decide how many KV blocks to allocate (memory profiling)? `V6`
- [ ] ★ TTFT, TPOT/ITL, throughput, goodput: definitions and trade-offs. `V9`
- [ ] Why separate the API server, engine core and workers into processes? `V9`
- [ ] ★ Speculative decoding: why the output distribution stays exact. `V11`
- [ ] Tensor vs pipeline parallelism for inference. `V11`
- [ ] KV-cache quantization (fp8/int8): capacity vs accuracy. `V11`
- [ ] Disaggregated prefill/decode (DistServe): concept. `X`
- [ ] ★ System design: "Serve a 7B model to 1,000 concurrent users on N GPUs." (KV memory math, batching, caching, quantization, autoscaling) `V10`
- [ ] Project story: "Tell me about tinyvllm." (2-minute version + the hardest bug) `V10`
- [ ] Project story: "How did you verify your kernel is correct?" (oracle-first, interpreter on CPU) `V7`

---

## 14. Reading list (in the order you'll need it)
1. **Kwon et al., 2023.** *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP '23), arXiv:2309.06180. **P0**
2. vLLM blog, *vLLM V1: A Major Upgrade to vLLM's Core Architecture* (Jan 2025): the unified scheduler and `{req_id: num_tokens}`. **P0/P3**
3. **Yu et al., 2022.** *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI '22): iteration-level scheduling. **P3**
4. **Agrawal et al., 2024.** *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (OSDI '24): chunked prefill. **P3**
5. **Ainslie et al., 2023.** *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* **P1**
6. **Zheng et al., 2023.** *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention), arXiv:2312.07104. **P5**
7. **Dao et al., 2022 / Dao, 2023.** FlashAttention 1 & 2 (you met FA in L9), plus the Flash-Decoding blog post (Dao, Haziza, Massa, Sizov, 2023). **P7**
8. Triton tutorials (vector add, fused softmax, matmul, fused attention), at triton-lang.org. **P7.0**
9. vLLM blog, *vLLM Triton Attention Backend Deep Dive* (Mar 2026), and the IBM/Red Hat PyTorch Conference talks on the Triton backend. **P7.5**
10. **Pope et al., 2022.** *Efficiently Scaling Transformer Inference*: memory-bandwidth analysis and partitioning. **P7/P11**
11. **Leviathan et al., 2023.** *Fast Inference from Transformers via Speculative Decoding.* **P11**
12. **Zhong et al., 2024.** *DistServe: disaggregating prefill and decoding.* **X**
13. Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* (2025): batch invariance. **P3.5**
14. Aleksa Gordić, *Inside vLLM: Anatomy of a High-Throughput LLM Inference System* (2025): a guided tour of the V1 code. **P12**
15. nano-vllm source (~1.2K LOC): read it once **after M1** to compare designs; not as a template.

---

## 15. Resume & portfolio

**Headline:** *tinyvllm: an LLM inference engine with hand-written PagedAttention (Python, PyTorch, Triton)*

Bullet templates. Fill `<…>` **only** from `results/` (Law 5):
- Built **tinyvllm**, a from-scratch LLM inference engine (~`<LOC>` lines) reimplementing vLLM's core: a paged KV cache with hash-chained prefix caching, LRU eviction and copy-on-write parallel sampling; a vLLM-V1-style token-budget scheduler with chunked prefill and preemption; and an OpenAI-compatible streaming server with Prometheus metrics.
- Wrote **PagedAttention kernels in Triton** (paged decode, GQA-packed, split-KV, and a unified varlen prefill+decode kernel), reaching `<X>`% of peak HBM bandwidth on an NVIDIA L4 and `<Y>`× the speed of a PyTorch gather baseline, within `<Z>`% of FlashAttention's paged decode. Validated against a reference with `<N>` parametrized tests, developed GPU-free via Triton's CPU interpreter.
- Reproduced the PagedAttention paper's memory result: KV-cache waste fell from `<A>`% (contiguous max-length reservation) to `<B>`%, raising max concurrent requests `<C>`× and offline throughput `<D>`× over HF Transformers on Qwen3-0.6B.
- Contributed `<PR title>` to **vLLM** (`<link>`).

**Portfolio checklist:** README with results + plots · `docs/design.md` · blog post · CI badge · reproducible `results/` · a 2-minute demo GIF (scheduler timeline + streaming server) · links from the karpathynn README and the HF model card (your TinyStories-51M served by tinyvllm).

---

## 16. Logs & boards

### Session log
Template:
```
### Session N — YYYY-MM-DD — P<a>.<b> → P<c>.<d>
Done:              …
Code:              files created/changed · tests added
Key insights (your words): …
Struggled with / bugs: …
Hints used:        H1×_, H2×_, H3×_, H4×_
vLLM read:         file(s) + 1-line diff note
Interview answers written: …
Next:              …
```
*(no sessions yet)*

### Results board
| Date | Bench | Hardware | Model | Config | Result | File |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

### Decisions log
| Date | Decision | Why |
|---|---|---|
| 2026-10-11 | Write the PagedAttention kernels ourselves (PyTorch oracle → Triton) instead of calling flash-attn's `block_table` | The kernel is the point; nano-vllm hides it |
| 2026-10-11 | Triton, not CUDA C++, for kernels | Develop and test on the CPU via `TRITON_INTERPRET=1`; runs on T4/L4/A100; vLLM's Triton backend is a live contribution target |
| 2026-10-11 | vLLM-V1-style unified token-budget scheduler | Simpler than separate prefill/decode phases; chunked prefill, prefix hits and spec decoding all fall out of it; matches current vLLM |
| 2026-10-11 | vLLM naming throughout | 1:1 transfer to reading and contributing to vLLM |
| 2026-10-11 | Models: GPT-2 (incl. own TinyStories-51M), SmolLM2-135M, Qwen3-0.6B | Personal model + MHA/learned positions; GQA + RoPE small enough for CPU; Qwen3-0.6B matches nano-vllm's benchmark |
| 2026-10-11 | Default block_size 16 | vLLM's default; low internal fragmentation; validated by the P7.8 ablation |
| 2026-10-11 | GPU plan: Colab T4 (fp16) for bring-up, GCP L4 spot (bf16) for headline numbers | Free/cheap; T4 also shows tinyvllm runs where flash-attn 2 can't |

### Parking lot
*(empty)*
