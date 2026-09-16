# Hybrid Long-Context Attention

## Problem Statement:
Modern AI language models often rely heavily on attention mechanisms to understand relationships between tokens. However, full attention becomes computationally expensive as the context length increases, while simpler recurrent or local-attention methods may struggle to retain and retrieve important information from distant parts of a document.

This creates a challenge: How can we build an AI architecture that processes long sequences efficiently while still understanding local relationships and accessing relevant distant information?

## Our Solution:
An experimental PyTorch language-model layer for long-context research. It combines efficient local attention, a recurrent long-range state, retrieval over external documents, and source-grounded token copying. In simple words an alternative to the Transformer Architecture's Full Attention Layers with more efficiency.

Our idea is a fusion of multiple proved mechanisms(Mamba, Samba etc.) and borrows concepts from them. It is not exactly a novel architecture but a combination of them. This architecture efficiency levels assumably compatible with full attention.

> Important Note: This is a research proof of concept. The test results given below do not explicitly guarantee that hybrid layer will outperform full attention. The results are based on our experiments. 



## Architecture:

```text
Input text → GPT-2 tokenizer → token embeddings
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
      Sliding-window      Gated recurrent   Retrieved evidence
        local attention    long-range state     attention
               │                │                │
               └────────────────┴────────────────┘
                                ▼
               Contextual source bridge
        source position + document ID + source encoder
                                ▼
                  Pointer-generator copy head
                                ▼
                    Next-token distribution
```

### Components

- `SlidingWindowAttention`: causal or bidirectional RoPE local attention with bounded KV caching.
- `ParallelGatedStateMixer`: stable gated recurrence. The portable PyTorch fallback uses a parallel `O(N log N)` scan rather than a Python token loop.
- `RetrievedEvidenceAttention`: attends to shared document-level retrieved memory without copying evidence for every query position.
- `ContextualSourceEncoder`: adds document-local source positions and document IDs, then bidirectionally contextualizes retrieved source tokens.
- `SourceGroundedCopyHead`: a pointer-generator that mixes vocabulary generation with a distribution over exact retrieved token IDs.
- `FAISSVectorRetriever` and `SQLiteFTSRetriever`: semantic and lexical retrieval with text and metadata provenance.

## Repository layout

| File | Purpose |
|---|---|
| `App.py` | Hybrid layer, attention, state mixers, retrieval attention, and copy head. |
| `mini_hybrid_lm.py` | Small causal LM built from hybrid layers. |
| `source_bridge.py` | Position-aware contextual encoding bridge for retrieved source tokens. |
| `memory_bridge.py` | Converts retrieved documents into model/device tensors. |
| `retrieval.py` | FAISS semantic and SQLite FTS retrieval implementations. |
| `dataset.py` | Synthetic external-knowledge retrieval benchmark. |
| `train_poc.py` | GPU-only hybrid retrieval/copy training and evaluation. |
| `full_copy_baseline.py` | Full-attention pointer-reader baseline. |
| `benchmark_attention.py` | CUDA latency and peak-VRAM benchmark. |
| `profile_hybrid.py` | Per-component activation-memory profile and ablations. |
| `test_app.py` | Reference, cache, padding, and copy-path tests. |

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA-enabled PyTorch for training and benchmarks
- Tested on an NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB VRAM and PyTorch `2.14.0+cu126`.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify CUDA before training:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Do not use a PyTorch build ending in `+cpu`; the proof-of-concept training script deliberately refuses CPU execution.

## Quick start

Run correctness checks:

```powershell
.\.venv\Scripts\python.exe test_app.py
.\.venv\Scripts\python.exe App.py
```

Run GPU retrieval/copy training:

```powershell
.\.venv\Scripts\python.exe train_poc.py --steps 300 --synthetic-size 16 --d-model 64 --layers 1 --window 16 --top-k 1
```

The script trains only on answer tokens, retrieves using the question alone, and evaluates answers autoregressively. It reports retrieval recall@1, answer-token accuracy, and exact-answer accuracy.

Use oracle evidence to isolate reader quality from retrieval quality:

```powershell
.\.venv\Scripts\python.exe train_poc.py --steps 300 --synthetic-size 16 --d-model 64 --layers 1 --window 16 --oracle-retrieval
```

Run the full-attention pointer-reader baseline on the same oracle task:

```powershell
.\.venv\Scripts\python.exe full_copy_baseline.py --steps 300 --synthetic-size 16 --d-model 64
```

## CUDA benchmarks

Compare local attention, full PyTorch SDPA, and the complete hybrid layer:

```powershell
.\.venv\Scripts\python.exe benchmark_attention.py --tokens 1024 2048 4096 8192
.\.venv\Scripts\python.exe benchmark_attention.py --training --tokens 4096 8192
.\.venv\Scripts\python.exe benchmark_attention.py --tokens 8192 16384 --attention-chunk-size 512
.\.venv\Scripts\python.exe benchmark_attention.py --tokens 32768 --attention-chunk-size 512 --prefill-chunk-size 4096 --repeats 5 --warmup 2
```

The benchmark uses synchronized CUDA timing and `torch.cuda.max_memory_allocated`, reporting GPU latency and allocated VRAM rather than CPU RSS. PyTorch SDPA may select Flash Attention on supported hardware.

Profile every hybrid stage and run retrieval/copy ablations:

```powershell
.\.venv\Scripts\python.exe profile_hybrid.py --tokens 2048 --source-tokens 128 --d-model 128 --heads 8 --window 128
```

This reports cumulative live activation memory after embedding, local attention, recurrence, source encoding, retrieval, cross-attention, copy, LM head, and backward, followed by separate inference/training ablations.

For exact, low-VRAM cached prefill, process the hybrid prompt in 4K blocks:

```powershell
.\.venv\Scripts\python.exe benchmark_attention.py --tokens 8192 16384 --attention-chunk-size 512 --prefill-chunk-size 4096 --repeats 8 --warmup 3
```

`--prefill-chunk-size` is inference-only. It preserves causal outputs while
bounding the local-attention and recurrent-scan working set to one block.
`MiniHybridLM` automatically selects the same 4,096-token cached-prefill path
for inference prompts longer than 4,096 tokens; pass
`inference_prefill_chunk_size=None` at construction time to disable it.

## Experimental results

The retrieval/copy task uses random access codes held out from training. It tests copying unseen values from evidence rather than memorizing them.

On the 16-item synthetic corpus, 300 GPU steps, `d_model=64`, and one layer:

| Reader / evidence setting | Recall@1 | Token accuracy | Exact answer accuracy |
|---|---:|---:|---:|
| Earlier hybrid reader with raw source embeddings, oracle evidence | 100% | 37.5% | 0% |
| Hybrid reader with contextual source bridge, oracle evidence | 100% | 100% | 100% |
| Hybrid reader with contextual source bridge, semantic top-1 retrieval | 75% | 75% | 75% |
| Full-attention pointer-reader, oracle evidence | 100% | 100% | 100% |

The contextual bridge removes the reader failure: with correct evidence, the hybrid matches the full-attention baseline on this small task. With semantic retrieval, the remaining errors align with retrieval misses.

### 8K/16K inference memory check

The copy head and output projection now emit vocabulary distributions only for
requested positions: answer-token positions during training and the final
position during autoregressive decoding. At 2K tokens with the GPT-2
vocabulary, this reduced copy-plus-output live allocation from about **2.08
GiB to 116 MiB**, and backward peak from **3.65 GiB to 199 MiB**.

With autograd disabled correctly during inference, the attention-only benchmark
tested on a laptop with RTX 4050 produced:

| Tokens | Full SDPA | Hybrid one-shot | Hybrid cached prefill |
|---:|---:|---:|---:|
| 8,192 | 20.48 ms / 36.4 MiB | 23.89 ms / 69.4 MiB | — |
| 16,384 | 64.36 ms / 64.4 MiB | 29.24 ms / 129.4 MiB | 41.21 ms / 59.8 MiB |

One-shot hybrid stays within 100 MiB of full attention at both lengths and
becomes faster at 16K. The 4K cached-prefill path is exact, retains less VRAM
than full SDPA at 16K, and remains faster than full SDPA in the stable
multi-repeat test. A fused FLA/Triton state backend is still the path to reduce
kernel-launch overhead further on Linux/WSL.

## Research status

This repository is suitable as the foundation for a technical report or a research paper.
 A defensible framing
is a hybrid long-context architecture with contextual retrieval copying and
memory-aware output/prefill optimizations.

The current evidence supports the following narrow claims:

- Contextualized retrieved source tokens are necessary for reliable exact
  copying in the provided controlled task.
- Output-position selection eliminates the dominant vocabulary-space memory
  allocation in the original copy path.
- At the tested small width, cached hybrid prefill can use less VRAM than full
  SDPA at 16K tokens while preserving exact causal outputs.

It does **not** yet establish general language-model quality or a universal
speed advantage over Flash Attention.

### Reproducible multi-model study

`long_context_study.py` is the controlled study harness. It instantiates the
same parameter tensors for every variant, routes them as full, sliding-only,
hybrid, or hybrid-plus-retrieval, and gives every condition the same training
token budget. It exports one CSV row per context length, corpus, seed, and
variant, including parameter count, final loss, held-out accuracy, training
time, and peak CUDA memory.

The two included corpora are deliberately diagnostic:

- `needle`: a unique marker appears at a random, potentially distant position;
  the final query marker asks for the token immediately following it;
- `retrieval`: the target is absent from the prompt and appears only in an
  external source. A real CPU FAISS `IndexFlatIP` lookup retrieves document IDs
  and reports recall@1/recall@k separately from reader accuracy.

The full-attention baseline now uses RoPE, matching the positional information
available to sliding attention. Retrieval CSV rows include
`retrieval_recall_at_1` and `retrieval_recall_at_k`.

Run a pilot:

```powershell
.\.venv\Scripts\python.exe long_context_study.py --tokens 256 1024 --seeds 7 19 --train-tokens 32768 --eval-batches 16 --batch-size 4 --d-model 64 --layers 1 --heads 4 --window 64 --bf16 --output study_pilot.csv
```

For the full local study, use three seeds and a substantially larger fixed
token budget. This takes longer but avoids giving longer contexts fewer
optimizer updates than shorter ones:

```powershell
.\.venv\Scripts\python.exe long_context_study.py --tokens 256 512 1024 2048 --seeds 7 19 41 --train-tokens 524288 --eval-batches 64 --batch-size 4 --d-model 128 --layers 2 --heads 4 --window 64 --bf16 --output study_full.csv
```

The included pilot validated equal parameter counts (197,249 in every
condition) and showed expected retrieval behavior. It is not a final result:
at a 32,768-token budget, the 1,024-token setting makes only eight optimizer
updates, which is insufficient for the needle task.
Further testing is still necessary to establish trusted claims.

### Cached-prefill scaling check

```powershell
.\.venv\Scripts\python.exe benchmark_attention.py --tokens 4096 8192 16384 --attention-chunk-size 512 --prefill-chunk-size 4096 --repeats 5 --warmup 2
```

| Tokens | Full SDPA | Hybrid cached prefill |
|---:|---:|---:|
| 4,096 | 6.10 ms / 22.4 MiB | 43.51 ms / 39.5 MiB |
| 8,192 | 20.90 ms / 36.4 MiB | 75.42 ms / 47.7 MiB |
| 16,384 | 61.23 ms / 64.4 MiB | 156.37 ms / 59.8 MiB |

Cached prefill protects VRAM at longer lengths, but the current native-Windows
implementation has Python-level cache-loop overhead. Treat it as a
memory-constrained inference option, not the low-latency default.

## Optional fused linear-attention backend

`HybridLongContextLayer` supports `state_backend="fla"`, which integrates FLA's chunk-parallel Gated Linear Attention on supported CUDA/Linux systems:

```bash
pip install 'flash-linear-attention[cuda]'
python train_poc.py --state-backend fla --no-local-attention
```
FLA is not currently usable from the native Windows environment used for this project because its required Triton runtime does not ship a compatible Windows wheel. The default `torch` state mixer is a portable correctness baseline.

To enable the existing FLA backend, install WSL2/Ubuntu from an **Administrator
PowerShell** (a system change that generally needs a restart), then install the
CUDA dependencies inside Ubuntu:

```powershell
wsl --install -d Ubuntu
```

```bash
# Run inside Ubuntu after the WSL installation/restart.
cd /mnt/c/Users/Hannah/Desktop/ABBA_HACKATHON/long-context-attention
python3 -m venv .venv-linux
source .venv-linux/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install 'flash-linear-attention[cuda]'
python train_poc.py --state-backend fla --no-local-attention
```

The FLA path replaces the portable training-state scan with chunk-parallel
Gated Linear Attention kernels. Validate CUDA and benchmark it before using it
for study results.

## Limitations and next steps

- The synthetic benchmark is small; it does not demonstrate general language modeling or broad retrieval quality.
- The default state-scan fallback is `O(N log N)`, not a fused linear-time CUDA kernel.
- Retrieval currently runs through CPU FAISS and Sentence Transformers; it is outside the GPU layer timing benchmark.
- A rigorous long-context study should compare parameter-matched full, sliding-only, hybrid, and hybrid-plus-retrieval models across context lengths, corpora, random seeds, and equal training-token budgets.
- A paper-quality evaluation needs larger standard long-context and retrieval
  benchmarks, multiple random seeds/confidence intervals, and results at
  larger model widths and depths.
