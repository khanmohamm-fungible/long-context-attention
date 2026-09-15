# Hybrid Long-Context Attention

An experimental PyTorch language-model layer for long-context research. It combines efficient local attention, a recurrent long-range state, retrieval over external documents, and source-grounded token copying.

> This is a research proof of concept, not a production LLM or a claim that the portable implementation outperforms Flash Attention.

## Architecture

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
```

The benchmark uses synchronized CUDA timing and `torch.cuda.max_memory_allocated`, reporting GPU latency and allocated VRAM rather than CPU RSS. PyTorch SDPA may select Flash Attention on supported hardware.

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

## Optional fused linear-attention backend

`HybridLongContextLayer` supports `state_backend="fla"`, which integrates FLA's chunk-parallel Gated Linear Attention on supported CUDA/Linux systems:

```bash
pip install 'flash-linear-attention[cuda]'
python train_poc.py --state-backend fla --no-local-attention
```

FLA is not currently usable from the native Windows environment used for this project because its required Triton runtime does not ship a compatible Windows wheel. The default `torch` state mixer is a portable correctness baseline; do not present it as Flash Attention-equivalent in throughput benchmarks.

## Limitations and next steps

- The synthetic benchmark is small; it does not demonstrate general language modeling or broad retrieval quality.
- The default state-scan fallback is `O(N log N)`, not a fused linear-time CUDA kernel.
- Full attention is often faster at short and moderate sequence lengths due to optimized Flash Attention kernels.
- Retrieval currently runs through CPU FAISS and Sentence Transformers; it is outside the GPU layer timing benchmark.
- A rigorous long-context study should compare parameter-matched full, sliding-only, hybrid, and hybrid-plus-retrieval models across context lengths, corpora, random seeds, and equal training-token budgets.

## License

No license file is currently included. Add one before distributing or accepting external contributions.
