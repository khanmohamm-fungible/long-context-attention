# Hybrid Long-Context Attention

A practical PyTorch long-context layer combining:

- bounded sliding-window attention using scaled dot-product attention;
- a gated long-range state mixer;
- shared or per-token retrieved evidence; and
- a source-grounded copy head.

The sliding attention path avoids materializing expanded
`[batch, tokens, window, heads, head_dim]` tensors. It supports causal KV
caching, padding masks, compact per-batch caches, and bidirectional local
attention.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## CUDA training on the RTX 4050

Use the project's CUDA-enabled virtual environment, not a system Python build
whose PyTorch suffix ends in ``+cpu``:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe train_poc.py --steps 200 --precision bf16
```

`train_poc.py` is GPU-only by design. It trains on an offline synthetic
external-knowledge corpus, uses question-only retrieval (no future-target
query leakage), bf16 autocast, token-level evidence copying, and a held-out
autoregressive answer-token/exact-answer evaluation. Replace that corpus with
your own train/validation documents before making language-quality claims.

For a fused chunk-parallel linear mixer on a supported CUDA/Linux environment,
install `flash-linear-attention[cuda]` and use:

```bash
python train_poc.py --state-backend fla --no-local-attention
```

Native Windows currently lacks FLA's Triton runtime. The default `torch`
backend remains portable but is a correctness baseline, not Flash-equivalent.

## Validation

```bash
python test_app.py
python App.py
```

## Benchmark

```bash
.\.venv\Scripts\python.exe benchmark_attention.py
.\.venv\Scripts\python.exe benchmark_attention.py --training --tokens 4096 8192
```

The GPU benchmark compares sliding-window attention, PyTorch SDPA (which may
select Flash Attention), and the complete hybrid layer. It reports synchronized
CUDA timing and peak allocated VRAM, and supports both inference and backward
passes.

## Memory-conscious inference

Use shared retrieved evidence with shape `[batch, top_k, d_model]` instead of
replicating evidence for every query token. When decoding, restrict copy-head
logits to the current position:

```python
log_probs = copy_head(
    hidden,
    source_states,
    source_token_ids,
    source_mask,
    output_positions=torch.tensor([-1], device=hidden.device),
)
```
