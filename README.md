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

## Validation

```bash
python test_app.py
python App.py
```

## Benchmark

```bash
python benchmark_attention.py
python benchmark_attention.py --tokens 4096 8192 16384 32768
```

The benchmark compares the sliding-window implementation with full causal
PyTorch SDPA using equivalent QKV, RoPE, and output projections.

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
