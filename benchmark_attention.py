"""CUDA benchmark for local attention, full Flash Attention, and the hybrid.

Peak values use ``torch.cuda.max_memory_allocated`` (not Windows RSS) and
therefore measure allocations attributable to the current benchmark process.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from App import HybridLongContextLayer, SlidingWindowAttention, streaming_causal_prefill


class FullCausalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by heads")
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.qkv, self.out = nn.Linear(d_model, 3 * d_model), nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, width = x.shape
        q, k, v = self.qkv(x).view(batch, tokens, 3, self.num_heads, self.head_dim).unbind(2)
        attended = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.out(attended.transpose(1, 2).reshape(batch, tokens, width))


def run(model: nn.Module, x: Tensor, kind: str, prefill_chunk_size: int | None = None) -> Tensor:
    if kind == "sliding":
        return model(x, causal=True, position_ids=None)[0]  # type: ignore[operator]
    if kind == "hybrid":
        if prefill_chunk_size is not None:
            return streaming_causal_prefill(model, x, chunk_size=prefill_chunk_size)  # type: ignore[arg-type]
        return model(x, causal=True)  # type: ignore[operator]
    return model(x)


def benchmark_one(kind: str, tokens: int, args: argparse.Namespace, device: torch.device) -> dict[str, float | str | int]:
    if args.training and args.prefill_chunk_size is not None:
        raise ValueError("--prefill-chunk-size is inference-only")
    factories = {
        "sliding": lambda: SlidingWindowAttention(args.d_model, args.heads, args.window, 0.0, attention_chunk_size=args.attention_chunk_size),
        "full": lambda: FullCausalAttention(args.d_model, args.heads),
        "hybrid": lambda: HybridLongContextLayer(args.d_model, args.window, num_heads=args.heads, attention_chunk_size=args.attention_chunk_size, state_inference_chunk_size=args.state_inference_chunk_size),
    }
    torch.manual_seed(0)
    model = factories[kind]().to(device)
    x = torch.randn(args.batch_size, tokens, args.d_model, device=device, requires_grad=args.training)
    model.train(args.training)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.set_grad_enabled(args.training):
        for _ in range(args.warmup):
            output = run(model, x, kind, args.prefill_chunk_size)
            if args.training:
                output.float().square().mean().backward()
                model.zero_grad(set_to_none=True)
                x.grad = None
    torch.cuda.synchronize(device)
    timings = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.repeats):
        start = time.perf_counter()
        with torch.set_grad_enabled(args.training):
            output = run(model, x, kind, args.prefill_chunk_size)
            if args.training:
                output.float().square().mean().backward()
                model.zero_grad(set_to_none=True)
                x.grad = None
        torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "kind": kind,
        "tokens": tokens,
        "median_ms": statistics.median(timings),
        "peak_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--attention-chunk-size", type=int, default=256)
    parser.add_argument("--state-inference-chunk-size", type=int, default=None, help="Exact bounded-memory recurrent scan for inference only.")
    parser.add_argument("--prefill-chunk-size", type=int, default=None, help="Hybrid-only exact cached streaming prefill; inference only.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA; CPU/RSS measurements are intentionally unsupported.")
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU={torch.cuda.get_device_name(device)} mode={'training' if args.training else 'inference'} batch={args.batch_size} d_model={args.d_model}")
    print("| tokens | layer | median ms | peak MiB |")
    print("|-------:|:------|----------:|---------:|")
    for tokens in args.tokens:
        for kind in ("sliding", "full", "hybrid"):
            row = benchmark_one(kind, tokens, args, device)
            print(f"| {tokens} | {kind} | {row['median_ms']:.2f} | {row['peak_mib']:.1f} |")


if __name__ == "__main__":
    main(parse_args())
