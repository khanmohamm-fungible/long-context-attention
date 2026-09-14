from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from App import SlidingWindowAttention


class FullCausalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, x: Tensor, positions: Tensor) -> Tensor:
        angles = positions.float().unsqueeze(-1) * self.inv_freq
        cos, sin = angles.cos().unsqueeze(2), angles.sin().unsqueeze(2)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos),
            dim=-1,
        ).flatten(-2)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, width = x.shape
        positions = torch.arange(tokens, device=x.device).expand(batch, -1)
        qkv = self.qkv(x).view(
            batch,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.unbind(dim=2)
        q, k = self._rope(q, positions), self._rope(k, positions)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=True,
        )
        return self.out(attended.transpose(1, 2).reshape(batch, tokens, width))


def worker(kind: str, tokens: int, d_model: int, heads: int, window: int) -> None:
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    x = torch.randn(1, tokens, d_model)
    if kind == "sliding":
        model: nn.Module = SlidingWindowAttention(
            d_model,
            heads,
            window,
            0.0,
        ).eval()

        def run() -> Tensor:
            output, _ = model(
                x,
                causal=True,
                position_ids=None,
            )
            return output

    else:
        model = FullCausalAttention(d_model, heads).eval()

        def run() -> Tensor:
            return model(x)

    with torch.inference_mode():
        run()
        durations = []
        for _ in range(5):
            start = time.perf_counter()
            output = run()
            durations.append((time.perf_counter() - start) * 1000)
    print(
        json.dumps(
            {
                "kind": kind,
                "tokens": tokens,
                "median_ms": statistics.median(durations),
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
                "finite": bool(torch.isfinite(output).all()),
            }
        )
    )


def benchmark(args: argparse.Namespace) -> None:
    rows = []
    for tokens in args.tokens:
        for kind in ("sliding", "full"):
            command = [
                sys.executable,
                __file__,
                "--worker",
                kind,
                "--tokens",
                str(tokens),
                "--d-model",
                str(args.d_model),
                "--heads",
                str(args.heads),
                "--window",
                str(args.window),
            ]
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            rows.append(json.loads(completed.stdout))

    print(
        f"CPU inference, batch=1, d_model={args.d_model}, "
        f"heads={args.heads}, window={args.window}"
    )
    print("| tokens | sliding ms | full ms | speed ratio | sliding MB | full MB |")
    print("|-------:|-----------:|--------:|------------:|-----------:|--------:|")
    for tokens in args.tokens:
        sliding = next(
            row for row in rows
            if row["tokens"] == tokens and row["kind"] == "sliding"
        )
        full = next(
            row for row in rows
            if row["tokens"] == tokens and row["kind"] == "full"
        )
        ratio = full["median_ms"] / sliding["median_ms"]
        print(
            f"| {tokens} | {sliding['median_ms']:.2f} | "
            f"{full['median_ms']:.2f} | {ratio:.2f}x | "
            f"{sliding['peak_rss_mb']:.1f} | {full['peak_rss_mb']:.1f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("sliding", "full"))
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--window", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        worker(
            arguments.worker,
            arguments.tokens[0],
            arguments.d_model,
            arguments.heads,
            arguments.window,
        )
    else:
        benchmark(arguments)
