"""CUDA activation profile and ablations for the hybrid long-context model."""
from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from App import HybridLongContextLayer, RMSNorm, SourceGroundedCopyHead
from mini_hybrid_lm import MiniHybridLM
from source_bridge import ContextualSourceEncoder


def mib(value: int) -> float:
    return value / 2**20


@dataclass
class Stage:
    name: str
    after_forward_mib: float


class IsolatedReader(nn.Module):
    """Same component family as MiniHybridLM, selectable for ablation."""

    def __init__(self, variant: str, vocab: int, d_model: int, heads: int, window: int) -> None:
        super().__init__()
        self.variant = variant
        self.embedding = nn.Embedding(vocab, d_model)
        self.norm = RMSNorm(d_model)
        self.full_qkv = nn.Linear(d_model, 3 * d_model)
        self.full_out = nn.Linear(d_model, d_model)
        self.local = HybridLongContextLayer(d_model, window, num_heads=heads)
        self.recurrent = self.local.state_mixer
        self.retrieval = self.local.retrieval
        self.source_encoder = ContextualSourceEncoder(d_model, heads)
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.copy = SourceGroundedCopyHead(d_model, vocab, generator=self.lm_head)

    def forward(self, ids: Tensor, source_ids: Tensor, source_mask: Tensor) -> Tensor:
        x = self.embedding(ids)
        if self.variant == "full":
            batch, tokens, width = x.shape
            q, k, v = self.full_qkv(self.norm(x)).view(batch, tokens, 3, self.local.local_attention.num_heads, self.local.local_attention.head_dim).unbind(2)
            attended = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
            x = x + self.full_out(attended.transpose(1, 2).reshape(batch, tokens, width))
        if self.variant in {"sliding", "hybrid", "hybrid_retrieval", "hybrid_copy"}:
            local, _ = self.local.local_attention(self.local.local_norm(x), causal=True, position_ids=None)
            x = x + local
        if self.variant in {"recurrent", "hybrid", "hybrid_retrieval", "hybrid_copy"}:
            mixed, _ = self.recurrent(self.local.state_norm(x), bidirectional=False, initial_state=None)
            x = x + mixed
        source_states = self.embedding(source_ids)
        if self.variant in {"hybrid_retrieval", "hybrid_copy"}:
            doc_memory = source_states.mean(dim=1, keepdim=True)
            x = x + self.retrieval(self.local.retrieval_norm(x), doc_memory, None)
        hidden = self.final_norm(x)
        if self.variant == "hybrid_copy":
            positions = torch.arange(source_ids.shape[1], device=ids.device)[None].expand(ids.shape[0], -1)
            docs = torch.zeros_like(positions)
            context = self.source_encoder(source_states, source_mask, positions, docs)
            return self.copy(hidden, context, source_ids, source_mask, output_positions=torch.tensor([-1], device=ids.device))
        return self.lm_head(hidden[:, -1:])


def profile_stages(args: argparse.Namespace, device: torch.device) -> None:
    """Measure live allocations after each explicit forward operation."""
    torch.manual_seed(0)
    model = MiniHybridLM(args.vocab, args.d_model, args.layers, args.window, args.heads).to(device).train()
    ids = torch.randint(0, args.vocab, (args.batch_size, args.tokens), device=device)
    source_ids = torch.randint(0, args.vocab, (args.batch_size, args.source_tokens), device=device)
    mask = torch.ones_like(source_ids, dtype=torch.bool)
    positions = torch.arange(args.source_tokens, device=device)[None].expand(args.batch_size, -1)
    documents = torch.zeros_like(positions)
    stages: list[Stage] = []
    torch.cuda.reset_peak_memory_stats(device)
    x = model.embedding(ids)
    stages.append(Stage("1. embeddings", mib(torch.cuda.memory_allocated(device))))
    layer = model.layers[0]
    local, _ = layer.local_attention(layer.local_norm(x), causal=True, position_ids=None)
    x = x + local
    stages.append(Stage("2. local attention", mib(torch.cuda.memory_allocated(device))))
    mixed, _ = layer.state_mixer(layer.state_norm(x), bidirectional=False, initial_state=None)
    x = x + mixed
    stages.append(Stage("3. recurrent state", mib(torch.cuda.memory_allocated(device))))
    raw_source = model.embedding(source_ids)
    contextual_source = model.source_encoder(raw_source, mask, positions, documents)
    stages.append(Stage("4. source encoder", mib(torch.cuda.memory_allocated(device))))
    memory = raw_source.mean(dim=1, keepdim=True)
    stages.append(Stage("5. document retrieval memory", mib(torch.cuda.memory_allocated(device))))
    evidence = layer.retrieval(layer.retrieval_norm(x), memory, None)
    x = x + evidence
    stages.append(Stage("6. retrieved cross attention", mib(torch.cuda.memory_allocated(device))))
    hidden = model.final_norm(x)
    last_position = torch.tensor([-1], device=device)
    copy_log_probs = model.copy_head(hidden, contextual_source, source_ids, mask, output_positions=last_position)
    stages.append(Stage("7. copy mechanism", mib(torch.cuda.memory_allocated(device))))
    # The copy head uses lm_head as its generator. Measure standalone logits
    # separately to expose vocabulary-space allocation rather than double count.
    lm_logits = model.lm_head(hidden[:, -1:])
    stages.append(Stage("8. LM head standalone", mib(torch.cuda.memory_allocated(device))))
    loss = -copy_log_probs[..., 0].mean() + lm_logits.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    peak = mib(torch.cuda.max_memory_allocated(device))
    print("\nComponent activation profile (training graph retained)")
    print("| Component | Live allocated after forward (MiB) |")
    print("|---|---:|")
    for stage in stages:
        print(f"| {stage.name} | {stage.after_forward_mib:.1f} |")
    print(f"| 9. backward peak | {peak:.1f} |")
    print("Note: rows are cumulative live CUDA allocations, not independent per-component totals.")


def benchmark_variant(variant: str, args: argparse.Namespace, device: torch.device, training: bool) -> tuple[float, float]:
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    model = IsolatedReader(variant, args.vocab, args.d_model, args.heads, args.window).to(device)
    model.train(training)
    ids = torch.randint(0, args.vocab, (args.batch_size, args.tokens), device=device)
    source_ids = torch.randint(0, args.vocab, (args.batch_size, args.source_tokens), device=device)
    source_mask = torch.ones_like(source_ids, dtype=torch.bool)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.set_grad_enabled(training):
        output = model(ids, source_ids, source_mask)
        if training:
            output.float().mean().backward()
    torch.cuda.synchronize(device)
    elapsed = 0.0
    for _ in range(args.repeats):
        if training:
            model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.set_grad_enabled(training):
            output = model(ids, source_ids, source_mask)
            if training:
                output.float().mean().backward()
        torch.cuda.synchronize(device)
        elapsed += (time.perf_counter() - start) * 1000
    return elapsed / args.repeats, mib(torch.cuda.max_memory_allocated(device))


def run_ablations(args: argparse.Namespace, device: torch.device) -> None:
    variants = ["full", "sliding", "recurrent", "hybrid", "hybrid_retrieval", "hybrid_copy"]
    print("\nAblation benchmark")
    print("| Variant | Inference ms | Inference peak MiB | Training ms | Training peak MiB |")
    print("|---|---:|---:|---:|---:|")
    for variant in variants:
        infer_ms, infer_mem = benchmark_variant(variant, args, device, training=False)
        train_ms, train_mem = benchmark_variant(variant, args, device, training=True)
        print(f"| {variant} | {infer_ms:.2f} | {infer_mem:.1f} | {train_ms:.2f} | {train_mem:.1f} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--source-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=50257)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This profiler requires CUDA")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU={torch.cuda.get_device_name(device)} tokens={args.tokens} source_tokens={args.source_tokens} vocab={args.vocab}")
    profile_stages(args, device)
    run_ablations(args, device)


if __name__ == "__main__":
    main()
