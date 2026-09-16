"""Reproducible, parameter-matched GPU long-context study.

Compares full attention, sliding-only, hybrid state mixing, and hybrid plus
retrieval across two controlled corpora. Each condition receives exactly the
same number of training tokens for a given seed and context length.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from App import HybridLongContextLayer, RMSNorm, SourceGroundedCopyHead
from source_bridge import ContextualSourceEncoder


VARIANTS = ("full", "sliding", "hybrid", "hybrid_retrieval")
CORPORA = ("needle", "retrieval")


class FullAttention(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, d_model // heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.qkv, self.out = nn.Linear(d_model, 3 * d_model), nn.Linear(d_model, d_model)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rope(self, x: Tensor, positions: Tensor) -> Tensor:
        angles = positions.float().unsqueeze(-1) * self.inv_freq
        cos, sin = angles.cos().unsqueeze(2), angles.sin().unsqueeze(2)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, width = x.shape
        q, k, v = self.qkv(x).view(batch, tokens, 3, self.heads, self.head_dim).unbind(2)
        positions = torch.arange(tokens, device=x.device).expand(batch, -1)
        q, k = self.rope(q, positions), self.rope(k, positions)
        attended = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.out(attended.transpose(1, 2).reshape(batch, tokens, width))


class StudyModel(nn.Module):
    """All variants retain identical parameter tensors; only routing differs."""

    def __init__(self, vocab: int, d_model: int, layers: int, heads: int, window: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.full_norm = nn.ModuleList(RMSNorm(d_model) for _ in range(layers))
        self.full = nn.ModuleList(FullAttention(d_model, heads) for _ in range(layers))
        self.hybrid = nn.ModuleList(HybridLongContextLayer(d_model, window, num_heads=heads) for _ in range(layers))
        self.mlp_norm = nn.ModuleList(RMSNorm(d_model) for _ in range(layers))
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)) for _ in range(layers))
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.source_encoder = ContextualSourceEncoder(d_model, heads, max_source_tokens=64)
        self.copy_head = SourceGroundedCopyHead(d_model, vocab, generator=self.lm_head)

    def forward(self, ids: Tensor, variant: str, source_ids: Tensor | None = None) -> Tensor:
        x = self.embedding(ids)
        for full_norm, full, hybrid, mlp_norm, mlp in zip(self.full_norm, self.full, self.hybrid, self.mlp_norm, self.mlp):
            if variant == "full":
                x = x + full(full_norm(x))
            elif variant == "sliding":
                local, _ = hybrid.local_attention(hybrid.local_norm(x), causal=True, position_ids=None, build_kv_cache=False)
                x = x + local
            else:
                local, _ = hybrid.local_attention(hybrid.local_norm(x), causal=True, position_ids=None, build_kv_cache=False)
                x = x + local
                mixed, _ = hybrid.state_mixer(hybrid.state_norm(x), bidirectional=False, initial_state=None)
                x = x + mixed
            x = x + mlp(mlp_norm(x))
        hidden = self.final_norm(x)
        if variant != "hybrid_retrieval" or source_ids is None:
            return self.lm_head(hidden[:, -1:])
        mask = torch.ones_like(source_ids, dtype=torch.bool)
        positions = torch.arange(source_ids.shape[1], device=ids.device)[None].expand(ids.shape[0], -1)
        doc_ids = torch.zeros_like(positions)
        sources = self.source_encoder(self.embedding(source_ids), mask, positions, doc_ids)
        return self.copy_head(hidden, sources, source_ids, mask, output_positions=torch.tensor([-1], device=ids.device))


def make_batch(corpus: str, batch: int, tokens: int, vocab: int, device: torch.device, top_k: int) -> tuple[Tensor, Tensor | None, Tensor, int, int]:
    """Return input, optional retrieved evidence, and final-token target."""
    # Reserve 0/1 for query markers. Values 2..vocab-1 are random facts.
    target = torch.randint(2, vocab, (batch,), device=device)
    ids = torch.randint(2, vocab, (batch, tokens), device=device)
    ids[:, -1] = 1
    if corpus == "needle":
        # Query marker 1 at the end asks for the token immediately after the
        # unique marker 0. The marker location is deliberately distant.
        marker_positions = torch.randint(1, tokens - 2, (batch,), device=device)
        ids.scatter_(1, marker_positions[:, None], 0)
        ids.scatter_(1, (marker_positions + 1)[:, None], target[:, None])
        return ids, None, target, 0, 0
    # Real FAISS search over an external synthetic document corpus. A document
    # is [marker, key, answer, ...]; prompts expose only the key. Exact one-hot
    # key vectors make retrieval deterministic, while preserving the real
    # FAISS index/search boundary and recall@k accounting.
    import faiss
    corpus_size = min(32, vocab - 2)
    keys = torch.randperm(vocab - 2, device=device)[:corpus_size] + 2
    values = torch.randint(2, vocab, (corpus_size,), device=device)
    selected = torch.randint(0, corpus_size, (batch,), device=device)
    key, target = keys[selected], values[selected]
    ids[:, 0] = key
    documents = torch.randint(2, vocab, (corpus_size, 16), device=device)
    documents[:, 0], documents[:, 1], documents[:, 2] = 0, keys, values
    vectors = F.one_hot(keys, num_classes=vocab).float().cpu().numpy()
    queries = F.one_hot(key, num_classes=vocab).float().cpu().numpy()
    index = faiss.IndexFlatIP(vocab); index.add(vectors)
    _, found = index.search(queries, min(top_k, corpus_size))
    found_tensor = torch.as_tensor(found, device=device)
    source = documents[found_tensor].reshape(batch, -1)
    recall1 = int((found_tensor[:, 0] == selected).sum())
    recallk = int((found_tensor == selected[:, None]).any(dim=1).sum())
    return ids, source, target, recall1, recallk


def evaluate(model: StudyModel, variant: str, corpus: str, args: argparse.Namespace, device: torch.device, seed: int) -> tuple[float, float | None, float | None]:
    generator = torch.Generator(device=device).manual_seed(seed + 10_000)
    correct = total = recall1 = recallk = 0
    model.eval()
    with torch.inference_mode():
        for _ in range(args.eval_batches):
            # Preserve deterministic but independent held-out draws.
            torch.manual_seed(generator.initial_seed() + total)
            ids, source, target, hit1, hitk = make_batch(corpus, args.batch_size, args.tokens, args.vocab, device, args.retrieval_top_k)
            pred = model(ids, variant, source).squeeze(1).argmax(-1)
            correct += int((pred == target).sum())
            total += target.numel()
            recall1 += hit1; recallk += hitk
    return correct / total, (recall1 / total if corpus == "retrieval" else None), (recallk / total if corpus == "retrieval" else None)


def run_condition(variant: str, corpus: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    torch.manual_seed(seed)
    model = StudyModel(args.vocab, args.d_model, args.layers, args.heads, args.window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(args.train_tokens / (args.batch_size * args.tokens))
    autocast = lambda: torch.autocast("cuda", torch.bfloat16) if args.bf16 else nullcontext()
    model.train(); torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(steps):
        ids, source, target, _, _ = make_batch(corpus, args.batch_size, args.tokens, args.vocab, device, args.retrieval_top_k)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(ids, variant, source).squeeze(1)
            loss = F.cross_entropy(logits, target)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    torch.cuda.synchronize(device)
    accuracy, recall1, recallk = evaluate(model, variant, corpus, args, device, seed)
    return {
        "variant": variant, "corpus": corpus, "seed": seed, "tokens": args.tokens,
        "train_tokens": steps * args.batch_size * args.tokens, "steps": steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "final_loss": round(loss.detach().item(), 6), "accuracy": round(accuracy, 6),
        "retrieval_recall_at_1": None if recall1 is None else round(recall1, 6),
        "retrieval_recall_at_k": None if recallk is None else round(recallk, 6),
        "train_seconds": round(time.perf_counter() - start, 3),
        "peak_train_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 41])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--corpora", nargs="+", choices=CORPORA, default=list(CORPORA))
    parser.add_argument("--train-tokens", type=int, default=131072)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--retrieval-top-k", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output", default="study_results.csv")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The study requires CUDA")
    if args.train_tokens <= 0:
        raise ValueError("--train-tokens must be positive")
    device = torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32 = True
    rows: list[dict[str, object]] = []
    total = len(args.tokens) * len(args.seeds) * len(args.variants) * len(args.corpora)
    print(f"GPU={torch.cuda.get_device_name(device)} conditions={total}; equal train-token budget={args.train_tokens}")
    for tokens in args.tokens:
        args.tokens = tokens
        for corpus in args.corpora:
            for seed in args.seeds:
                for variant in args.variants:
                    row = run_condition(variant, corpus, seed, args, device)
                    rows.append(row)
                    print(" ".join(f"{key}={value}" for key, value in row.items()))
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()
