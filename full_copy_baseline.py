"""Parameter-small full-attention pointer-reader baseline for the RAG smoke test."""
from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoTokenizer

from App import RMSNorm, SourceGroundedCopyHead
from dataset import make_synthetic_retrieval_data
from mini_hybrid_lm import causal_cross_entropy, make_next_token_batch


class FullCopyLM(nn.Module):
    """A full-causal-attention reader with the identical pointer-generator."""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int = 4, max_positions: int = 256) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positions = nn.Embedding(max_positions, d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = RMSNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.copy_head = SourceGroundedCopyHead(d_model, vocab_size, generator=self.lm_head)
        self.num_heads, self.head_dim = num_heads, d_model // num_heads

    def forward(self, source_ids: Tensor, text_ids: Tensor) -> Tensor:
        all_ids = torch.cat((source_ids, text_ids), dim=1)
        if all_ids.shape[1] > self.positions.num_embeddings:
            raise ValueError("sequence exceeds max_positions")
        position_ids = torch.arange(all_ids.shape[1], device=all_ids.device)
        x = self.embedding(all_ids) + self.positions(position_ids)[None]
        batch, tokens, width = x.shape
        q, k, v = self.qkv(self.norm(x)).view(batch, tokens, 3, self.num_heads, self.head_dim).unbind(2)
        attended = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + self.out(attended.transpose(1, 2).reshape(batch, tokens, width))
        x = x + self.mlp(self.norm(x))
        source_tokens = source_ids.shape[1]
        return self.copy_head(self.final_norm(x[:, source_tokens:]), x[:, :source_tokens], source_ids, torch.ones_like(source_ids, dtype=torch.bool))


def answer_mask(mask: Tensor, prompt_tokens: int) -> Tensor:
    return mask & ((torch.arange(mask.shape[1], device=mask.device) + 1)[None] >= prompt_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--synthetic-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(7)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    knowledge, examples = make_synthetic_retrieval_data(args.synthetic_size)
    split = int(.8 * len(examples))
    train, validation = examples[:split], examples[split:]
    model = FullCopyLM(len(tokenizer), args.d_model).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    amp = lambda: torch.autocast("cuda", torch.bfloat16)
    model.train()
    for step in range(args.steps):
        ex = train[step % len(train)]
        encoded = tokenizer(ex.text, return_tensors="pt")
        batch = make_next_token_batch(encoded.input_ids, encoded.attention_mask)
        source = tokenizer(knowledge[int(ex.document_id.removeprefix("fact_"))], return_tensors="pt").input_ids.cuda()
        inputs, targets = batch.input_ids.cuda(), batch.target_ids.cuda()
        mask = answer_mask(batch.target_mask.cuda(), tokenizer(ex.prompt, return_tensors="pt").input_ids.shape[1])
        optimizer.zero_grad(set_to_none=True)
        with amp():
            logits = model(source, inputs)
            loss = causal_cross_entropy(logits, targets, mask)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 0 or (step + 1) % 25 == 0:
            print(f"step={step + 1} loss={loss.item():.4f}")
    model.eval(); correct = total = exact = 0
    with torch.inference_mode(), amp():
        for ex in validation:
            source = tokenizer(knowledge[int(ex.document_id.removeprefix("fact_"))], return_tensors="pt").input_ids.cuda()
            ids = tokenizer(ex.prompt, return_tensors="pt").input_ids.cuda()
            expected = tokenizer(ex.answer, add_special_tokens=False, return_tensors="pt").input_ids[0].cuda()
            generated = []
            for token in expected:
                pred = model(source, ids)[:, -1].argmax(-1)
                generated.append(pred); correct += int(pred.item() == token.item()); total += 1
                ids = torch.cat((ids, pred[:, None]), dim=1)
            exact += int(torch.equal(torch.cat(generated), expected))
    print(f"oracle_full_attention_answer_token_accuracy={correct / total:.3%} ({correct}/{total})")
    print(f"oracle_full_attention_exact_answer_accuracy={exact / len(validation):.3%} ({exact}/{len(validation)})")


if __name__ == "__main__":
    main()
