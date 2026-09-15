"""GPU-first, retrieval-grounded training smoke test.

This verifies CUDA training, mixed precision, external retrieval, and
held-out evaluation. It is a proof-of-concept, not a language-quality claim.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer

from dataset import make_synthetic_retrieval_data
from memory_bridge import build_retrieved_memory
from mini_hybrid_lm import MiniHybridLM, causal_cross_entropy, make_next_token_batch
from retrieval import FAISSVectorRetriever
from source_bridge import prepare_contextual_copy_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--synthetic-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--state-backend", choices=("torch", "fla"), default="torch")
    parser.add_argument("--no-local-attention", action="store_true")
    parser.add_argument("--oracle-retrieval", action="store_true", help="Supply the known correct external document to isolate reader quality.")
    return parser.parse_args()


def _to_device(batch: object, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch.input_ids.to(device),  # type: ignore[attr-defined]
        batch.target_ids.to(device),  # type: ignore[attr-defined]
        batch.input_mask.to(device),  # type: ignore[attr-defined]
        batch.target_mask.to(device),  # type: ignore[attr-defined]
    )


def _answer_target_mask(target_mask: torch.Tensor, prompt_tokens: int) -> torch.Tensor:
    """Keep only targets belonging to the answer, not predictable prompt text."""
    positions = torch.arange(target_mask.shape[1], device=target_mask.device) + 1
    return target_mask & (positions.unsqueeze(0) >= prompt_tokens)


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This proof-of-concept is GPU-only. Activate the CUDA PyTorch .venv and use --device cuda.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    knowledge, examples = make_synthetic_retrieval_data(args.synthetic_size)
    split = int(len(examples) * 0.8)
    train_examples, validation_examples = examples[:split], examples[split:]

    retriever = FAISSVectorRetriever()
    for index, text in enumerate(knowledge):
        retriever.upsert(f"fact_{index}", text, {"split": "external_knowledge"})

    def evidence_for(example: object) -> list[object]:
        if args.oracle_retrieval:
            index = int(example.document_id.removeprefix("fact_"))  # type: ignore[attr-defined]
            from retrieval import RetrievedDocument
            return [RetrievedDocument(example.document_id, knowledge[index], 0.0, {"oracle": True})]  # type: ignore[attr-defined]
        return retriever.search(example.query, top_k=args.top_k)  # type: ignore[attr-defined]

    retrieval_hits = sum(
        bool(evidence_for(example)) and evidence_for(example)[0].document_id == example.document_id
        for example in validation_examples
    )
    print(f"validation_retrieval_recall_at_1={retrieval_hits / len(validation_examples):.3%} ({retrieval_hits}/{len(validation_examples)})")

    model = MiniHybridLM(
        vocab_size=len(tokenizer), d_model=args.d_model, num_layers=args.layers,
        window_size=args.window, num_heads=4, state_backend=args.state_backend,
        use_local_attention=not args.no_local_attention,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    amp_enabled = args.precision == "bf16"
    amp = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()

    model.train()
    for step in range(args.steps):
        example = train_examples[step % len(train_examples)]
        encoded = tokenizer(example.text, return_tensors="pt")
        batch = make_next_token_batch(encoded["input_ids"], encoded["attention_mask"])
        input_ids, target_ids, input_mask, target_mask = _to_device(batch, device)
        prompt_tokens = tokenizer(example.prompt, return_tensors="pt")["input_ids"].shape[1]
        answer_mask = _answer_target_mask(target_mask, prompt_tokens)
        answer_positions = answer_mask[0].nonzero(as_tuple=False).squeeze(-1)
        # Querying uses no answer tokens. Evidence lives outside the LM input.
        retrieved = evidence_for(example)
        memory = build_retrieved_memory(retrieved, tokenizer, model.embedding, batch_size=1, device=device)
        sources = prepare_contextual_copy_sources(memory, batch_size=1)
        optimizer.zero_grad(set_to_none=True)
        with amp():
            logits = model(input_ids, token_mask=input_mask, retrieved_memory=memory.retrieved_memory, retrieved_mask=memory.retrieved_mask, source_states=sources.states, source_token_ids=sources.token_ids, source_mask=sources.mask, source_positions=sources.positions, source_document_ids=sources.document_ids, output_positions=answer_positions)
            loss = causal_cross_entropy(logits, target_ids[:, answer_positions])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0:
            print(f"step={step + 1} loss={loss.item():.4f} allocated_mib={torch.cuda.memory_allocated() / 2**20:.1f}")

    model.eval()
    correct_tokens = total_tokens = exact_answers = 0
    with torch.inference_mode(), amp():
        for example in validation_examples:
            encoded = tokenizer(example.prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            memory = build_retrieved_memory(evidence_for(example), tokenizer, model.embedding, batch_size=1, device=device)
            sources = prepare_contextual_copy_sources(memory, batch_size=1)
            expected = tokenizer(example.answer, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
            generated: list[torch.Tensor] = []
            for expected_token in expected:
                input_mask = torch.ones_like(input_ids, dtype=torch.bool)
                logits = model(input_ids, token_mask=input_mask, retrieved_memory=memory.retrieved_memory, retrieved_mask=memory.retrieved_mask, source_states=sources.states, source_token_ids=sources.token_ids, source_mask=sources.mask, source_positions=sources.positions, source_document_ids=sources.document_ids, output_positions=torch.tensor([-1], device=device))
                predicted = logits[:, -1].argmax(dim=-1)
                generated.append(predicted)
                correct_tokens += int(predicted.item() == expected_token.item())
                total_tokens += 1
                input_ids = torch.cat((input_ids, predicted[:, None]), dim=1)
            exact_answers += int(torch.equal(torch.cat(generated), expected))
    print(f"validation_answer_token_accuracy={correct_tokens / total_tokens:.3%} ({correct_tokens}/{total_tokens})")
    print(f"validation_exact_answer_accuracy={exact_answers / len(validation_examples):.3%} ({exact_answers}/{len(validation_examples)})")


if __name__ == "__main__":
    main(parse_args())
