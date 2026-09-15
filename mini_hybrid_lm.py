from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from App import HybridLongContextLayer, RMSNorm, SourceGroundedCopyHead, streaming_causal_prefill
from source_bridge import ContextualSourceEncoder


class MiniHybridLM(nn.Module):
    """Small causal language model built around the hybrid attention layer."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 2,
        window_size: int = 32,
        num_heads: int = 4,
        state_backend: str = "torch",
        use_local_attention: bool = True,
        inference_prefill_chunk_size: int | None = 4096,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or num_layers <= 0:
            raise ValueError("vocab_size and num_layers must be positive")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.embedding = nn.Embedding(vocab_size, d_model)
        if inference_prefill_chunk_size is not None and inference_prefill_chunk_size <= 0:
            raise ValueError("inference_prefill_chunk_size must be positive or None")
        self.inference_prefill_chunk_size = inference_prefill_chunk_size
        self.layers = nn.ModuleList(
            HybridLongContextLayer(
                d_model,
                window_size=window_size,
                num_heads=num_heads,
                dropout=0.0,
                state_backend=state_backend,
                use_local_attention=use_local_attention,
            )
            for _ in range(num_layers)
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.copy_head = SourceGroundedCopyHead(d_model, vocab_size, generator=self.lm_head)
        self.source_encoder = ContextualSourceEncoder(d_model, num_heads)

    def forward(
        self,
        input_ids: Tensor,
        *,
        token_mask: Tensor | None = None,
        retrieved_memory: Tensor | None = None,
        retrieved_mask: Tensor | None = None,
        source_states: Tensor | None = None,
        source_token_ids: Tensor | None = None,
        source_mask: Tensor | None = None,
        source_positions: Tensor | None = None,
        source_document_ids: Tensor | None = None,
        output_positions: Tensor | None = None,
    ) -> Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            # Long inference prompts use exact cached prefill. It bounds the
            # local-attention and portable recurrence workspaces to one 4K
            # block, while training and short prompts retain the one-shot path.
            if (
                not self.training
                and not torch.is_grad_enabled()
                and self.inference_prefill_chunk_size is not None
                and x.shape[1] > self.inference_prefill_chunk_size
            ):
                x = streaming_causal_prefill(
                    layer,
                    x,
                    chunk_size=self.inference_prefill_chunk_size,
                    token_mask=token_mask,
                    retrieved_memory=retrieved_memory,
                    retrieved_mask=retrieved_mask,
                )
            else:
                x = layer(
                    x,
                    causal=True,
                    token_mask=token_mask,
                    retrieved_memory=retrieved_memory,
                    retrieved_mask=retrieved_mask,
                )
        hidden = self.final_norm(x)
        copy_inputs = (source_states, source_token_ids, source_mask)
        if any(item is not None for item in copy_inputs):
            if any(item is None for item in copy_inputs):
                raise ValueError("source_states, source_token_ids, and source_mask must be supplied together")
            if (source_positions is None) != (source_document_ids is None):
                raise ValueError("source_positions and source_document_ids must be supplied together")
            if source_positions is not None:
                source_states = self.source_encoder(source_states, source_mask, source_positions, source_document_ids)
            return self.copy_head(hidden, source_states, source_token_ids, source_mask, output_positions=output_positions)
        if output_positions is not None:
            output_positions = output_positions.to(device=hidden.device, dtype=torch.long)
            output_positions = torch.where(output_positions < 0, output_positions + hidden.shape[1], output_positions)
            hidden = hidden.index_select(1, output_positions)
        return self.lm_head(hidden)


@dataclass
class NextTokenBatch:
    input_ids: Tensor
    target_ids: Tensor
    input_mask: Tensor
    target_mask: Tensor

    @property
    def token_mask(self) -> Tensor:
        """Backward-compatible alias for the loss/target mask."""
        return self.target_mask


def make_next_token_batch(input_ids: Tensor, attention_mask: Tensor | None = None) -> NextTokenBatch:
    """Create teacher-forcing inputs and next-token targets from token IDs."""
    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("input_ids must have shape [batch, tokens] with at least two tokens")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
    return NextTokenBatch(
        input_ids=input_ids[:, :-1],
        target_ids=input_ids[:, 1:],
        # Inputs and labels are shifted independently.  With right padding,
        # the final real input predicts a padded label and must not contribute
        # to loss, but it is still a valid attention key/value.
        input_mask=attention_mask[:, :-1],
        target_mask=attention_mask[:, 1:],
    )


def causal_cross_entropy(logits: Tensor, targets: Tensor, token_mask: Tensor | None = None) -> Tensor:
    """Compute cross-entropy while ignoring padded target positions."""
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("logits must be [batch, tokens, vocab] and targets must match its first two dimensions")
    losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    if token_mask is None:
        return losses.mean()
    token_mask = token_mask.to(dtype=losses.dtype, device=losses.device)
    return (losses * token_mask).sum() / token_mask.sum().clamp_min(1)
