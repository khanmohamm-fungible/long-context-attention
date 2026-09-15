"""Position-aware contextual bridge from retrieved documents to copy evidence."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from memory_bridge import RetrievedMemoryBatch


@dataclass(frozen=True)
class ContextualCopySources:
    """Shared source tokens plus their document-local structural metadata."""

    states: Tensor
    token_ids: Tensor
    mask: Tensor
    positions: Tensor
    document_ids: Tensor


def prepare_contextual_copy_sources(memory: RetrievedMemoryBatch, batch_size: int) -> ContextualCopySources:
    """Flatten documents while retaining token positions and document identity."""
    states, token_ids, mask = memory.copy_sources(batch_size)
    documents, source_tokens = memory.source_input_ids.shape
    positions = torch.arange(source_tokens, device=states.device).repeat(documents)
    document_ids = torch.arange(documents, device=states.device).repeat_interleave(source_tokens)
    return ContextualCopySources(
        states=states,
        token_ids=token_ids,
        mask=mask,
        positions=positions.unsqueeze(0).expand(batch_size, -1),
        document_ids=document_ids.unsqueeze(0).expand(batch_size, -1),
    )


class ContextualSourceEncoder(nn.Module):
    """Small bidirectional evidence encoder used only for retrieved tokens.

    It adds document-local position and document-ID embeddings, then lets
    source tokens attend to the other source tokens. The resulting keys encode
    relations such as ``item 14 -> code 15595`` rather than a bare digit ID.
    """

    def __init__(self, d_model: int, num_heads: int, *, max_source_tokens: int = 256, max_documents: int = 32) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.position_embedding = nn.Embedding(max_source_tokens, d_model)
        self.document_embedding = nn.Embedding(max_documents, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, states: Tensor, mask: Tensor, positions: Tensor, document_ids: Tensor) -> Tensor:
        if states.ndim != 3:
            raise ValueError("states must be [batch, source_tokens, d_model]")
        if mask.shape != states.shape[:2] or positions.shape != mask.shape or document_ids.shape != mask.shape:
            raise ValueError("source mask, positions, and document IDs must be [batch, source_tokens]")
        if positions.min() < 0 or positions.max() >= self.position_embedding.num_embeddings:
            raise ValueError("source position exceeds max_source_tokens")
        if document_ids.min() < 0 or document_ids.max() >= self.document_embedding.num_embeddings:
            raise ValueError("document ID exceeds max_documents")
        valid = mask.to(device=states.device, dtype=torch.bool)
        x = states + self.position_embedding(positions.to(states.device)) + self.document_embedding(document_ids.to(states.device))
        x = self.encoder(x, src_key_padding_mask=~valid)
        return self.output_norm(x) * valid.unsqueeze(-1).to(x.dtype)
