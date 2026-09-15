from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from retrieval import RetrievedDocument


@dataclass
class RetrievedMemoryBatch:
	"""Tensor forms of retrieved documents for the hybrid model."""

	retrieved_memory: Tensor
	retrieved_mask: Tensor
	source_input_ids: Tensor
	source_attention_mask: Tensor
	source_states: Tensor

	def to(self, device: torch.device | str) -> "RetrievedMemoryBatch":
		"""Move every model-facing retrieval tensor to ``device``."""
		return RetrievedMemoryBatch(
			self.retrieved_memory.to(device),
			self.retrieved_mask.to(device),
			self.source_input_ids.to(device),
			self.source_attention_mask.to(device),
			self.source_states.to(device),
		)

	def copy_sources(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
		"""Return token-level evidence shared by every target position."""
		if batch_size <= 0:
			raise ValueError("batch_size must be positive")
		documents, source_tokens, width = self.source_states.shape
		states = self.source_states.reshape(1, documents * source_tokens, width).expand(batch_size, -1, -1)
		ids = self.source_input_ids.reshape(1, documents * source_tokens).expand(batch_size, -1)
		mask = self.source_attention_mask.reshape(1, documents * source_tokens).expand(batch_size, -1)
		return states, ids, mask


def build_retrieved_memory(
	documents: list[RetrievedDocument],
	tokenizer: object,
	embedding: nn.Embedding,
	*,
	batch_size: int,
	max_source_tokens: int = 128,
	device: torch.device | str | None = None,
) -> RetrievedMemoryBatch:
	"""Convert retrieved raw documents into model-sized evidence tensors."""
	if not documents:
		raise ValueError("documents must contain at least one retrieved document")
	if batch_size <= 0:
		raise ValueError("batch_size must be positive")
	if max_source_tokens <= 0:
		raise ValueError("max_source_tokens must be positive")

	encoded = tokenizer(
		[document.text for document in documents],
		padding=True,
		truncation=True,
		max_length=max_source_tokens,
		return_tensors="pt",
	)
	# Tokenizers return CPU tensors.  Embedding indices must be on the same
	# device as the embedding parameters for CUDA training.
	device = torch.device(device) if device is not None else embedding.weight.device
	source_input_ids = encoded["input_ids"].to(device)
	source_attention_mask = encoded["attention_mask"].to(device=device, dtype=torch.bool)
	source_states = embedding(source_input_ids)

	masked_states = source_states * source_attention_mask.unsqueeze(-1).to(source_states.dtype)
	valid_count = source_attention_mask.sum(dim=1, keepdim=True).clamp_min(1).to(source_states.dtype)
	pooled_states = masked_states.sum(dim=1) / valid_count

	retrieved_memory = pooled_states.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
	retrieved_mask = torch.ones(
		batch_size,
		len(documents),
		dtype=torch.bool,
		device=source_states.device,
	)

	return RetrievedMemoryBatch(
		retrieved_memory=retrieved_memory,
		retrieved_mask=retrieved_mask,
		source_input_ids=source_input_ids,
		source_attention_mask=source_attention_mask,
		source_states=source_states,
	)

