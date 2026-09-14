"""Practical hybrid long-context layer.

This is not a claim that one compressed state replaces all attention. It uses:
  * sliding-window attention with RoPE for exact nearby token interactions,
  * bidirectional or causal bounded state mixing for long-range context, and
  * retrieved, uncompressed top-k evidence for precise distant fact recall.

The current PyTorch scan fallback is O(N log N) work (sub-quadratic), not
O(N). Replace it with an associative-scan CUDA/Triton kernel for true O(N)
work when profiling shows it is a bottleneck.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from torch._higher_order_ops.associative_scan import associative_scan as _associative_scan
except ImportError:  # Older PyTorch builds do not expose the prototype API.
    _associative_scan = None


@dataclass
class LocalKVCache:
    """The last ``window_size - 1`` RoPE-transformed local-attention KV pairs."""

    keys: Tensor
    values: Tensor
    valid: Tensor
    next_position: Tensor

    def detach(self) -> "LocalKVCache":
        return LocalKVCache(self.keys.detach(), self.values.detach(), self.valid, self.next_position)


@dataclass
class HybridCache:
    """Streaming state for a causal ``HybridLongContextLayer`` call."""

    state: Tensor
    local_kv: LocalKVCache

    def detach(self) -> "HybridCache":
        return HybridCache(self.state.detach(), self.local_kv.detach())


@dataclass(frozen=True)
class RetrievedDocument:
    """Raw, attributable evidence returned by the lexical index."""

    document_id: str
    text: str
    score: float
    metadata: dict[str, object]


class RMSNorm(nn.Module):
    """Small RMSNorm implementation, compatible with older PyTorch releases."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)
        return x * scale * self.weight.to(dtype=x.dtype)


def _parallel_affine_scan(decay: Tensor, update: Tensor, initial_state: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate ``h_t = decay_t * h_(t-1) + update_t`` by a parallel scan.

    This portable Hillis-Steele fallback has O(log N) vectorized stages and
    O(N log N) work. It eliminates the Python loop over individual tokens.
    """
    _, tokens, _ = decay.shape
    if tokens == 0:
        return update, initial_state
    work_dtype = torch.float32 if decay.dtype in (torch.float16, torch.bfloat16) else decay.dtype
    a, b = decay.to(work_dtype), update.to(work_dtype)
    offset = 1
    while offset < tokens:
        current_a = a[:, offset:]
        a = torch.cat((a[:, :offset], current_a * a[:, :-offset]), dim=1)
        b = torch.cat((b[:, :offset], b[:, offset:] + current_a * b[:, :-offset]), dim=1)
        offset *= 2
    h0 = initial_state.to(work_dtype).unsqueeze(1)
    states = b + a * h0
    return states.to(update.dtype), states[:, -1].to(update.dtype)


def _affine_combine(left: tuple[Tensor, Tensor], right: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """Associative composition of two affine recurrence segments.

    A segment is represented as ``(a, b)`` for ``h_out = a * h_in + b``.
    """
    left_a, left_b = left
    right_a, right_b = right
    return right_a * left_a, right_b + right_a * left_b


def fused_affine_scan_inference(decay: Tensor, update: Tensor, initial_state: Tensor) -> tuple[Tensor, Tensor]:
    """CUDA associative-scan path for inference only.

    PyTorch's associative scan is currently a prototype: it requires CUDA plus
    ``torch.compile`` code generation and should be guarded by correctness tests
    on each PyTorch upgrade.  This function deliberately refuses training,
    where the portable autograd-safe fallback remains the supported path.
    """
    if not decay.is_cuda:
        raise RuntimeError("fused_affine_scan_inference requires CUDA tensors")
    if torch.is_grad_enabled():
        raise RuntimeError("fused associative scan is inference-only; call under torch.inference_mode()")
    if _associative_scan is None:
        raise RuntimeError("this PyTorch build does not provide associative_scan")
    prefix_a, prefix_b = _associative_scan(
        _affine_combine, (decay, update), dim=1, combine_mode="pointwise"
    )
    h0 = initial_state.to(prefix_b.dtype).unsqueeze(1)
    states = prefix_b + prefix_a * h0
    return states.to(update.dtype), states[:, -1].to(update.dtype)


def _naive_affine_scan(decay: Tensor, update: Tensor, initial_state: Tensor) -> tuple[Tensor, Tensor]:
    """Reference implementation used only by the smoke test."""
    h = initial_state
    states: list[Tensor] = []
    for token in range(decay.shape[1]):
        h = decay[:, token] * h + update[:, token]
        states.append(h)
    return torch.stack(states, dim=1), h


class ParallelGatedStateMixer(nn.Module):
    """Stable long-range mixer; bidirectional mode uses both sequence directions."""

    def __init__(self, d_model: int, min_decay: float = 0.005, max_decay: float = 0.995, *, use_fused_scan: bool = False) -> None:
        super().__init__()
        if not 0.0 <= min_decay < max_decay < 1.0:
            raise ValueError("decays must satisfy 0 <= min < max < 1")
        self.min_decay, self.max_decay = min_decay, max_decay
        self.use_fused_scan = use_fused_scan
        self.decay_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.forward_out = nn.Linear(d_model, d_model, bias=False)
        self.backward_out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        *,
        bidirectional: bool,
        initial_state: Tensor | None,
        token_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, _, width = x.shape
        if initial_state is None:
            initial_state = x.new_zeros(batch, width)
        if initial_state.shape != (batch, width):
            raise ValueError(f"state must have shape {(batch, width)}, got {tuple(initial_state.shape)}")
        initial_state = initial_state.to(device=x.device, dtype=x.dtype).clamp(-1.0, 1.0)

        decay = self.min_decay + (self.max_decay - self.min_decay) * torch.sigmoid(self.decay_proj(x))
        # This is a bounded exponential moving average, preventing state
        # explosion. Exact facts travel through retrieval, not this state.
        update = (1.0 - decay) * torch.tanh(self.value_proj(x))
        mask = None
        if token_mask is not None:
            if token_mask.shape != x.shape[:2]:
                raise ValueError("token_mask must be [batch, tokens]")
            mask = token_mask.to(device=x.device, dtype=torch.bool).unsqueeze(-1)
            decay = torch.where(mask, decay, torch.ones_like(decay))
            update = torch.where(mask, update, torch.zeros_like(update))
        scan = fused_affine_scan_inference if self.use_fused_scan else _parallel_affine_scan
        forward_states, final_state = scan(decay, update, initial_state)
        mixed = self.forward_out(forward_states)
        if bidirectional:
            reverse_states, _ = scan(
                torch.flip(decay, (1,)), torch.flip(update, (1,)), x.new_zeros(batch, width)
            )
            mixed = mixed + self.backward_out(torch.flip(reverse_states, (1,)))
        if mask is not None:
            mixed = mixed * mask
        return mixed, final_state


class SlidingWindowAttention(nn.Module):
    """RoPE local attention with an exact causal KV-window cache."""

    def __init__(self, d_model: int, num_heads: int, window_size: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even per-head dimension")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size, self.dropout = window_size, dropout
        self.qkv, self.out = nn.Linear(d_model, 3 * d_model), nn.Linear(d_model, d_model)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, x: Tensor, position_ids: Tensor) -> Tensor:
        angles = position_ids.float().unsqueeze(-1) * self.inv_freq
        cos, sin = angles.cos().unsqueeze(2), angles.sin().unsqueeze(2)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def _positions(self, batch: int, tokens: int, device: torch.device, position_ids: Tensor | None, cache: LocalKVCache | None) -> Tensor:
        if position_ids is None:
            start = torch.zeros(batch, 1, dtype=torch.long, device=device) if cache is None else cache.next_position.to(device).unsqueeze(1)
            return start + torch.arange(tokens, device=device)
        if position_ids.ndim == 1:
            if position_ids.shape[0] != tokens:
                raise ValueError("position_ids must have one entry per token")
            return position_ids.to(device).expand(batch, -1)
        if position_ids.shape != (batch, tokens):
            raise ValueError("position_ids must be [tokens] or [batch, tokens]")
        return position_ids.to(device)

    def _initial_cache(self, batch: int, x: Tensor) -> LocalKVCache:
        past = self.window_size - 1
        empty_kv = x.new_zeros(batch, past, self.num_heads, self.head_dim)
        return LocalKVCache(
            empty_kv,
            empty_kv.clone(),
            torch.zeros(batch, past, dtype=torch.bool, device=x.device),
            torch.zeros(batch, dtype=torch.long, device=x.device),
        )

    def _validate_cache(self, cache: LocalKVCache, batch: int, x: Tensor) -> None:
        expected = (batch, self.window_size - 1, self.num_heads, self.head_dim)
        if cache.keys.shape != expected or cache.values.shape != expected or cache.valid.shape != expected[:2]:
            raise ValueError("KV cache does not match this batch size or attention configuration")

    def _compact_cache(
        self,
        keys: Tensor,
        values: Tensor,
        valid: Tensor,
        next_position: Tensor,
    ) -> LocalKVCache:
        past = self.window_size - 1
        if past == 0:
            return LocalKVCache(
                keys[:, :0].clone(),
                values[:, :0].clone(),
                valid[:, :0].clone(),
                next_position,
            )
        batch, available = valid.shape
        take = min(past, available)
        positions = torch.arange(available, device=valid.device).expand(batch, -1)
        selected = positions.masked_fill(~valid, -1).topk(
            take,
            dim=1,
        ).values.flip(1)
        cache_indices = torch.zeros(
            batch,
            past,
            dtype=torch.long,
            device=valid.device,
        )
        cache_indices[:, past - take :] = selected.clamp_min(0)
        cache_valid = torch.zeros(
            batch,
            past,
            dtype=torch.bool,
            device=valid.device,
        )
        cache_valid[:, past - take :] = selected >= 0
        kv_indices = cache_indices[:, :, None, None].expand(
            -1,
            -1,
            self.num_heads,
            self.head_dim,
        )
        cache_keys = keys.gather(1, kv_indices)
        cache_values = values.gather(1, kv_indices)
        kv_valid = cache_valid[:, :, None, None]
        return LocalKVCache(
            cache_keys.masked_fill(~kv_valid, 0),
            cache_values.masked_fill(~kv_valid, 0),
            cache_valid,
            next_position,
        )

    def _attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        valid: Tensor | None,
        *,
        query_offset: int,
        left: int,
        right: int,
    ) -> Tensor:
        """Apply SDPA to bounded key ranges without constructing N x W windows."""
        tokens = q.shape[1]
        chunk_size = min(tokens, max(16, min(self.window_size, 256)))
        attended = torch.empty_like(q)
        dropout_p = self.dropout if self.training else 0.0

        for query_start in range(0, tokens, chunk_size):
            query_end = min(query_start + chunk_size, tokens)
            key_start = max(0, query_offset + query_start - left)
            key_end = min(k.shape[1], query_offset + query_end + right)
            query_positions = torch.arange(
                query_offset + query_start,
                query_offset + query_end,
                device=q.device,
            )
            key_positions = torch.arange(key_start, key_end, device=q.device)
            local_mask = (
                (key_positions.unsqueeze(0) >= query_positions.unsqueeze(1) - left)
                & (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1) + right)
            )
            if valid is None:
                attention_mask = local_mask
            else:
                attention_mask = (
                    local_mask.unsqueeze(0)
                    & valid[:, None, key_start:key_end]
                ).unsqueeze(1)
            chunk = F.scaled_dot_product_attention(
                q[:, query_start:query_end].transpose(1, 2),
                k[:, key_start:key_end].transpose(1, 2),
                v[:, key_start:key_end].transpose(1, 2),
                attn_mask=attention_mask,
                dropout_p=dropout_p,
            )
            attended[:, query_start:query_end] = chunk.transpose(1, 2)
        return attended

    def forward(
        self,
        x: Tensor,
        *,
        causal: bool,
        position_ids: Tensor | None,
        kv_cache: LocalKVCache | None = None,
        token_mask: Tensor | None = None,
    ) -> tuple[Tensor, LocalKVCache | None]:
        batch, tokens, width = x.shape
        if tokens == 0:
            return x, kv_cache
        if not causal and kv_cache is not None:
            raise ValueError("bidirectional attention cannot reuse a causal KV cache")
        if token_mask is None:
            current_valid = torch.ones(
                batch,
                tokens,
                dtype=torch.bool,
                device=x.device,
            )
        else:
            if token_mask.shape != (batch, tokens):
                raise ValueError("token_mask must be [batch, tokens]")
            current_valid = token_mask.to(device=x.device, dtype=torch.bool)
        position_ids = self._positions(batch, tokens, x.device, position_ids, kv_cache)
        qkv = self.qkv(x).view(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k = self._rope(q, position_ids), self._rope(k, position_ids)

        if causal:
            past = self.window_size - 1
            if kv_cache is None:
                all_k, all_v = k, v
                all_valid = current_valid
                attention_valid = None if token_mask is None else current_valid
                query_offset = 0
            else:
                self._validate_cache(kv_cache, batch, x)
                all_k = torch.cat((kv_cache.keys.to(x), k), dim=1)
                all_v = torch.cat((kv_cache.values.to(x), v), dim=1)
                all_valid = torch.cat(
                    (kv_cache.valid.to(x.device), current_valid),
                    dim=1,
                )
                attention_valid = all_valid
                query_offset = past
            attended = self._attend(
                q,
                all_k,
                all_v,
                attention_valid,
                query_offset=query_offset,
                left=past,
                right=0,
            )
            output = self.out(attended.reshape(batch, tokens, width))
            if token_mask is not None:
                output = output * current_valid.unsqueeze(-1)
            valid_indices = torch.arange(tokens, device=x.device).expand(batch, -1)
            valid_indices = valid_indices.masked_fill(~current_valid, -1).amax(dim=1)
            last_valid_position = position_ids.gather(
                1,
                valid_indices.clamp_min(0).unsqueeze(1),
            ).squeeze(1)
            previous_position = (
                torch.zeros(batch, dtype=torch.long, device=x.device)
                if kv_cache is None
                else kv_cache.next_position.to(x.device)
            )
            next_position = torch.where(
                valid_indices >= 0,
                last_valid_position + 1,
                previous_position,
            )
            next_cache = self._compact_cache(
                all_k,
                all_v,
                all_valid,
                next_position,
            )
            return output, next_cache
        left, right = self.window_size // 2, self.window_size - self.window_size // 2 - 1
        attended = self._attend(
            q,
            k,
            v,
            None if token_mask is None else current_valid,
            query_offset=0,
            left=left,
            right=right,
        )
        output = self.out(attended.reshape(batch, tokens, width))
        if token_mask is not None:
            output = output * current_valid.unsqueeze(-1)
        return output, None


class RetrievedEvidenceAttention(nn.Module):
    """Attend over shared or per-token evidence in bounded query chunks."""

    def __init__(self, d_model: int, query_chunk_size: int = 256) -> None:
        super().__init__()
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
        self.query_chunk_size = query_chunk_size

    def forward(self, x: Tensor, retrieved_memory: Tensor, retrieved_mask: Tensor | None) -> Tensor:
        shared_memory = retrieved_memory.ndim == 3
        if shared_memory:
            if retrieved_memory.shape[0] != x.shape[0] or retrieved_memory.shape[2] != x.shape[2]:
                raise ValueError("shared retrieved_memory must be [batch, top_k, d_model]")
        elif (
            retrieved_memory.ndim != 4
            or retrieved_memory.shape[:2] != x.shape[:2]
            or retrieved_memory.shape[3] != x.shape[2]
        ):
            raise ValueError(
                "retrieved_memory must be [batch, top_k, d_model] or "
                "[batch, tokens, top_k, d_model]"
            )
        if retrieved_memory.shape[-2] == 0:
            raise ValueError("retrieved_memory must contain at least one item")
        memory = retrieved_memory.to(device=x.device, dtype=x.dtype)
        if retrieved_mask is not None:
            if retrieved_mask.shape != memory.shape[:-1]:
                raise ValueError("retrieved_mask must match retrieved_memory without d_model")
            retrieved_mask = retrieved_mask.to(device=x.device, dtype=torch.bool)

        queries = self.query(x)
        evidence = torch.empty_like(x)
        if shared_memory:
            shared_keys = self.key(memory)
            shared_values = self.value(memory)
        for start in range(0, x.shape[1], self.query_chunk_size):
            end = min(start + self.query_chunk_size, x.shape[1])
            if shared_memory:
                keys, values = shared_keys, shared_values
                scores = queries[:, start:end] @ keys.transpose(-1, -2)
                mask = None if retrieved_mask is None else retrieved_mask[:, None]
            else:
                chunk_memory = memory[:, start:end]
                keys, values = self.key(chunk_memory), self.value(chunk_memory)
                scores = (queries[:, start:end].unsqueeze(2) * keys).sum(dim=-1)
                mask = None if retrieved_mask is None else retrieved_mask[:, start:end]
            scores = scores * self.scale
            if mask is None:
                weights = torch.softmax(scores, dim=-1)
            else:
                weights = torch.softmax(
                    scores.masked_fill(~mask, torch.finfo(scores.dtype).min),
                    dim=-1,
                )
                weights = weights * mask.to(weights.dtype)
                weights = weights / weights.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(torch.finfo(weights.dtype).eps)
            if shared_memory:
                evidence[:, start:end] = weights @ values
            else:
                evidence[:, start:end] = (
                    weights.unsqueeze(-1) * values
                ).sum(dim=2)
        return torch.sigmoid(self.gate(x)) * evidence


class BruteForceTopKRetriever(nn.Module):
    """Small live demo retriever; do not use it for production-scale indexes.

    It returns evidence and a validity mask. Replace it in production with an
    ANN, lexical, or ID lookup service and pass its output to the main layer.
    """

    def __init__(self, top_k: int) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = top_k

    def forward(self, queries: Tensor, memory_bank: Tensor, memory_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if memory_bank.ndim != 3 or memory_bank.shape[0] != queries.shape[0] or memory_bank.shape[2] != queries.shape[2]:
            raise ValueError("memory_bank must be [batch, memory_tokens, d_model]")
        batch, tokens, width = queries.shape
        memory_tokens = memory_bank.shape[1]
        if memory_tokens == 0:
            raise ValueError("memory_bank must contain at least one vector")
        if memory_mask is None:
            memory_mask = torch.ones(batch, memory_tokens, dtype=torch.bool, device=queries.device)
        else:
            memory_mask = memory_mask.to(device=queries.device, dtype=torch.bool)
            if memory_mask.shape != (batch, memory_tokens):
                raise ValueError("memory_mask must be [batch, memory_tokens]")
        scores = F.normalize(queries, dim=-1) @ F.normalize(memory_bank.to(queries), dim=-1).transpose(-1, -2)
        scores = scores.masked_fill(~memory_mask[:, None], torch.finfo(scores.dtype).min)
        k = min(self.top_k, memory_tokens)
        indices = scores.topk(k, dim=-1).indices
        bank = memory_bank[:, None].expand(-1, tokens, -1, -1)
        evidence = bank.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, width))
        valid = memory_mask[:, None].expand(-1, tokens, -1).gather(2, indices)
        return evidence, valid


class SQLiteFTSRetriever:
    """Persistent lexical retrieval with raw text and provenance.

    SQLite FTS5 is a real, dependency-free retrieval backend suitable for a
    demo or a small private corpus. It performs exact lexical matching, unlike
    the vector-only demo retriever. For a large corpus, keep this API and swap
    the implementation for a managed BM25/ANN service plus a reranker.
    """

    def __init__(self, database_path: str | Path) -> None:
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS documents (document_id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata_json TEXT NOT NULL)")
        self.connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(document_id UNINDEXED, text, tokenize='unicode61')")

    def close(self) -> None:
        self.connection.close()

    def upsert(self, document_id: str, text: str, metadata: dict[str, object] | None = None) -> None:
        if not document_id or not text:
            raise ValueError("document_id and text must be non-empty")
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        with self.connection:
            self.connection.execute("DELETE FROM documents_fts WHERE document_id = ?", (document_id,))
            self.connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            self.connection.execute("INSERT INTO documents(document_id, text, metadata_json) VALUES (?, ?, ?)", (document_id, text, metadata_json))
            self.connection.execute("INSERT INTO documents_fts(document_id, text) VALUES (?, ?)", (document_id, text))

    def search(self, query: str, top_k: int = 4) -> list[RetrievedDocument]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        # Quote individual terms so ordinary user punctuation cannot alter FTS
        # query syntax. This is lexical retrieval, not semantic retrieval.
        terms = re.findall(r"\w+", query, flags=re.UNICODE)
        if not terms:
            return []
        fts_query = " AND ".join(f'"{term}"' for term in terms)
        rows = self.connection.execute(
            "SELECT d.document_id, d.text, d.metadata_json, bm25(documents_fts) "
            "FROM documents_fts JOIN documents d USING(document_id) "
            "WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts) LIMIT ?",
            (fts_query, top_k),
        ).fetchall()
        return [RetrievedDocument(row[0], row[1], float(row[3]), json.loads(row[2])) for row in rows]


class SourceGroundedCopyHead(nn.Module):
    """Mix vocabulary logits with an exact pointer distribution over source IDs.

    ``source_token_ids`` are tokenizer IDs from retrieved raw documents, not
    vectors. A selected source token receives probability mass by ``scatter_add``
    and can be decoded back to user-visible text with its document provenance.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size
        self.generator = nn.Linear(d_model, vocab_size)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.copy_gate = nn.Linear(d_model, 1)
        self.scale = d_model ** -0.5

    def forward(
        self,
        hidden: Tensor,
        source_states: Tensor,
        source_token_ids: Tensor,
        source_mask: Tensor,
        *,
        output_positions: Tensor | None = None,
    ) -> Tensor:
        if source_states.ndim != 4 or source_states.shape[:2] != hidden.shape[:2] or source_states.shape[-1] != hidden.shape[-1]:
            raise ValueError("source_states must be [batch, tokens, source_tokens, d_model]")
        if source_states.shape[2] == 0:
            raise ValueError("source_states must contain at least one source token")
        if source_token_ids.shape != source_states.shape[:-1] or source_mask.shape != source_token_ids.shape:
            raise ValueError("source token IDs and mask must match source_states")
        if source_token_ids.min() < 0 or source_token_ids.max() >= self.vocab_size:
            raise ValueError("source_token_ids must be valid vocabulary IDs")
        if output_positions is not None:
            if output_positions.ndim != 1:
                raise ValueError("output_positions must be a one-dimensional tensor")
            output_positions = output_positions.to(device=hidden.device, dtype=torch.long)
            output_positions = torch.where(
                output_positions < 0,
                output_positions + hidden.shape[1],
                output_positions,
            )
            if (
                output_positions.numel() == 0
                or output_positions.min() < 0
                or output_positions.max() >= hidden.shape[1]
            ):
                raise ValueError("output_positions contains an invalid token index")
            hidden = hidden.index_select(1, output_positions)
            source_states = source_states.index_select(
                1,
                output_positions.to(source_states.device),
            )
            source_token_ids = source_token_ids.index_select(
                1,
                output_positions.to(source_token_ids.device),
            )
            source_mask = source_mask.index_select(
                1,
                output_positions.to(source_mask.device),
            )
        source_token_ids = source_token_ids.to(device=hidden.device, dtype=torch.long)
        source_states = source_states.to(hidden)
        source_mask = source_mask.to(device=hidden.device, dtype=torch.bool)
        scores = (self.query(hidden).unsqueeze(2) * self.key(source_states)).sum(-1) * self.scale
        weights = torch.softmax(scores.masked_fill(~source_mask, torch.finfo(scores.dtype).min), dim=-1)
        weights = weights * source_mask.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        copy_probs = hidden.new_zeros(*hidden.shape[:2], self.vocab_size)
        copy_probs.scatter_add_(2, source_token_ids, weights)
        generate_probs = torch.softmax(self.generator(hidden), dim=-1)
        copy_gate = torch.sigmoid(self.copy_gate(hidden))
        return torch.log((1.0 - copy_gate) * generate_probs + copy_gate * copy_probs + torch.finfo(hidden.dtype).tiny)


class HybridLongContextLayer(nn.Module):
    """Sliding attention + stable scan + optional retrieved evidence.

    Provide position IDs or let the layer generate contiguous IDs. For streamed
    causal calls, pass global rather than block-local position IDs yourself.
    Retrieved evidence may be shared across tokens as ``[batch, top_k, d_model]``
    to avoid replicating it for every query.
    """

    def __init__(self, d_model: int, window_size: int = 128, *, num_heads: int = 8, dropout: float = 0.0, mlp_ratio: int = 4, use_fused_scan: bool = False) -> None:
        super().__init__()
        if d_model <= 0 or mlp_ratio <= 0:
            raise ValueError("d_model and mlp_ratio must be positive")
        self.d_model, self.window_size = d_model, window_size
        self.local_norm, self.state_norm = RMSNorm(d_model), RMSNorm(d_model)
        self.retrieval_norm, self.mlp_norm = RMSNorm(d_model), RMSNorm(d_model)
        self.local_attention = SlidingWindowAttention(d_model, num_heads, window_size, dropout)
        self.state_mixer = ParallelGatedStateMixer(d_model, use_fused_scan=use_fused_scan)
        self.retrieval = RetrievedEvidenceAttention(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(), nn.Linear(mlp_ratio * d_model, d_model))

    def forward(
        self,
        x: Tensor,
        *,
        causal: bool = False,
        state: Tensor | None = None,
        position_ids: Tensor | None = None,
        token_mask: Tensor | None = None,
        retrieved_memory: Tensor | None = None,
        retrieved_mask: Tensor | None = None,
        kv_cache: LocalKVCache | None = None,
        cache: HybridCache | None = None,
        return_state: bool = False,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, HybridCache]:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [batch, tokens, {self.d_model}]")
        if token_mask is not None and token_mask.shape != x.shape[:2]:
            raise ValueError("token_mask must be [batch, tokens]")
        if cache is not None:
            if state is not None or kv_cache is not None:
                raise ValueError("pass either cache or state/kv_cache, not both")
            state, kv_cache = cache.state, cache.local_kv
        if not causal and (kv_cache is not None or cache is not None):
            raise ValueError("KV caching is available only for causal inference")
        local, next_kv = self.local_attention(
            self.local_norm(x),
            causal=causal,
            position_ids=position_ids,
            kv_cache=kv_cache,
            token_mask=token_mask,
        )
        x = x + local
        mixed, final_state = self.state_mixer(
            self.state_norm(x),
            bidirectional=not causal,
            initial_state=state,
            token_mask=token_mask,
        )
        x = x + mixed
        if retrieved_memory is not None:
            x = x + self.retrieval(self.retrieval_norm(x), retrieved_memory, retrieved_mask)
        output = x + self.mlp(self.mlp_norm(x))
        if token_mask is not None:
            output = output * token_mask.to(
                device=x.device,
                dtype=output.dtype,
            ).unsqueeze(-1)
        if return_cache:
            if not causal or next_kv is None:
                raise ValueError("return_cache requires causal=True")
            return output, HybridCache(final_state, next_kv)
        return (output, final_state) if return_state else output


class StableChunkedLinearSequenceLayer(HybridLongContextLayer):
    """Compatibility alias; ``chunk_size`` now means sliding attention window."""

    def __init__(self, d_model: int, chunk_size: int = 128, **kwargs: object) -> None:
        super().__init__(d_model, window_size=chunk_size, **kwargs)


def compile_for_inference(model: nn.Module, *, dynamic: bool = False) -> nn.Module:
    """Compile safely; use static shapes for fastest serving and warm each shape."""
    if not hasattr(torch, "compile"):
        raise RuntimeError("compile_for_inference requires PyTorch 2.0 or newer")
    return torch.compile(model, dynamic=dynamic)


def _smoke_test() -> None:
    torch.manual_seed(0)
    decay = 0.1 + 0.8 * torch.rand(2, 257, 8)
    update, h0 = torch.randn(2, 257, 8), torch.randn(2, 8)
    scanned, last = _parallel_affine_scan(decay, update, h0)
    expected, expected_last = _naive_affine_scan(decay, update, h0)
    assert torch.allclose(scanned, expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(last, expected_last, atol=1e-5, rtol=1e-5)

    layer = HybridLongContextLayer(d_model=64, num_heads=8, window_size=32).eval()
    x, bank = torch.randn(2, 257, 64), torch.randn(2, 40, 64)
    evidence, evidence_mask = BruteForceTopKRetriever(top_k=4)(x, bank)
    y, state = layer(x, retrieved_memory=evidence, retrieved_mask=evidence_mask, return_state=True)
    assert y.shape == x.shape and state.shape == (2, 64) and torch.isfinite(y).all()

    attention = SlidingWindowAttention(64, 8, 32, 0.0).eval()
    full, _ = attention(x, causal=True, position_ids=None)
    first, cache = attention(x[:, :129], causal=True, position_ids=None)
    second, _ = attention(x[:, 129:], causal=True, position_ids=None, kv_cache=cache)
    assert torch.allclose(full, torch.cat((first, second), dim=1), atol=1e-5, rtol=1e-5)

    copy_head = SourceGroundedCopyHead(64, 128).eval()
    source_states = torch.randn(2, 257, 4, 64)
    source_ids = torch.randint(0, 128, (2, 257, 4), dtype=torch.int32)
    source_mask = torch.ones_like(source_ids, dtype=torch.bool)
    copy_logits = copy_head(x, source_states, source_ids, source_mask)
    assert copy_logits.shape == (2, 257, 128) and torch.isfinite(copy_logits).all()
    print("Smoke test passed:", tuple(y.shape))


if __name__ == "__main__":
    _smoke_test()
