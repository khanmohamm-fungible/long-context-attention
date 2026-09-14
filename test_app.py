import torch
import torch.nn.functional as F

from App import (
    HybridLongContextLayer,
    RetrievedEvidenceAttention,
    SlidingWindowAttention,
    SourceGroundedCopyHead,
)


def reference_attention(
    attention: SlidingWindowAttention,
    x: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    batch, tokens, width = x.shape
    positions = torch.arange(tokens, device=x.device).expand(batch, -1)
    qkv = attention.qkv(x).view(
        batch,
        tokens,
        3,
        attention.num_heads,
        attention.head_dim,
    )
    q, k, v = qkv.unbind(dim=2)
    q = attention._rope(q, positions)
    k = attention._rope(k, positions)
    left = attention.window_size - 1 if causal else attention.window_size // 2
    right = 0 if causal else attention.window_size - left - 1
    outputs = []

    for token in range(tokens):
        start = max(0, token - left)
        end = min(tokens, token + right + 1)
        output = F.scaled_dot_product_attention(
            q[:, token : token + 1].transpose(1, 2),
            k[:, start:end].transpose(1, 2),
            v[:, start:end].transpose(1, 2),
            dropout_p=0.0,
        )
        outputs.append(output.transpose(1, 2))

    attended = torch.cat(outputs, dim=1)
    return attention.out(attended.reshape(batch, tokens, width))


def main() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 37, 32)
    attention = SlidingWindowAttention(32, 4, 8, 0.0).eval()

    for causal in (False, True):
        actual, cache = attention(x, causal=causal, position_ids=None)
        expected = reference_attention(attention, x, causal=causal)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        if cache is not None:
            assert cache.keys.untyped_storage().nbytes() == (
                cache.keys.numel() * cache.keys.element_size()
            )
            assert cache.values.untyped_storage().nbytes() == (
                cache.values.numel() * cache.values.element_size()
            )

    token_mask = torch.ones(2, 37, dtype=torch.bool)
    token_mask[1, 23:] = False
    for causal in (False, True):
        padded, cache = attention(
            x,
            causal=causal,
            position_ids=None,
            token_mask=token_mask,
        )
        unpadded, _ = attention(
            x[1:2, :23],
            causal=causal,
            position_ids=None,
        )
        torch.testing.assert_close(
            padded[1:2, :23],
            unpadded,
            atol=1e-6,
            rtol=1e-5,
        )
        assert torch.count_nonzero(padded[1, 23:]) == 0
        if cache is not None:
            assert cache.next_position.tolist() == [37, 23]
            assert cache.valid[1].all()
            next_tokens = torch.randn(2, 1, 32)
            streamed, _ = attention(
                next_tokens,
                causal=True,
                position_ids=None,
                kv_cache=cache,
            )
            for row, length in enumerate((37, 23)):
                combined = torch.cat(
                    (x[row : row + 1, :length], next_tokens[row : row + 1]),
                    dim=1,
                )
                full, _ = attention(
                    combined,
                    causal=True,
                    position_ids=None,
                )
                torch.testing.assert_close(
                    streamed[row : row + 1],
                    full[:, -1:],
                    atol=1e-6,
                    rtol=1e-5,
                )

    hybrid = HybridLongContextLayer(32, window_size=8, num_heads=4).eval()
    padded_output, padded_state = hybrid(
        x,
        causal=True,
        token_mask=token_mask,
        return_state=True,
    )
    unpadded_output, unpadded_state = hybrid(
        x[1:2, :23],
        causal=True,
        return_state=True,
    )
    torch.testing.assert_close(
        padded_output[1:2, :23],
        unpadded_output,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        padded_state[1:2],
        unpadded_state,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.count_nonzero(padded_output[1, 23:]) == 0

    training_attention = SlidingWindowAttention(32, 4, 8, 0.1).train()
    training_input = x.clone().requires_grad_()
    training_output, _ = training_attention(
        training_input,
        causal=True,
        position_ids=None,
    )
    training_output.square().mean().backward()
    assert training_input.grad is not None
    assert torch.isfinite(training_input.grad).all()

    copy_head = SourceGroundedCopyHead(32, 101).eval()
    source_states = torch.randn(2, 37, 5, 32)
    source_ids = torch.randint(0, 101, (2, 37, 5), dtype=torch.int32)
    source_mask = torch.ones_like(source_ids, dtype=torch.bool)
    logits = copy_head(x, source_states, source_ids, source_mask)
    assert logits.shape == (2, 37, 101)
    assert torch.isfinite(logits).all()
    selected_logits = copy_head(
        x,
        source_states,
        source_ids,
        source_mask,
        output_positions=torch.tensor([-1]),
    )
    torch.testing.assert_close(selected_logits, logits[:, -1:])

    retrieval = RetrievedEvidenceAttention(32, query_chunk_size=7).eval()
    shared_memory = torch.randn(2, 5, 32)
    shared_mask = torch.tensor(
        [[True, True, False, True, False], [True, False, True, True, True]]
    )
    shared_output = retrieval(x, shared_memory, shared_mask)
    expanded_output = retrieval(
        x,
        shared_memory[:, None].expand(-1, x.shape[1], -1, -1),
        shared_mask[:, None].expand(-1, x.shape[1], -1),
    )
    torch.testing.assert_close(
        shared_output,
        expanded_output,
        atol=1e-6,
        rtol=1e-5,
    )
    print("Reference tests passed")


if __name__ == "__main__":
    main()
