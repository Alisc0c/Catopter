import numpy as np
import torch
import torch.nn.functional as F
import hashlib
import hmac
from collections import deque


# ---------------------------------------------------------------------------
#  KV cache / past helpers
# ---------------------------------------------------------------------------

def limit_past(past):
    """Truncate past KV cache to the last 1022 positions to bound memory."""
    if past is None:
        return None
    for layer in past.layers:
        if hasattr(layer, 'is_initialized'):
            if layer.is_initialized:
                layer.keys = layer.keys[:, :, -1022:]
                layer.values = layer.values[:, :, -1022:]
    return past


from catopter.prf import (
    _prf_hash, token_hash, check_boundary, token_hash_batch, close_pool,
)


# ---------------------------------------------------------------------------
#  Model inference (incremental KV-cache aware)
# ---------------------------------------------------------------------------

def get_probs_past(model,
                   prev=None,
                   past=None,
                   top_p=None,
                   top_k=None,
                   device='cuda'):
    """One forward pass returning (probs, indices, past)."""
    if past is not None:
        past = limit_past(past)

    model_output = model(prev, past_key_values=past)
    past = model_output.past_key_values

    logits = model_output.logits[0, -1, :].to(device)

    # top-p (nucleus) filtering
    if 0 < top_p < 1.0:
        logits, indices = logits.sort(descending=True)
        indices = indices.int()
        probs = F.softmax(logits, dim=-1)
        cum_probs = probs.cumsum(0)
        nonzero = (cum_probs > top_p).nonzero()[0]
        if len(nonzero) == 0:
            k = len(probs)                                 # rare float edge case
        else:
            k = nonzero[0].item() + 1
        probs = probs[:k]
        indices = indices[:k]
        probs = 1.0 / cum_probs[k - 1] * probs           # renormalise
    elif top_p == 1.0:
        probs = F.softmax(logits, dim=-1)
        indices = None                    # identity map — skip GPU arange + 1MB transfer
    else:
        raise ValueError("top_p should be in (0, 1]")

    probs = probs.detach().float().cpu().numpy()
    if indices is None:
        indices = np.arange(probs.shape[0])
    else:
        indices = indices.detach().cpu().numpy()

    if top_k is not None and top_k > 0:
        # legacy top-k path (renormalising truncation), kept for compatibility
        if len(probs) > top_k:
            top_indices = np.argsort(probs)[-top_k:]
            probs = probs[top_indices]
            indices = indices[top_indices]
            probs /= probs.sum()                          # renormalise
    else:
        keep = probs > 1e-9
        if np.sum(keep) < len(probs):
            probs = probs[keep]
            indices = indices[keep]

    return probs, indices, past


# ---------------------------------------------------------------------------
#  Bit ↔ integer conversion  (MSB-first, numpy vectorised)
# ---------------------------------------------------------------------------

def int_to_bits(x, m: int) -> np.ndarray:
    """Convert integers to MSB-first bit vectors."""
    x = np.asarray(x, dtype=np.int64)
    shifts = np.arange(m - 1, -1, -1, dtype=np.int64)
    return ((x[..., None] >> shifts) & 1).astype(np.uint8)


def bits_to_int(bits) -> np.ndarray:
    """Convert MSB-first bit vectors to integers (inverse of int_to_bits)."""
    bits = np.asarray(bits, dtype=np.int64)
    m = bits.shape[-1]
    powers = 1 << np.arange(m - 1, -1, -1, dtype=np.int64)
    return (bits * powers).sum(axis=-1)


# ---------------------------------------------------------------------------
#  Dyadic mirror transform  (vectorised over u and/or candidate bits)
# ---------------------------------------------------------------------------

def dyadic_mirror(a_bits, u):

    a_bits = np.asarray(a_bits, dtype=np.uint8)
    u = np.asarray(u, dtype=np.float64)
    m = a_bits.shape[-1]

    out_shape = np.broadcast_shapes(a_bits.shape[:-1], u.shape)
    x = np.broadcast_to(u, out_shape)

    if m == 1:
        # Binary case: identity vs mirror (reflection family has no identity).
        return np.mod(np.where(a_bits[..., 0], 1.0 - x, x), 1.0)

    # m >= 2: global reflection about ψ_M = M / 2^(m+1),  i.e. (M / 2^m − u) mod 1
    M = bits_to_int(a_bits).astype(np.float64)
    return np.mod(M / float(1 << m) - x, 1.0)


# def dyadic_mirror_recursive(a_bits, u):
#     """Apply recursive per-bit dyadic mirror transform."""
#     a_bits = np.asarray(a_bits, dtype=np.uint8)
#     u = np.asarray(u, dtype=np.float64)
#     m = a_bits.shape[-1]
#
#     out_shape = np.broadcast_shapes(a_bits.shape[:-1], u.shape)
#     x = np.broadcast_to(u, out_shape).copy()
#
#     for i in range(m):
#         bit = a_bits[..., i]          # broadcasts with x via np.where
#         L = 2.0 ** (-i)
#         k = (x / L).astype(np.int64)
#         x_new = (2 * k + 1) * L - x
#         x = np.where(bit, x_new, x)
#
#         # Safety wrap into [0, 1) — handles multi-step overflow
#         x = np.mod(x, 1.0)
#
#     return x


# ---------------------------------------------------------------------------
#  Score function & stop condition  (vectorised)
# ---------------------------------------------------------------------------

def score_func(x):
    """Steganography score:  −log(1 − x),  x ∈ [0, 1).  Vectorised."""
    eps = 1e-12
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0 - eps)
    return -np.log(1.0 - x)


def check_stop_condition(candidate_scores, T_bound):
    """Return True when the top candidate leads the runner-up by > T_bound."""
    scores = np.asarray(candidate_scores, dtype=np.float64)
    if scores.size < 2:
        return True
    sorted_scores = np.sort(scores)
    top1 = sorted_scores[-1]
    top2 = sorted_scores[-2]
    return (top1 - top2) > T_bound


# ---------------------------------------------------------------------------
#  Gumbel-max sampling
# ---------------------------------------------------------------------------

def gumbel_max_sample(probs, u_values: np.ndarray):
    """Gumbel-max trick: argmax_i u_i^{1/p_i}, returning (winner_local, p_selected)."""
    probs = np.asarray(probs, dtype=np.float64)
    u_values = np.asarray(u_values, dtype=np.float64)

    eps = 1e-12
    valid = probs > 0.0
    scores = np.full(probs.shape, -np.inf, dtype=np.float64)

    log_u = np.log(np.clip(u_values, eps, 1.0))
    scores[valid] = log_u[valid] / np.clip(probs[valid], eps, None)

    winner_local = int(np.argmax(scores))
    p_selected = float(probs[winner_local])

    return winner_local, p_selected


# ---------------------------------------------------------------------------
#  Single-chunk encoder
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_step(model, history, secret_bits, past,
                T_bound=10, top_p=1.0, top_k=100, max_n=1024,
                sk: str = "42", prefix_len: int = 1,
                prev_generated: list = None, device='cpu',
                use_hash_boundary: bool = True,
                hash_w: int = 3, hash_k: int = 1,
                hash_e: int = 1, fixed_len: int = None):
    """Embed one chunk of secret_bits into generated tokens."""
    assert prefix_len >= 1, "prefix_len must be >= 1 (Python history[-0:] == history[:])"

    if prev_generated is None:
        prev_generated = []

    secret_bits = np.asarray(secret_bits, dtype=np.uint8)      # (m,)
    bits_num = len(secret_bits)
    q = 1 << bits_num

    # Pre-compute all candidate bit patterns: shape (q, bits_num)  MSB-first
    all_candidate_bits = int_to_bits(np.arange(q, dtype=np.int64), bits_num)

    candidate_scores = np.zeros(q, dtype=np.float64)

    generated_tokens = []
    step_nlls = []
    step_entropies = []

    # O(1) sliding-window prefix buffer (avoids O(n²) list copy per token)
    prefix_buf = deque(prev_generated[-prefix_len:] if (prev_generated and prefix_len > 0) else [],
                       maxlen=prefix_len)

    is_embedded = False
    if past is None:
        probs_array, token_indices, past = get_probs_past(
            model,
            prev=torch.tensor([history], device=device),
            past=None,
            top_p=top_p,
            top_k=top_k,
            device=device,
        )
    else:
        probs_array, token_indices, past = get_probs_past(
            model,
            prev=torch.tensor([[history[-1]]], device=device),
            past=past,
            top_p=top_p,
            top_k=top_k,
            device=device,
        )

    while True:
        # PRF context: last prefix_len generated tokens only (decoder has no prompt/history).
        prefix = tuple(prefix_buf)

        # PRF u-values for every candidate token (multi-core batch,
        # bitwise identical to per-candidate token_hash calls)
        u_values = token_hash_batch(sk, token_indices, prefix)

        # Apply dyadic mirror according to secret bits — vectorised over vocab
        new_u_values = dyadic_mirror(secret_bits, u_values)

        # Gumbel-max sample
        winner_local, p_selected = gumbel_max_sample(probs_array, new_u_values)

        token_idx = int(token_indices[winner_local])
        selected_u = float(u_values[winner_local])

        generated_tokens.append(token_idx)
        history.append(token_idx)
        prefix_buf.append(token_idx)

        entropy = -np.sum(probs_array * np.log2(probs_array + 1e-12))
        step_entropies.append(float(entropy))

        nll = -np.log(p_selected + 1e-12)
        step_nlls.append(float(nll))

        # Accumulate scores for all candidates (vectorised over q patterns).
        candidate_scores += score_func(
            dyadic_mirror(all_candidate_bits, selected_u)
        )

        # ── stopping condition ──
        if fixed_len is not None:
            # Fixed-rate framing: emit exactly fixed_len tokens (position-anchored).
            if len(generated_tokens) >= fixed_len:
                is_embedded = True
                break
        else:
            score_ok = check_stop_condition(candidate_scores, T_bound)

            if score_ok:
                if not use_hash_boundary:
                    is_embedded = True
                    break

                # Require both score gap and hash marker (soft per-token vote).
                hash_ok = (
                    len(generated_tokens) >= hash_w
                    and check_boundary(sk, generated_tokens, hash_w, hash_e, hash_k)
                )

                if hash_ok:
                    is_embedded = True
                    break
                # else: score converged but no hash marker — continue accumulating

        if len(history) >= max_n:
            is_embedded = False
            break

        # ── incremental forward pass: only the newly generated token ──
        probs_array, token_indices, past = get_probs_past(
            model,
            prev=torch.tensor([[token_idx]], device=device),
            past=past,
            top_p=top_p,
            top_k=top_k,
            device=device,
        )

    return (generated_tokens, history, candidate_scores,
            past, step_nlls, step_entropies, is_embedded)


# ---------------------------------------------------------------------------
#  Multi-chunk encoder
# ---------------------------------------------------------------------------

def encode(model, history, secret_bits,
           chunk_size=2, T_bound=10,
           top_p=1.0, top_k=100,
           sk: str = "42", max_n=1024, prefix_len: int = 1, device='cpu',
           use_hash_boundary: bool = True,
           hash_w: int = 3, hash_k: int = 1,
           hash_e: int = 1, fixed_len: int = None):
    """Embed secret_bits in chunks of chunk_size bits."""
    generated_tokens = []
    all_nlls = []
    all_entropies = []
    current_history = history.copy()
    past = None
    cnt = 0
    max_n += len(current_history)

    import tqdm
    pbar = tqdm.tqdm(total=len(secret_bits), desc="Encoded bits", unit="bit")
    for i in range(0, len(secret_bits), chunk_size):
        bit = secret_bits[i:i + chunk_size]

        # Snapshot lengths before this chunk (for rollback on failure)
        prev_token_len = len(generated_tokens)
        prev_nll_len = len(all_nlls)
        prev_entropy_len = len(all_entropies)

        new_tokens, current_history, _, past, nlls, entropies, is_embedded = \
            encode_step(model, current_history, bit, past,
                        T_bound=T_bound, top_p=top_p, top_k=top_k,
                        sk=sk, max_n=max_n, prefix_len=prefix_len,
                        prev_generated=generated_tokens.copy(), device=device,
                        use_hash_boundary=use_hash_boundary,
                        hash_w=hash_w, hash_k=hash_k,
                        hash_e=hash_e, fixed_len=fixed_len)

        generated_tokens.extend(new_tokens)
        all_nlls.extend(nlls)
        all_entropies.extend(entropies)

        if is_embedded:
            cnt += len(bit)
            pbar.update(len(bit))
        else:
            # Roll back all three lists to pre-chunk state
            generated_tokens = generated_tokens[:prev_token_len]
            all_nlls = all_nlls[:prev_nll_len]
            all_entropies = all_entropies[:prev_entropy_len]
            print(f"Stopped encoding at bit index {cnt} due to max token limit.")
            break

    pbar.close()
    remaining_bits = secret_bits[cnt:]
    remaining_len = len(remaining_bits)
    return generated_tokens, remaining_len, all_nlls, all_entropies


# ---------------------------------------------------------------------------
#  Single-chunk decoder  (no LLM, no history — only generated tokens)
# ---------------------------------------------------------------------------

def decode_step(tokens: list,
                bits_num: int,
                T_bound: float,
                sk: str = "42",
                prefix_len: int = 1,
                prev_generated: list = None,
                force_output: bool = False,
                use_hash_boundary: bool = True,
                allow_hash_fallback: bool = None,
                hash_w: int = 3, hash_k: int = 1,
                hash_e: int = 1, fixed_len: int = None):
    """Decode one chunk from a slice of generated token ids."""
    if prev_generated is None:
        prev_generated = []

    # allow_hash_fallback defaults to use_hash_boundary (one self-sync knob).
    if allow_hash_fallback is None:
        allow_hash_fallback = use_hash_boundary

    q = 1 << bits_num
    all_candidate_bits = int_to_bits(np.arange(q, dtype=np.int64), bits_num)
    scores = np.zeros(q, dtype=np.float64)

    consumed = 0
    embedded = False
    decoded_bits = None

    # O(1) sliding-window prefix buffer
    prefix_buf = deque(prev_generated[-prefix_len:] if (prev_generated and prefix_len > 0) else [],
                       maxlen=prefix_len)

    hash_candidate = None  # (consumed, scores)  — only used when allow_hash_fallback

    for offset, token_idx in enumerate(tokens):
        token_idx = int(token_idx)

        prefix = tuple(prefix_buf)
        prefix_buf.append(token_idx)

        selected_u = token_hash(sk, token_idx, prefix)

        scores += score_func(
            dyadic_mirror(all_candidate_bits, float(selected_u))
        )
        consumed += 1

        if fixed_len is not None:
            # Fixed-rate framing: read exactly *fixed_len* tokens (position-anchored)
            if consumed >= fixed_len:
                embedded = True
                break
        else:
            score_ok = check_stop_condition(scores, T_bound)
            hash_ok = (
                use_hash_boundary and consumed >= hash_w
                and check_boundary(
                    sk, tokens[offset - hash_w + 1 : offset + 1],
                    hash_w, hash_e, hash_k,
                )
            )

            # Symmetric with encoder: require score convergence (and hash marker if enabled).
            if score_ok:
                if not use_hash_boundary:
                    embedded = True
                    break

                if hash_ok:
                    embedded = True
                    break
                # else: score converged but no hash marker — keep going

            # Track first hash marker as optional fallback (asymmetric recovery)
            if allow_hash_fallback and hash_ok and hash_candidate is None:
                hash_candidate = (consumed, scores.copy())

    # ── fallback: hash candidate when score never converged ──
    if not embedded:
        if allow_hash_fallback and hash_candidate is not None:
            (consumed_copy, scores_copy) = hash_candidate
            consumed = consumed_copy
            scores = scores_copy[:] if hasattr(scores_copy, 'copy') else scores_copy
        elif not force_output:
            return None, consumed, False, scores

    if consumed == 0:
        decoded_bits = np.zeros(bits_num, dtype=np.uint8)
    else:
        decoded_int = int(np.argmax(scores))
        decoded_bits = int_to_bits(decoded_int, bits_num)

    return decoded_bits, consumed, embedded, scores


# ---------------------------------------------------------------------------
#  Multi-chunk decoder  (no LLM, no history — only generated tokens)
# ---------------------------------------------------------------------------

def decode(tokens: list,
           num_bits: int,
           chunk_size: int = 2,
           T_bound: float = 10.0,
           sk: str = "42",
           prefix_len: int = 1,
           force_output: bool = False,
           use_hash_boundary: bool = True,
           allow_hash_fallback: bool = None,
           hash_w: int = 3, hash_k: int = 1,
           hash_e: int = 1, fixed_len: int = None):
    """Decode multiple chunks from a generated token sequence."""
    tokens = list(tokens)

    if allow_hash_fallback is None:
        allow_hash_fallback = use_hash_boundary

    decoded_bits = []
    embedded_flags = []
    chunk_scores = []

    consumed_total = 0

    for start in range(0, num_bits, chunk_size):
        if consumed_total >= len(tokens):
            break

        bits_num = min(chunk_size, num_bits - start)
        prev_generated = tokens[:consumed_total]

        decoded_chunk, consumed, embedded, scores = decode_step(
            tokens=tokens[consumed_total:],
            bits_num=bits_num,
            T_bound=T_bound,
            sk=sk,
            prefix_len=prefix_len,
            prev_generated=prev_generated,
            force_output=force_output,
            use_hash_boundary=use_hash_boundary,
            allow_hash_fallback=allow_hash_fallback,
            hash_w=hash_w, hash_k=hash_k,
            hash_e=hash_e, fixed_len=fixed_len,
        )

        consumed_total += consumed
        embedded_flags.append(embedded)
        chunk_scores.append(scores)

        if decoded_chunk is None:
            break

        decoded_bits.extend(decoded_chunk.tolist())

        if not force_output and not embedded:
            break

    remaining_bits = num_bits - len(decoded_bits)

    return decoded_bits, remaining_bits, consumed_total, embedded_flags, chunk_scores
