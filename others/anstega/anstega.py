"""ANStega (Asymmetric Numeral Systems steganography) — core module.

Refactored from a single bundled script into a clean module:

* Single import block; no wrapper-concatenation leftovers.
* Dead ``to_cache`` call (a ``transformers.to_cache`` that does not exist)
  removed — ``past_key_values`` is passed straight to the model.
* ``encode``/``decode`` take plain parameters and return plain values,
  mirroring ``catopter/catopter.py`` and ``others/ARS/ars.py``; no pandas.

The ANS arithmetic-coding mathematics is unchanged.  ``encode`` embeds a
message bit-string while sampling tokens; ``decode`` recovers it from the
generated token ids plus the final ANS state.
"""

import math
import random
import time
from math import ceil, log2

import torch
import torch.nn.functional as F


def generate_random_mask_bitstring(length, rng=None):
    if rng is None:
        rng = random
    return ''.join(rng.choice('01') for _ in range(length))


def xor_low_bits(x, mask):
    if isinstance(x, str):
        x = int(x, 2)
    if isinstance(mask, str):
        mask = int(mask, 2)
    mask_len = mask.bit_length()
    low_mask = (1 << mask_len) - 1
    x_high = x & (~low_mask)
    x_low = x & low_mask
    result = x_high | (x_low ^ mask)
    return result


def get_probs_past_float64(model, prev=None, past=None, device='cuda', top_p=1.0):
    """One forward pass returning top-k (descending) (probs, indices, past).

    Sorted and cast to float64 for ANS arithmetic coding.  When *past* is
    ``None``, *prev* should be the full context; otherwise *prev* is the
    single most-recent token and *past* the incremental KV cache.
    """
    if prev is not None:
        prev = prev.to(device)
    model_output = model(prev, past_key_values=past)
    past = model_output.past_key_values
    logits = model_output.logits[0, -1, :].to(device)
    logits, indices = logits.sort(descending=True)
    logits = logits.double()
    indices = indices.int()
    probs = F.softmax(logits, dim=-1)
    # full-vocab entropy (bits) computed BEFORE top-p truncation, so the
    # entropy is comparable across methods (tail tokens contribute ~0 anyway).
    full_entropy = float(-torch.sum(probs * torch.log2(probs + 1e-10)))
    if 0 < top_p < 1.0:
        cum_probs = probs.cumsum(0)
        k = (cum_probs > top_p).nonzero()[0].item() + 1
        probs = probs[:k]
        indices = indices[:k]
        probs = 1 / cum_probs[k - 1] * probs
    return probs, indices, past, full_entropy


def _anstega_encode_step(x, remaining_message, probs, indices, cumulative_probs,
                         precision, M, L, H):
    block_id = x >> precision
    slot_int = x & (M - 1)
    slot_fraction = slot_int / M
    # Select the token whose interval contains slot_fraction.  Interval i is
    # (cum[i], cum[i+1]] — with cum = [0, ...].  right=False returns the first
    # j with cum[j] > slot_fraction, i.e. j = i+1 → index = i.  right=True would
    # push a slot that lands exactly on a boundary to the *next* token
    # (off-by-one that flips the recovered bits).
    index = torch.searchsorted(cumulative_probs, slot_fraction, right=False).item() - 1
    index = max(index, 0)
    s = indices[index]
    x_shrunk = block_id * ceil(probs[index] * M) + slot_int - ceil(cumulative_probs[index] * M)

    x_prev = x_shrunk
    bits_read_count = 0
    message_len = len(remaining_message)
    while not (L <= x_prev <= H) and bits_read_count < message_len:
        x_prev = (x_prev << 1) | int(remaining_message[bits_read_count])
        bits_read_count += 1

    consumed_bits = remaining_message[:bits_read_count]
    new_remaining_message = remaining_message[bits_read_count:]
    return s.item(), x_prev, consumed_bits, new_remaining_message


@torch.no_grad()
def encode(model, context, message: str, token_num_need_generated: int,
           device='cuda', precision=32, ans_t=16, top_p=1.0, seed=42):
    """Embed *message* (a ``'0'``/``'1'`` string) into generated tokens.

    Parameters
    ----------
    context : list[int] or Tensor
        Token ids of the prompt/context (shape ``(1, seq_len)`` or a flat
        list, as accepted by the model forward pass).
    token_num_need_generated : int
        Number of stego tokens to generate.
    precision : int
        ANS state precision in bits (default 32).
    ans_t : int
        ANS scaling factor (default 16).

    Returns
    -------
    (generated_ids, final_state, encoded_message_chunks, total_entropy,
     stat_time, model_time, nll_list, initial_pulls)
        ``generated_ids`` : list[int] — the generated token ids.
        ``final_state``   : int — the final ANS state (needed by ``decode``).
        ``encoded_message_chunks`` : list[str] — per-token message chunks.
        ``total_entropy`` : float — sum of per-step entropy (bit).
        ``stat_time``     : float — seconds spent on stats (entropy).
        ``model_time``    : float — seconds spent in the model forward pass.
        ``nll_list``      : list[float] — per-token negative log-likelihood.
        ``initial_pulls`` : int — message bits pulled to lift the initial
        state into ``[L, 2L)``; ``decode`` needs it to size the final chunk.
    """
    generated_ids = []
    encoded_message_chunks = []
    token_num_generated = 0
    M = 2 ** precision
    L = M * ans_t
    H = 2 * L - 1
    rng = random.Random(seed)

    num_states_bits = ceil(log2(H))
    initial_bits = message[:num_states_bits]
    x = int(initial_bits, 2)
    message = message[num_states_bits:]
    encoded_message_chunks.append(initial_bits)

    # Pull message bits until the state enters [L, 2L).  This restores the
    # rANS invariant q = x // M ∈ [t, 2t) for the very first token; without
    # it the first token's renormalization can't be inverted by the decoder.
    initial_pulls = 0
    while x < L and message:
        x = (x << 1) | int(message[0])
        message = message[1:]
        initial_pulls += 1

    total_entropy = 0.0
    stat_time = 0.0
    model_time = 0.0
    prev = context
    past = None
    nll_list = []

    while token_num_generated < token_num_need_generated:
        mask = generate_random_mask_bitstring(precision, rng)
        x = xor_low_bits(x, mask)

        model_time_1 = time.time()
        probs, indices, past, cur_entropy = get_probs_past_float64(model, prev=prev, past=past, device=device, top_p=top_p)
        model_time_2 = time.time()
        model_time += model_time_2 - model_time_1

        t1 = time.time()
        total_entropy += cur_entropy
        t2 = time.time()
        stat_time += t2 - t1

        probs_with_zero = torch.cat((torch.tensor([0.0], device=device), probs), dim=0)
        cumulative_probs = torch.cumsum(probs_with_zero, dim=-1)

        s, x, consumed_bits, message = _anstega_encode_step(
            x, message, probs, indices, cumulative_probs, precision, M, L, H)
        encoded_message_chunks.append(consumed_bits)

        # NLL
        idx = (indices == s).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            nll_list.append(-math.log(max(probs[idx[0].item()], 1e-12)))
        else:
            nll_list.append(0.0)

        generated_ids.append(s)
        prev = torch.tensor([s], device=device).unsqueeze(0)
        token_num_generated += 1
    return (generated_ids, x, encoded_message_chunks, total_entropy,
            stat_time, model_time, nll_list, initial_pulls)


@torch.no_grad()
def decode(model, context, generated_ids, final_state, device='cuda',
           precision=32, ans_t=16, top_p=1.0, seed=42, initial_pulls=0):
    """Recover the message bit-string from *generated_ids* and *final_state*.

    *context*, *precision*, *ans_t*, *top_p* and *seed* must match ``encode``.
    *initial_pulls* is the value returned by ``encode`` (message bits pulled
    to lift the initial state into ``[L, 2L)``); it sizes the final chunk so
    leading-zero initial states are recovered exactly.

    Returns
    -------
    str — the full reconstructed message (including the initial state bits).
    """
    prev = context
    past = None
    out_bits_list = []
    M = 2 ** precision
    L = M * ans_t
    H = 2 * L - 1
    x = final_state
    rng = random.Random(seed)

    # Forward: collect masks and probs
    mask_list = []
    for _ in range(len(generated_ids)):
        mask = generate_random_mask_bitstring(precision, rng)
        mask_list.append(mask)
    mask_list.reverse()

    probs_s_list = []
    cs_list = []
    for tokenID in generated_ids:
        probs, indices, past, _ = get_probs_past_float64(model, prev=prev, past=past, device=device, top_p=top_p)
        probs_with_zero = torch.cat((torch.tensor([0.0], device=device), probs), dim=0)
        cumulative_probs = torch.cumsum(probs_with_zero, dim=-1)
        idx = torch.where(indices == tokenID)[0]
        probs_s_list.append(probs[idx].item())
        cs_list.append(cumulative_probs[idx].item())
        prev = torch.tensor([tokenID], device=device).unsqueeze(0)

    reverse_ids = list(reversed(generated_ids))
    probs_s_list.reverse()
    cs_list.reverse()

    for i, tokenID in enumerate(reverse_ids):
        out_bits = ""
        # Pseudocode Algorithm 2: xmax = 2·t·⌈P_s·M⌉.  Note this uses the
        # *unrounded* ⌈P_s·M⌉ (not ceil(t·P_s·M)), matching the encoder's
        # Expand invariant; the original `2*ceil(t*P_s*M)` mis-sizes the
        # pop count and loses one bit per diverging token.
        f_s = ceil(probs_s_list[i] * M)
        xmax = 2 * ans_t * f_s
        while x >= xmax:
            out_bits = str(x % 2) + out_bits
            x = x // 2
        block_id = x // f_s
        slot = ceil(cs_list[i] * M) + x % f_s
        x = block_id * M + slot
        x = xor_low_bits(x, mask_list[i])
        out_bits_list.append(out_bits)

    num_state_bits = ceil(log2(H))
    binary_string = bin(x)[2:]
    # Final chunk: pad the recovered state to `num_state_bits + initial_pulls`
    # bits so leading-zero initial chunks (which needed extra Expand pulls)
    # are recovered exactly.
    final_part_x = binary_string.zfill(num_state_bits + initial_pulls)
    out_bits_list.append(final_part_x)

    # Reconstruct message: chunks are in reverse order
    out_bits_list.reverse()
    return "".join(out_bits_list)
