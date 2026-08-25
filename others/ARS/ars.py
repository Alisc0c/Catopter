import numpy as np
import math
import random
import torch
import torch.nn.functional as F


def _model_vocab(model) -> int:
    """Robust vocab size across architectures (Gemma4 stores it in
    ``config.text_config`` and exposes no ``model.vocab_size``)."""
    try:
        return int(model.config.vocab_size)
    except AttributeError:
        pass
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for name in ("text_config", "language_config", "decoder"):
            sub = getattr(cfg, name, None)
            if sub is not None and hasattr(sub, "vocab_size"):
                return int(sub.vocab_size)
    raise AttributeError("cannot determine vocab size")


def limit_past(past):
    """No-op: let the KV cache grow naturally.  Truncating the cache
    (keys/values) without rewinding the RoPE position encoding misaligns the
    rotary offsets and corrupts the logits, which showed up as inflated
    per-token entropy in the tail of long generations."""
    return past


def get_probs_past(model,
                   prev=None,
                   past=None,
                   device='cuda'):
    if past is not None:
        past = limit_past(past)

    model_output = model(prev, past_key_values=past)
    past = model_output.past_key_values

    logits = model_output.logits[0, -1, :].to(device)
    probs = F.softmax(logits, dim=-1)
    indices = torch.arange(len(logits), device=device)

    probs = probs.detach().float().cpu().numpy()
    indices = indices.detach().cpu().numpy()

    return probs, indices, past


def sampling_function(r):
    return math.cos(math.pi * r)


def token_to_bits(token_id, bits_per_token):
    # Accelerate with bitwise operations
    bits = []
    for i in range(bits_per_token):
        bits.append((token_id >> (bits_per_token - 1 - i)) & 1)
    return bits


def bits_to_token(bits):
    # Accelerate with bitwise operations
    token_id = 0
    for bit in bits:
        token_id = (token_id << 1) | bit
    return token_id


@torch.no_grad()
def encode_step(model, history, secret_bit, past, error_bound=0.1, max_n = 512, sampling_fn=sampling_function, device='cpu'):
    # Calculate bits required per token
    bits_per_token = math.ceil(math.log2(_model_vocab(model)))

    # Initialize variables
    token_bits = []  # Binary representation of current token
    generated_tokens = []  # Generated token sequence
    step_nlls = []
    step_entropies = []
    s_values = []  # Accumulated s values
    n = 0  # Sampling count

    # Get initial token distribution
    if past is None:
        prev = torch.tensor([history], device=device)          # prefill
    else:
        prev = torch.tensor([[history[-1]]], device=device)     # incremental
    probs_array, token_ids, past = get_probs_past(model, prev=prev, past=past, device=device)

    # Loop until secret bit is successfully encoded
    while True:
        # Split vocabulary into two groups by current token_bits prefix
        prefix_len = len(token_bits)
        if prefix_len < bits_per_token:
            # Calculate prefix value
            prefix_val = 0
            for bit in token_bits:
                prefix_val = (prefix_val << 1) | bit

            # Find matched tokens using bitwise operations
            shift = bits_per_token - prefix_len
            token_prefixes = token_ids >> shift
            matched = (token_prefixes == prefix_val)

            if np.any(matched):
                matched_indices = np.where(matched)[0]
                # Get next bits for matched tokens
                next_bit_pos = bits_per_token - 1 - prefix_len
                next_bits = (token_ids[matched_indices] >> next_bit_pos) & 1

                group0_prob = np.sum(probs_array[matched_indices[next_bits == 0]])
                group1_prob = np.sum(probs_array[matched_indices[next_bits == 1]])
            else:
                group0_prob = 0.0
                group1_prob = 0.0
        else:
            group0_prob = 0.0
            group1_prob = 0.0

        # Normalize probabilities
        total_prob = group0_prob + group1_prob
        if total_prob == 0:
            # Restart if no matching token
            token_bits = []
            continue

        p0 = group0_prob / total_prob
        p1 = group1_prob / total_prob

        # Get random number from PRG
        r = random.random()

        # Select arrangement according to secret bit
        if secret_bit == 0:
            # Arrangement (0,1): group0 in [0, p0), group1 in [p0, 1)
            threshold = p0
            next_bit = 0 if r < threshold else 1
        else:  # secret_bit == 1
            # Arrangement (1,0): group1 in [0, p1), group0 in [p1, 1)
            threshold = p1
            next_bit = 1 if r < threshold else 0

        # Calculate s value
        if next_bit == 0:
            s_val = sampling_fn(r)
        else:
            s_val = sampling_fn(1 - r)

        # Update state
        token_bits.append(next_bit)
        s_values.append(s_val)
        n += 1

        # Check if a token is completed
        if len(token_bits) == bits_per_token:
            # Convert binary bits to token
            token_id = bits_to_token(token_bits)

            # Check if it is an end token (optional)
            # if token_id == model.config.eos_token_id:
            #     break


            generated_tokens.append(token_id)

            token_prob = probs_array[token_id]
            step_nlls.append(-math.log(token_prob + 1e-12))
            step_entropies.append(-np.sum(probs_array * np.log2(probs_array + 1e-12)))

            # Update history
            history.append(token_id)

            if len(history) > max_n:
                is_embedded = False
                break

            # Reset token_bits for next token
            token_bits = []

            # past already covers history[:-1]; pass only the newest token as
            # prev (feeding full history here duplicates the cached prefix and
            # corrupts the logits, inflating entropy).
            probs_array, token_ids, past = get_probs_past(model, prev=torch.tensor([[history[-1]]], device=device), past=past, device=device)

        # Check if confidence is sufficient
        if len(token_bits) == 0:  # Ensure check after a complete token
            sum_s = sum(s_values)
            confidence = math.exp(-(sum_s ** 2) / (2 * n)) if n > 0 else 1.0

            if confidence <= error_bound and n % bits_per_token == 0:
                is_embedded = True
                break

    return generated_tokens, history, s_values, past, step_nlls, step_entropies, is_embedded


def encode(model, history, secret_bits, error_bound=0.1, sampling_fn=sampling_function,  random_seed=42, max_n=1024, device='cpu'):
    random.seed(random_seed)
    generated_tokens = []
    all_nlls = []
    all_entropies = []
    current_history = history.copy()
    past = None
    cnt = 0
    curr_tokens_len = 0
    max_n += len(current_history)
    import tqdm
    for bit in tqdm.tqdm(secret_bits, desc="Encoded bits"):
        new_tokens, current_history, _, past, nlls, entropies, is_embedded = encode_step(model, current_history, bit, past, error_bound, max_n, sampling_fn, device)
        generated_tokens.extend(new_tokens)
        all_nlls.extend(nlls)
        all_entropies.extend(entropies)
        if is_embedded:
            cnt += 1
            curr_tokens_len = len(generated_tokens)
        else:
            generated_tokens = generated_tokens[:curr_tokens_len]
            print(f"Stopped encoding at bit index {cnt} due to max token limit.")
            break 
    remaining_bits = secret_bits[cnt:]
    remaining_len = len(remaining_bits)
    return generated_tokens, remaining_len, all_nlls, all_entropies


def decode_step(token, s_values, n, bits_per_token, error_bound=0.1, sampling_fn=sampling_function):
    # Calculate E[s|H_∅], equals 0 when f(r) = cos(πr)
    expected_s_H0 = 0.0

    bits = token_to_bits(token, bits_per_token)

    # Process bit sequence in a loop
    for bit in bits:
        # Get synchronized random number from PRG
        r = random.random()

        # Calculate s value
        if bit == 0:
            s_val = sampling_fn(r)
        else:
            s_val = sampling_fn(1 - r)

        # Update state
        s_values.append(s_val)
        n += 1

        # Check confidence
        sum_s = sum(s_values)
        confidence = math.exp(-(sum_s ** 2) / (2 * n)) if n > 0 else 1.0

    if confidence <= error_bound:
        if sum_s > expected_s_H0:
            return 0, [], 0
        else:
            return 1, [], 0
    else:
        return None, s_values, n

def decode(token_sequence, vocab_size, error_bound=0.1, sampling_fn=sampling_function, random_seed=42):
    random.seed(random_seed)
    bits_per_token = math.ceil(math.log2(vocab_size))
    secret_bits = []
    s_values = []
    n = 0
    for token in token_sequence:
        secret_bit, s_values, n = decode_step(token, s_values, n, bits_per_token, error_bound, sampling_fn)
        if secret_bit is not None:
            secret_bits.append(secret_bit)
            s_values = []
            n = 0

    return secret_bits