from typing import Tuple, List, Dict, Any, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from dataclasses import dataclass, asdict
import os
import sys
import numpy as np
import random
import json

from .catopter import encode, decode
from .edits import apply_attack


@dataclass
class CatopterConfig:
    secret_bits: str
    message_length: int
    context: str
    model_path: str
    random_seed: int
    chunk_size: int = 1
    T_bound: float = 10.0
    prefix_len: int = 1
    sk: str = "42"
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_n: int = 1024
    output_dir: str = "./results/Catopter"
    attack: str = "none"
    attack_rate: float = 0.1
    attack_times: int = 5
    device: str = "cuda"  # or "cpu"
    # ── soft hash boundary marker (default) ──
    use_hash_boundary: bool = True  # whether to use hash boundary for chunking
    hash_w: int = 3    # trailing token window for the boundary vote
    hash_e: int = 1    # tolerated corrupted tokens inside the window (>= 0)
    hash_k: int = 1    # per-token predicate bits (each votes yes w.p. 2^-hash_k)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _to_bits(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if hasattr(x, 'tolist'):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return ''.join(str(b) for b in x)
    return str(x)


def compare_bits(original, reconstructed):
    """Return (correct, wrong, lost) bit counts over reference length N."""
    orig = _to_bits(original)
    recon = _to_bits(reconstructed)
    if not orig:
        return 0, 0, 0
    correct = sum(1 for a, b in zip(orig, recon) if a == b)
    compared = min(len(orig), len(recon))
    wrong = compared - correct
    lost = len(orig) - compared
    return correct, wrong, lost


def compute_correct_rate(original, reconstructed) -> float:
    """Per-bit correct rate = correct / len(original)."""
    correct, _, _ = compare_bits(original, reconstructed)
    orig_len = len(_to_bits(original))
    return correct / orig_len if orig_len else 0.0


def process_message(message: str) -> str:
    """Process message to ensure it's binary"""
    # Filter only 0s and 1s
    message = ''.join(c for c in message if c in '01')
    return message


def load_model(model_path, device):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype='auto').to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        import warnings
        warnings.warn("Failed to load model locally, load gpt2 directly from Transformers library")
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    model.eval()
    return model, tokenizer


def process_context(tokenizer, context: str) -> np.ndarray:
    """Tokenize context into token IDs for Catopter's history parameter."""
    try:
        prompt_template = [
            {"role": "user", "content": context}
        ]
        # Force enable_thinking=False to suppress Qwen thinking mode
        chat_template_kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        context = tokenizer.apply_chat_template(
            prompt_template,
            **chat_template_kwargs
        )
        context_ids = tokenizer(context, return_tensors='pt').input_ids.squeeze().numpy()

    except Exception as e:
        import warnings
        warnings.warn(f"Chat template failed ({e}), falling back to direct encoding.")
        context_ids = tokenizer.encode(context, return_tensors='pt', add_special_tokens=False).squeeze().numpy()

    return context_ids


def test_catopter(config: CatopterConfig, device: torch.device) -> Dict[str, Any]:
    """Run the Catopter encode/decode pipeline and return results."""
    print("\n=== Initialization Phase ===")

    # Load model and tokenizer
    print(f"Loading model from {config.model_path}")
    model, tokenizer = load_model(config.model_path, device)

    # Read and process message
    message = config.secret_bits
    message = process_message(message)
    message = message[:config.message_length]
    secret_bits = [int(bit) for bit in message]
    print(f"Number of message bits used: {len(secret_bits)}")

    # Process context
    context_ids = process_context(tokenizer, config.context)
    context_tokens = context_ids.tolist()  # Catopter encode expects list
    print(f"Context tokens: {len(context_tokens)}")

    print("\n=== Encoding Phase ===")
    encode_start_time = time.time()
    generated_tokens, remain_len, nll_list, entropy_list = encode(
        model=model,
        history=context_tokens,
        secret_bits=secret_bits,
        chunk_size=config.chunk_size,
        T_bound=config.T_bound,
        top_p=config.top_p,
        top_k=config.top_k,
        sk=config.sk,
        max_n=config.max_n,
        prefix_len=config.prefix_len,
        device=device,
        use_hash_boundary=config.use_hash_boundary,
        hash_w=config.hash_w,
        hash_k=config.hash_k,
        hash_e=config.hash_e,
    )
    encoding_time = time.time() - encode_start_time
    encoded_len = len(secret_bits) - remain_len
    print(f"Encoded bits: {encoded_len}, Generated tokens: {len(generated_tokens)}")

    original_generated_tokens = list(generated_tokens)
    attack = config.attack.lower()
    attack_rate = config.attack_rate

    if attack != "none":
        attack_times = config.attack_times if hasattr(config, 'attack_times') else 5
        print(f"\n=== Applying Attack: {attack} with rate {attack_rate} for {attack_times} times ===")
        success_times = 0
        correct_rates = []
        correct_total = 0
        wrong_total = 0
        lost_total = 0
        decoding_times = []

        for _ in range(attack_times):
            generated_tokens = apply_attack(
                token_list=original_generated_tokens,
                attack_type=attack,
                rate=attack_rate,
                tokenizer=tokenizer
            )

            # Decoding Phase inside loop
            decode_start_time = time.time()
            reconstructed_bits, remain_bits, consumed, embedded_flags, chunk_scores = decode(
                tokens=generated_tokens,
                num_bits=encoded_len,
                chunk_size=config.chunk_size,
                T_bound=config.T_bound,
                sk=config.sk,
                prefix_len=config.prefix_len,
                force_output=False,
                use_hash_boundary=config.use_hash_boundary,
                hash_w=config.hash_w,
                hash_k=config.hash_k,
            )
            decoding_time = time.time() - decode_start_time

            if reconstructed_bits[:encoded_len] == secret_bits[:encoded_len] and encoded_len > 0:
                success_times += 1
            correct_rates.append(compute_correct_rate(
                ''.join(map(str, secret_bits[:encoded_len])),
                ''.join(map(str, reconstructed_bits[:encoded_len]))
            ))
            c, w, l = compare_bits(
                ''.join(map(str, secret_bits[:encoded_len])),
                ''.join(map(str, reconstructed_bits[:encoded_len]))
            )
            correct_total += c
            wrong_total += w
            lost_total += l

            decoding_times.append(decoding_time)

        success_rate = success_times / attack_times
        avg_correct_rate = sum(correct_rates) / len(correct_rates)
        avg_correct = correct_total / attack_times
        avg_wrong = wrong_total / attack_times
        avg_lost = lost_total / attack_times
        decoding_time = sum(decoding_times) / len(decoding_times)
        is_successful = success_rate == 1.0
        print(f"Decoding Success Rate after attack: {success_rate:.4f}")
        print(f"Bit Correct Rate after attack: {avg_correct_rate:.4f}")
        print(f"Bit classification after attack: correct={avg_correct:.2f} wrong={avg_wrong:.2f} lost={avg_lost:.2f} (per trial, N={encoded_len})")

    else:
        print("\n=== Decoding Phase ===")
        decode_start_time = time.time()
        reconstructed_bits, remain_bits, consumed, embedded_flags, chunk_scores = decode(
            tokens=generated_tokens,
            num_bits=encoded_len,
            chunk_size=config.chunk_size,
            T_bound=config.T_bound,
            sk=config.sk,
            prefix_len=config.prefix_len,
            force_output=False,
            use_hash_boundary=config.use_hash_boundary,
            hash_w=config.hash_w,
            hash_k=config.hash_k,
        )
        decoding_time = time.time() - decode_start_time

        is_successful = reconstructed_bits[:encoded_len] == secret_bits[:encoded_len] and encoded_len > 0
        success_rate = 1.0 if is_successful else 0.0
        avg_correct_rate = compute_correct_rate(
            ''.join(map(str, secret_bits[:encoded_len])),
            ''.join(map(str, reconstructed_bits[:encoded_len]))
        )
        avg_correct, avg_wrong, avg_lost = compare_bits(
            ''.join(map(str, secret_bits[:encoded_len])),
            ''.join(map(str, reconstructed_bits[:encoded_len]))
        )
        print("Original Encoded bits:", ''.join(map(str, secret_bits[:encoded_len])))
        print("Reconstructed bits:   ", ''.join(map(str, reconstructed_bits[:encoded_len])))
        print(f"Bit Correct Rate: {avg_correct_rate:.4f}")
        print(f"Bit classification: correct={avg_correct} wrong={avg_wrong} lost={avg_lost} (N={encoded_len})")

    # Calculate metrics
    avg_ppl = np.exp(np.mean(nll_list)) if nll_list else 0.0
    total_entropy = sum(entropy_list)

    original_tokens = len(context_tokens)
    generated_tokens_len = len(generated_tokens)

    tokens_per_second = generated_tokens_len / encoding_time if encoding_time > 0 else 0
    decode_tokens_per_second = generated_tokens_len / decoding_time if decoding_time > 0 else 0
    bits_encoded = encoded_len
    bits_per_second = bits_encoded / encoding_time if encoding_time > 0 else 0
    decode_bits_per_second = bits_encoded / decoding_time if decoding_time > 0 else 0
    entropy_utilization = bits_encoded / total_entropy if total_entropy > 0 else 0
    total_tokens = original_tokens + generated_tokens_len
    bits_per_token = encoded_len / generated_tokens_len if generated_tokens_len > 0 else 0

    stega_text = tokenizer.decode(original_generated_tokens, skip_special_tokens=True)
    results = {
        'config': config,
        'message': message,
        'generated_text': stega_text,
        'token_seq': original_generated_tokens,
        'remaining_bits': remain_len,
        'encoding_time': encoding_time,
        'decoding_time': decoding_time,
        'success': is_successful,
        'original_tokens': original_tokens,
        'generated_tokens': generated_tokens_len,
        'total_tokens': total_tokens,
        'bits_per_token': bits_per_token,
        'avg_ppl': avg_ppl,
        'avg_entropy': total_entropy/generated_tokens_len if generated_tokens_len > 0 else 0,
        'tokens_per_second': tokens_per_second,
        'decode_tokens_per_second': decode_tokens_per_second,
        'bits_per_second': bits_per_second,
        'decode_bits_per_second': decode_bits_per_second,
        'entropy_utilization': entropy_utilization,
        'success_rate': success_rate,
        'correct_rate': avg_correct_rate,
        'correct_bits': avg_correct,
        'wrong_bits': avg_wrong,
        'lost_bits': avg_lost,
    }

    return results


def _print_report(title: str, rows):
    """Print an aligned, single-run summary table."""
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    for label, value in rows:
        print(f"  {label:<24}  {value}")
    print(bar)


def _clip(text, limit=200):
    """Collapse whitespace and truncate a preview string with an ellipsis."""
    if text is None:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _print_text_block(title, text):
    """Print a text preview in its own wrapped block below the metrics."""
    import textwrap
    print(f"\n  {title}")
    print("  " + "-" * len(title))
    for line in textwrap.wrap(text, width=90) or [""]:
        print(f"  {line}")


def summarize_results(results: dict):
    correct_rate = results.get('correct_rate', 0.0)
    success_rate = results.get('success_rate', 1.0)
    corr = results.get('correct_bits', 0)
    wrong = results.get('wrong_bits', 0)
    lost = results.get('lost_bits', 0)

    total_bits = corr + wrong + lost
    if total_bits > 0:
        corr_pct = corr / total_bits * 100
        wrong_pct = wrong / total_bits * 100
        lost_pct = lost / total_bits * 100
    else:
        corr_pct = wrong_pct = lost_pct = 0.0

    config = results.get('config')
    prompt = config.get('context', '') if isinstance(config, dict) else getattr(config, 'context', '')
    generated = results.get('generated_text', '')

    rows = [
        ("Correct Rate", f"{correct_rate * 100:.2f}%"),
        ("Success Rate", f"{success_rate * 100:.2f}%"),
        ("Bit Classification", f"correct={corr_pct:.2f}%  wrong={wrong_pct:.2f}%  lost={lost_pct:.2f}%"),
        ("Encode Time", f"{results['encoding_time']:.2f} s"),
        ("Decode Time", f"{results['decoding_time']:.2f} s"),
        ("Encode Speed", f"{results['tokens_per_second']:,.1f} tok/s  |  "
                         f"{results['bits_per_second']:,.1f} bit/s"),
        ("Decode Speed", f"{results['decode_tokens_per_second']:,.1f} tok/s  |  "
                         f"{results['decode_bits_per_second']:,.1f} bit/s"),
        ("Tokens", f"original={results['original_tokens']}  "
                   f"generated={results['generated_tokens']}  "
                   f"total={results['total_tokens']}"),
        ("Bits per Token", f"{results['bits_per_token']:.4f}"),
        ("Entropy Utilization", f"{results['entropy_utilization'] * 100:.2f}%"),
        ("Average PPL", f"{results['avg_ppl']:.2f}"),
        ("Average Entropy", f"{results['avg_entropy']:.3f} bit/token"),
    ]
    _print_report("Evaluation Results", rows)
    _print_text_block("Original Prompt", _clip(prompt))
    _print_text_block("Generated Text", _clip(generated))


def save_results_to_json(results, filename="results.json"):
    config = results['config']
    if isinstance(config, dict):
        output_dir = config.get('output_dir', './results/catopter')
    else:
        output_dir = getattr(config, 'output_dir', './results/catopter')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, filename)

    # Create a copy to avoid modifying the original dict
    results_to_save = results.copy()
    if 'token_seq' in results_to_save:
        del results_to_save['token_seq']
    if 'config' in results_to_save and not isinstance(results_to_save['config'], dict):
        results_to_save['config'] = asdict(results_to_save['config'])

    # Helper for numpy and other types
    def default_serializer(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        return str(obj)

    data = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []

    data.append(results_to_save)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, default=default_serializer, ensure_ascii=False)

    print(f"Results appended to {filepath}")


def main(config: CatopterConfig):
    device = config.device if hasattr(config, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_results = test_catopter(config, device)
    summarize_results(test_results)
    save_results_to_json(test_results)
