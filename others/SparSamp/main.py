from typing import Tuple, List, Dict, Any
import torch
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import math
import json
import os
import numpy as np
import random

from .utils import load_model, read_binary_message
from .sparsamp import encode_spar, decode_spar
from .edits import apply_attack

@dataclass
class SparsampConfig:
    secret_bits: str
    message_length: int
    block_size: int
    context: str
    model_path: str
    random_seed: int
    top_p: float = 1.0
    output_dir: str = "./results/SparSamp"
    attack: str = "none"
    attack_rate: float = 0
    attack_times: int = 5
    device: str = "cuda"  # or "cpu"

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
        output_dir = config.get('output_dir', './results/sparsamp')
    else:
        output_dir = getattr(config, 'output_dir', './results/sparsamp')

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

def verify_token_ids(original_ids: torch.Tensor, regenerated_ids: torch.Tensor) -> bool:
    """Verify if original and regenerated token IDs match"""
    return torch.equal(torch.tensor(original_ids), regenerated_ids[0])

def process_message(message: str, block_size: int) -> str:
    """Process message to ensure length meets requirements"""
    if len(message) % block_size != 0:
        padding_length = block_size - (len(message) % block_size)
        message = message + '0' * padding_length
        print(f"Message length padded to {len(message)} bits")
    return message

def test_sparsamp(config: SparsampConfig, device: torch.device) -> Dict[str, Any]:
    """Run the SparSamp encode/decode pipeline and return results."""
    print("\n=== Initialization Phase ===")
    message = config.secret_bits
    message = message[:config.message_length]
    message = process_message(message, config.block_size)
    print(f"Number of message bits used: {len(message)}")
    
    try:
        model, tokenizer = load_model(config.model_path, device)
    except:
        print("Failed to load model locally, load GPT-2 directly from Transformers library")
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        print("Model loaded successfully")
        model.to(device)
        model.eval()

    
    context = tokenizer.encode(config.context, return_tensors='pt').to(device)

    print("\n=== Encoding Phase ===")
    encode_start_time = time.time()
    generated_ids, encoded_message, entropy_list, nll_list, stat_time, model_time = encode_spar(
        model=model,
        context=context,
        message_bits=message,
        device=device,
        block_size=config.block_size,
        top_p=config.top_p,
        random_seed=config.random_seed
    )
    encode_time = time.time() - encode_start_time

    stega_text = tokenizer.decode(generated_ids)
    print(f"Generated steganographic text: {stega_text}")
    
    original_generated_ids = list(generated_ids)
    attack = config.attack.lower()
    attack_rate = config.attack_rate
    
    encoded_message_bits = len(encoded_message) * config.block_size
    
    if attack != "none":
        attack_times = config.attack_times if hasattr(config, 'attack_times') else 5
        print(f"\n=== Applying Attack: {attack} with rate {attack_rate} for {attack_times} times ===")
        success_times = 0
        correct_rates = []
        correct_total = 0
        wrong_total = 0
        lost_total = 0
        decoding_times = []
        try:
            for _ in range(attack_times):
                generated_ids = apply_attack(
                    token_list=original_generated_ids,
                    attack_type=attack,
                    rate=attack_rate,
                    tokenizer=tokenizer
                )

                decode_start_time = time.time()
                reconstructed_message = decode_spar(
                    model=model,
                    generated_ids=generated_ids,
                    context=context,
                    device=device,
                    block_size=config.block_size,
                    top_p=config.top_p,
                    random_seed=config.random_seed
                )
                decode_time = time.time() - decode_start_time

                if encoded_message == reconstructed_message:
                    success_times += 1
                recon_trimmed = reconstructed_message[:len(encoded_message)]
                correct_rates.append(compute_correct_rate(
                    ''.join(encoded_message), ''.join(recon_trimmed)
                ))
                c, w, l = compare_bits(
                    ''.join(encoded_message), ''.join(recon_trimmed)
                )
                correct_total += c
                wrong_total += w
                lost_total += l

                decoding_times.append(decode_time)

            success_rate = success_times / attack_times
            avg_correct_rate = sum(correct_rates) / len(correct_rates)
            avg_correct = correct_total / attack_times
            avg_wrong = wrong_total / attack_times
            avg_lost = lost_total / attack_times
            decode_time = sum(decoding_times) / len(decoding_times)
            is_successful = success_rate == 1.0
            print(f"Decoding Success Rate after attack: {success_rate:.4f}")
            print(f"Bit Correct Rate after attack: {avg_correct_rate:.4f}")
            print(f"Bit classification after attack: correct={avg_correct:.2f} wrong={avg_wrong:.2f} lost={avg_lost:.2f} (per trial, N={encoded_message_bits})")

        except Exception as e:
            print(f"Error during attack/decoding: {e}")
            print("Skipping attack results. For it has no robustness.")
            success_rate = 0.0
            avg_correct_rate = 0.0
            avg_correct = 0.0
            avg_wrong = 0.0
            avg_lost = 0.0
            decode_time = 0.0
            is_successful = False

    else:
        # stega_file = save_stega_text(stega_text, config.stega_file)
        # print(f"Steganographic text saved to: {stega_file}")

        # loaded_stega_text = load_stega_text(stega_file)
        regenerated_ids = tokenizer.encode(stega_text, return_tensors='pt')

        is_same = verify_token_ids(generated_ids, regenerated_ids)
        print(f"Whether there is Token Ambiguity: {'No' if is_same else 'Yes'}")
        # We decode messages from sentences that without Token Ambiguity.
        print("\n=== Decoding Phase ===")
        decode_start_time = time.time()
        reconstructed_message = decode_spar(
            model=model,
            generated_ids=generated_ids,# We decode messages from sentences that without Token Ambiguity.
            context=context,
            device=device,
            block_size=config.block_size,
            top_p=config.top_p,
            random_seed=config.random_seed
        )
        decode_time = time.time() - decode_start_time

        is_successful = encoded_message == reconstructed_message
        success_rate = 1.0 if is_successful else 0.0
        recon_trimmed = reconstructed_message[:len(encoded_message)]
        avg_correct_rate = compute_correct_rate(
            ''.join(encoded_message), ''.join(recon_trimmed)
        )
        avg_correct, avg_wrong, avg_lost = compare_bits(
            ''.join(encoded_message), ''.join(recon_trimmed)
        )
        print("Original encoded blocks:", ''.join(encoded_message))
        print("Reconstructed blocks:   ", ''.join(recon_trimmed))
        print(f"Bit Correct Rate: {avg_correct_rate:.4f}")
        print(f"Bit classification: correct={avg_correct} wrong={avg_wrong} lost={avg_lost} (N={encoded_message_bits})")


    token_num_generated = len(generated_ids)
    # encoded_message_bits = len(encoded_message) * config.block_size # Moved up
    
    total_entropy = sum(entropy_list)
    utilization = (encoded_message_bits / total_entropy * 100) if total_entropy != 0 else 0
    Embedding_Speed = encoded_message_bits / (encode_time - stat_time)
    Decoding_Speed = encoded_message_bits / decode_time
    ATST = encode_time / token_num_generated
    Generation_Speed = token_num_generated / encode_time
    SITR = (encode_time - model_time - stat_time) / encode_time
    Embedding_Rate = encoded_message_bits / token_num_generated

    # Calculate new metrics
    avg_ppl = math.exp(sum(nll_list) / len(nll_list)) if nll_list else 0.0

    original_tokens = len(context[0]) # context is tensor
    generated_tokens_len = len(generated_ids)

    tokens_per_second = generated_tokens_len / encode_time if encode_time > 0 else 0
    decode_tokens_per_second = generated_tokens_len / decode_time if decode_time > 0 else 0

    bits_encoded = encoded_message_bits
    bits_per_second = bits_encoded / encode_time if encode_time > 0 else 0
    decode_bits_per_second = bits_encoded / decode_time if decode_time > 0 else 0

    entropy_utilization = bits_encoded / total_entropy if total_entropy > 0 else 0
    total_tokens = original_tokens + generated_tokens_len
    bits_per_token = encoded_message_bits / generated_tokens_len if generated_tokens_len > 0 else 0


    results = {
        'config': config,
        'message': message,
        'generated_text': tokenizer.decode(original_generated_ids),
        'token_seq': original_generated_ids,
        'success': is_successful,
        'token_num_generated': token_num_generated,
        'encoded_message_bits': encoded_message_bits,
        'utilization': utilization,
        'Embedding_Speed': Embedding_Speed,
        'Decoding_Speed': Decoding_Speed,
        'ATST': ATST,
        'Generation_Speed': Generation_Speed,
        'SITR': SITR,
        'Embedding_Rate': Embedding_Rate,
        # New metrics
        'encoding_time': encode_time,
        'decoding_time': decode_time,
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

def main(config: SparsampConfig):
    device = config.device if hasattr(config, 'device') else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_results = test_sparsamp(config, device)    
    summarize_results(test_results)
    save_results_to_json(test_results)
