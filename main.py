import argparse
import torch
import warnings
from dataclasses import asdict

# Import from others (registered methods)
from others import (
    catopter_main, CatopterConfig,
    sparsamp_main, SparsampConfig,
    ars_main, ArsConfig,
    anstega_main, AnstegaConfig,
)

import random
import numpy as np
random.seed(42)
default_bits = ''.join(random.choice(['0', '1']) for _ in range(1024))
default_context = "Can you tell me a long story about a beautiful sunrise?"
default_model_path = "./llm/Llama-3.1-8B"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_config(name, config):
    print(f"\n{'=' * 70}")
    print(f"  Running {name}")
    print(f"{'=' * 70}")
    for k, v in asdict(config).items():
        s = str(v)
        if len(s) > 48:
            s = s[:45] + "..."
        print(f"  {str(k):<24}  {s}")
    print(f"{'=' * 70}")

def main():

    parser = argparse.ArgumentParser(description="Steganography Method Runner")

    parser.add_argument("--method", type=str, default="catopter",
                        choices=["catopter", "sparsamp", "ars", "anstega"],
                        help="The steganography method to run (REQUIRED)")

    parser.add_argument("--secret_bits", type=str, default=None,
                        help="The secret bits to encode. Can be a string or a file path. If not provided, uses message_bits.txt")
    parser.add_argument("--message_length", type=int, default=None,
                        help="Length of the message (in bits) to encode. Default: 64")
    parser.add_argument("--context", type=str, default=None,
                        help="Context or prompt for generation. If not provided, uses prompts.json")
    parser.add_argument("--model_path", type=str, default=default_model_path,
                        help="Path to the model or HuggingFace model name. Default: openai-community/gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu). Default: cuda if available")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed. Default: 42")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling parameter. Default: 1.0")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results. Defaults to ./results/<method>")
    parser.add_argument("--attack", type=str, default="none",
                        help="Attack type. Default: none")
    parser.add_argument("--attack_rate", type=float, default=0.1,
                        help="Attack rate. Default: 0.1")
    parser.add_argument("--attack_times", type=int, default=5,
                        help="Number of attack attempts for evaluation. Default: 5")

    # Method Specific Arguments
    # Catopter
    group_catopter = parser.add_argument_group('Catopter Config')
    group_catopter.add_argument("--chunk_size", type=int, default=1, help="[Catopter] Chunk size")
    group_catopter.add_argument("--T_bound", type=float, default=10.0, help="[Catopter] T bound")
    group_catopter.add_argument("--prefix_len", type=int, default=1, help="[Catopter] Prefix length")
    group_catopter.add_argument("--sk", type=str, default="42", help="[Catopter] Secret key")
    group_catopter.add_argument("--top_k", type=int, default=None, help="[Catopter] Top-k (None = full vocab)")
    group_catopter.add_argument("--catopter_max_n", type=int, default=1024, help="[Catopter] Max N")
    group_catopter.add_argument("--use_hash_boundary", type=bool, default=True, help="[Catopter] Use hash boundary")
    group_catopter.add_argument("--hash_w", type=int, default=3, help="[Catopter] Hash boundary window size")
    group_catopter.add_argument("--hash_e", type=int, default=1, help="[Catopter] Hash boundary tolerated corruptions")
    group_catopter.add_argument("--hash_k", type=int, default=1, help="[Catopter] Hash boundary predicate bits")

    # SparSamp
    group_sparsamp = parser.add_argument_group('SparSamp Config')
    group_sparsamp.add_argument("--block_size", type=int, default=32, help="[SparSamp] Block size")

    # ARS
    group_ars = parser.add_argument_group('ARS Config')
    group_ars.add_argument("--error_bound", type=float, default=0.2, help="[ARS] Error bound")
    group_ars.add_argument("--ars_max_n", type=int, default=512, help="[ARS] Max N")

    # ANStega
    group_anstega = parser.add_argument_group('ANStega Config')
    group_anstega.add_argument("--token_num_need_generated", type=int, default=128, help="[ANStega] Number of stego tokens")
    group_anstega.add_argument("--anstega_precision", type=int, default=32, help="[ANStega] ANS state precision in bits")
    group_anstega.add_argument("--ans_t", type=int, default=16, help="[ANStega] ANS scaling factor")

    args = parser.parse_args()

    # --- Pre-processing / Defaults Handling ---
    print(f"\n[Configuration] Selected Method: {args.method}")

    # Handle secret_bits
    if args.secret_bits is None:
        args.secret_bits = default_bits

    # Handle message_length (limit bits)
    if args.message_length is None:
        print("[Defaults] 'message_length' not configured. Using default: 64")
        args.message_length = 64

    # Truncate secret_bits if needed (or check length)
    if len(args.secret_bits) < args.message_length:
        warnings.warn(f"Warning: Provided bits length ({len(args.secret_bits)}) is less than message_length ({args.message_length}). Repeating bits.")
        args.secret_bits = (args.secret_bits * (args.message_length // len(args.secret_bits) + 1))[:args.message_length]
    else:
        args.secret_bits = args.secret_bits[:args.message_length]

    # Handle context
    if args.context is None:
        print(f"[Defaults] 'context' not configured. Using default {default_context}.")
        args.context = default_context
    else:
        print(f"[Configuration] Using provided context: {args.context}")

    # Handle output_dir
    if args.output_dir is None:
        args.output_dir = f"./results/{args.method}"
        print(f"[Defaults] 'output_dir' not configured. Using default: {args.output_dir}")

    # Set random seed for reproducibility
    set_seed(args.random_seed)

    if args.method == "catopter":
        config = CatopterConfig(
            secret_bits=args.secret_bits,
            message_length=args.message_length,
            context=args.context,
            model_path=args.model_path,
            random_seed=args.random_seed,
            top_p=args.top_p,
            output_dir=args.output_dir,
            attack=args.attack,
            attack_rate=args.attack_rate,
            attack_times=args.attack_times,
            device=args.device,
            # Specifics
            chunk_size=args.chunk_size,
            T_bound=args.T_bound,
            prefix_len=args.prefix_len,
            sk=args.sk,
            top_k=args.top_k,
            max_n=args.catopter_max_n,
            use_hash_boundary=args.use_hash_boundary,
            hash_w=args.hash_w,
            hash_e=args.hash_e,
            hash_k=args.hash_k,
        )
        print_config("Catopter", config)
        catopter_main(config)

    elif args.method == "sparsamp":
        config = SparsampConfig(
            secret_bits=args.secret_bits,
            message_length=args.message_length,
            context=args.context,
            model_path=args.model_path,
            random_seed=args.random_seed,
            top_p=args.top_p,
            output_dir=args.output_dir,
            attack=args.attack,
            attack_rate=args.attack_rate,
            attack_times=args.attack_times,
            device=args.device,
            # Specifics
            block_size=args.block_size
        )

        print_config("SparSamp", config)
        sparsamp_main(config)

    elif args.method == "ars":
        config = ArsConfig(
            secret_bits=args.secret_bits,
            message_length=args.message_length,
            context=args.context,
            model_path=args.model_path,
            random_seed=args.random_seed,
            output_dir=args.output_dir,
            attack=args.attack,
            attack_rate=args.attack_rate,
            attack_times=args.attack_times,
            device=args.device,
            # Specifics
            error_bound=args.error_bound,
            max_n=args.ars_max_n
        )
        print_config("ARS", config)
        ars_main(config)

    elif args.method == "anstega":
        config = AnstegaConfig(
            secret_bits=args.secret_bits,
            message_length=args.message_length,
            context=args.context,
            model_path=args.model_path,
            random_seed=args.random_seed,
            top_p=args.top_p,
            output_dir=args.output_dir,
            attack=args.attack,
            attack_rate=args.attack_rate,
            attack_times=args.attack_times,
            device=args.device,
            # Specifics
            token_num_need_generated=args.token_num_need_generated,
            precision=args.anstega_precision,
            ans_t=args.ans_t
        )
        print_config("ANStega", config)
        anstega_main(config)

if __name__ == "__main__":
    main()
