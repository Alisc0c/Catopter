"""exp4 — encode/decode speed table (plan §3 exp4)"""

import argparse
import os
import random
import time

import numpy as np
import torch

from .common import (
    Checkpoint,
    METHODS,
    METHOD_ORDER,
    RESULTS_DIR,
    common_args,
    context_texts,
    default_seeds,
    generate_secret_bits,
    load_model,
    make_decoder,
    save_json,
    set_seed,
    stats,
    suppress_tqdm,
)

suppress_tqdm()

# ══════════════════════════════════════════════════════════════════════
#  Experiment parameters — designed per experiment; CLI overrides these.
# ══════════════════════════════════════════════════════════════════════
TOPS = [0.8, 0.9, 0.95, 1.0]          # top-p sweep (full encode speed)
ML = {"Catopter": 4, "ARS": 4,
      "SparSamp": 128, "AnStega": 128}

# Catopter
CAT = dict(chunk_size=1, T_bound=10.0, prefix_len=1, sk="42", top_k=None,
           max_n=1024, use_hash_boundary=True, hash_w=3, hash_e=1, hash_k=1, top_p=1.0)
# ARS (no top-p: its encode() takes error_bound only)
ARS = dict(error_bound=0.2, max_n=1500)
# SparSamp (top-p fixed to 1.0: arithmetic coding needs the full vocab)
SPARSAMP = dict(block_size=32, top_p=1.0)
# AnStega
ANSTEGA = dict(token_num_need_generated=100, precision=32, ans_t=16, top_p=0.95)


def add_params(parser):
    """This exp's parameter entry (defaults = design above, CLI overrides)."""
    parser.add_argument("--chunk_size", type=int, default=CAT["chunk_size"])
    parser.add_argument("--T_bound", type=float, default=CAT["T_bound"])
    parser.add_argument("--hash_w", type=int, default=CAT["hash_w"])
    parser.add_argument("--hash_e", type=int, default=CAT["hash_e"])
    parser.add_argument("--hash_k", type=int, default=CAT["hash_k"])
    parser.add_argument("--error_bound", type=float, default=ARS["error_bound"])
    parser.add_argument("--ars_max_n", type=int, default=ARS["max_n"])
    return parser


def build_params(args):
    """Per-method encode kwargs from CLI (unlisted knobs keep design values)."""
    return {
        "Catopter": dict(chunk_size=args.chunk_size, T_bound=args.T_bound,
                         prefix_len=CAT["prefix_len"], sk=CAT["sk"],
                         top_k=CAT["top_k"], max_n=CAT["max_n"],
                         use_hash_boundary=CAT["use_hash_boundary"],
                         hash_w=args.hash_w, hash_e=args.hash_e, hash_k=args.hash_k,
                         top_p=CAT["top_p"]),
        "ARS": dict(error_bound=args.error_bound, max_n=args.ars_max_n),
        "SparSamp": dict(SPARSAMP),
        "AnStega": dict(ANSTEGA),
    }


def print_table(summary):
    hdr = f"{'Method':<10}|{'top-p':<6}|{'enc tok/s':<12}|{'dec tok/s':<12}|{'enc bit/s':<12}|{'dec bit/s':<12}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in summary["rows"]:
        print(f"{r['method']:<10}|{r['top_p']:<6}|{r['encode_tok_s'][0]:<12.1f}|"
              f"{r['decode_tok_s'][0]:<12.1f}|{r['encode_bit_s'][0]:<12.1f}|"
              f"{r['decode_bit_s'][0]:<12.1f}")


def run(model_path, samples, seeds, device, output_dir,
        lang="en", params=None):
    from catopter.prf import warmup_pool
    model, tokenizer = load_model(model_path, device)
    warmup_pool()          # one-time pool spawn must not pollute timings
    contexts = context_texts(lang)[:samples]
    out_dir = output_dir or os.path.join(RESULTS_DIR, "exp4")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = Checkpoint("exp4", out_dir, model_path)
    t0_all = time.time()

    rows = []
    for method in METHOD_ORDER:
        tops = [1.0] if method == "ARS" else TOPS
        for top_p in tops:
            print(f"\n===== {method} top-p={top_p} =====", flush=True)
            for ctx_idx, context in enumerate(contexts):
                for seed in seeds:
                    key = ckpt.enc_key(method, ctx_idx, seed) + f"_{top_p}"
                    if not ckpt.get_encode(key):
                        ml = ML[method]
                        kw = dict(params[method], random_seed=seed)
                        if method not in ("ARS", "SparSamp"):
                            kw["top_p"] = top_p
                        secret = generate_secret_bits(ml, random.Random(seed))
                        set_seed(seed)
                        try:
                            enc = METHODS[method]["encode"](
                                model=model, tokenizer=tokenizer, context=context,
                                secret_bits=secret, device=device, **kw)
                        except Exception as e:
                            print(f"    [encode fail] {method} p={top_p} seed={seed}: "
                                  f"{e}", flush=True)
                            continue
                        if enc.get("n_embedded", 0) <= 0 or enc.get("n_tokens", 0) <= 0:
                            continue
                        try:
                            bits, dec_t = make_decoder(method, enc, model, tokenizer,
                                                         context, device, top_p, seed)(
                                list(enc["token_seq"]))
                        except Exception as e:
                            print(f"    [decode fail] {method} p={top_p}: {e}",
                                  flush=True)
                            continue
                        enc["dec_time"] = dec_t
                        ckpt.set_encode(key, enc)
                        ckpt.save()
                    else:
                        enc = ckpt.get_encode(key)
                        dec_t = enc.get("dec_time", 0.0)
                    nt, nb = enc["n_tokens"], enc["n_embedded"]
                    rows.append({"method": method, "top_p": top_p,
                                 "encode_tok_s": nt / enc["encoding_time"]
                                 if enc["encoding_time"] > 0 else 0.0,
                                 "decode_tok_s": nt / dec_t if dec_t > 0 else 0.0,
                                 "encode_bit_s": nb / enc["encoding_time"]
                                 if enc["encoding_time"] > 0 else 0.0,
                                 "decode_bit_s": nb / dec_t if dec_t > 0 else 0.0})

    if not rows:
        raise RuntimeError(
            "exp4: no encode/decode trial succeeded "
            "(GPU unavailable / wrong model path?) — refusing to write an empty summary")

    save_json(rows, os.path.join(out_dir, "exp4_records.json"))    # persist first

    summary = {"rows": []}
    for method in METHOD_ORDER:
        tops = [1.0] if method == "ARS" else TOPS
        for top_p in tops:
            sub = [r for r in rows if r["method"] == method and r["top_p"] == top_p]
            if not sub:
                continue
            summary["rows"].append({"method": method, "top_p": top_p,
                                    "encode_tok_s": stats(
                                        [r["encode_tok_s"] for r in sub]),
                                    "decode_tok_s": stats(
                                        [r["decode_tok_s"] for r in sub]),
                                    "encode_bit_s": stats(
                                        [r["encode_bit_s"] for r in sub]),
                                    "decode_bit_s": stats(
                                        [r["decode_bit_s"] for r in sub]),
                                    "n_samples": len(sub)})
    p = os.path.join(out_dir, "exp4_summary.json")
    save_json(summary, p)                                          # persist first
    print(f"\nSaved summary → {p}")

    print_table(summary)

    del model, tokenizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"\n[exp4 done] {time.time()-t0_all:.1f}s", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description="exp4 — encode/decode speed table")
    common_args(p)
    add_params(p)
    a = p.parse_args()
    params = build_params(a)
    run(a.model_path, a.samples, a.seeds or default_seeds(), a.device,
        a.output_dir, a.lang, params)


if __name__ == "__main__":
    main()
