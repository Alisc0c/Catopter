"""exp3 — capacity table, Catopter vs ARS (plan §2 exp3)"""

import argparse
import os
import random
import time

import numpy as np
import torch

from .common import (
    Checkpoint,
    METHODS,
    RESULTS_DIR,
    common_args,
    context_texts,
    default_seeds,
    generate_secret_bits,
    load_model,
    save_json,
    set_seed,
    stats,
    suppress_tqdm,
)

suppress_tqdm()

# ══════════════════════════════════════════════════════════════════════
#  Experiment parameters — designed per experiment; CLI overrides these.
# ══════════════════════════════════════════════════════════════════════
CAT_TOPS = [1.0, 0.95, 0.9, 0.8]     # Catopter top-p sweep (capacity surface)
MSG_LEN = 16                          # message bits offered

# Catopter (cs=1 Pareto point, hash boundary on)
CAT = dict(chunk_size=1, T_bound=10.0, prefix_len=1, sk="42", top_k=None,
           max_n=1024, use_hash_boundary=True, hash_w=3, hash_e=1, hash_k=1, top_p=1.0)
# ARS (no top-p: its encode() takes error_bound only)
ARS = dict(error_bound=0.2, max_n=1500)


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
    """Per-method encode kwargs from CLI (top-p overridden per sweep point)."""
    return {
        "Catopter": dict(chunk_size=args.chunk_size, T_bound=args.T_bound,
                         prefix_len=CAT["prefix_len"], sk=CAT["sk"],
                         top_k=CAT["top_k"], max_n=CAT["max_n"],
                         use_hash_boundary=CAT["use_hash_boundary"],
                         hash_w=args.hash_w, hash_e=args.hash_e, hash_k=args.hash_k,
                         top_p=CAT["top_p"]),
        "ARS": dict(error_bound=args.error_bound, max_n=args.ars_max_n),
    }


def _util(enc):
    total = sum(enc["entropy_list"])
    return enc["n_embedded"] / total if total > 0 else 0.0


def print_table(summary):
    hdr = f"{'Method':<10}|{'top-p':<6}|{'bits/token':<12}|{'entr-util %':<14}|{'entropy':<10}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in summary["rows"]:
        print(f"{r['method']:<10}|{r['top_p']:<6}|{r['bits_per_token'][0]:<12.4f}|"
              f"{r['entropy_util'][0]*100:<14.1f}|{r['entropy'][0]:<10.3f}")


def run(model_path, samples, seeds, device, output_dir,
        lang="en", params=None):
    model, tokenizer = load_model(model_path, device)
    contexts = context_texts(lang)[:samples]
    out_dir = output_dir or os.path.join(RESULTS_DIR, "exp3")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = Checkpoint("exp3", out_dir, model_path)
    t0_all = time.time()

    rows = []
    for method, tops in (("Catopter", CAT_TOPS), ("ARS", [1.0])):
        for top_p in tops:
            print(f"\n===== {method} top-p={top_p} =====", flush=True)
            for ctx_idx, context in enumerate(contexts):
                for seed in seeds:
                    key = ckpt.enc_key(method, ctx_idx, seed) + f"_{top_p}"
                    if not ckpt.get_encode(key):
                        kw = dict(params[method], random_seed=seed)
                        if method != "ARS":
                            kw["top_p"] = top_p
                        secret = generate_secret_bits(MSG_LEN,
                                                        random.Random(seed))
                        set_seed(seed)
                        try:
                            enc = METHODS[method]["encode"](
                                model=model, tokenizer=tokenizer, context=context,
                                secret_bits=secret, device=device, **kw)
                        except Exception as e:
                            print(f"    [fail] p={top_p} seed={seed}: {e}", flush=True)
                            continue
                        ckpt.set_encode(key, enc)
                        ckpt.save()
                    else:
                        enc = ckpt.get_encode(key)
                    if enc.get("n_tokens", 0) <= 0:
                        # token budget exhausted → 0 bits embedded.  Mark as a
                        # failure and EXCLUDE from the averages (a 0-entropy
                        # "sample" is not a valid capacity measurement — it
                        # would drag the mean down and fake low entropy).
                        rows.append({"method": method, "top_p": top_p,
                                     "bits_per_token": float("nan"),
                                     "entropy_util": float("nan"),
                                     "entropy": float("nan"),
                                     "failed": True})
                        continue
                    ent_mean = (sum(enc["entropy_list"]) / len(enc["entropy_list"])
                                if enc["entropy_list"] else 0.0)
                    bpt = enc["n_embedded"] / enc["n_tokens"]
                    eu = _util(enc)
                    rows.append({"method": method, "top_p": top_p,
                                 "bits_per_token": bpt,
                                 "entropy_util": eu, "entropy": ent_mean,
                                 "failed": False})

    save_json(rows, os.path.join(out_dir, "exp3_records.json"))   # persist first

    summary = {"rows": []}
    for method, tops in (("Catopter", CAT_TOPS), ("ARS", [1.0])):
        for top_p in tops:
            sub = [r for r in rows if r["method"] == method and r["top_p"] == top_p]
            ok_sub = [r for r in sub if not r.get("failed")]
            if not ok_sub:
                continue
            summary["rows"].append({"method": method, "top_p": top_p,
                                    "bits_per_token": stats(
                                        [r["bits_per_token"] for r in ok_sub]),
                                    "entropy_util": stats(
                                        [r["entropy_util"] for r in ok_sub]),
                                    "entropy": stats([r["entropy"] for r in ok_sub]),
                                    "n_samples": len(ok_sub),
                                    "n_failed": len(sub) - len(ok_sub)})
    p = os.path.join(out_dir, "exp3_summary.json")
    save_json(summary, p)                                        # persist first
    print(f"\nSaved summary → {p}")

    print_table(summary)

    del model, tokenizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"\n[exp3 done] {time.time()-t0_all:.1f}s", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description="exp3 — capacity (Catopter vs ARS)")
    common_args(p)
    add_params(p)
    a = p.parse_args()
    params = build_params(a)
    run(a.model_path, a.samples, a.seeds or default_seeds(), a.device,
        a.output_dir, a.lang, params)


if __name__ == "__main__":
    main()
