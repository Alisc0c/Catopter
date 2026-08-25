"""exp7 — Catopter cross-model availability table (plan §5 exp7)"""

import argparse
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

from .common import (
    Checkpoint,
    PROJECT_ROOT,
    RESULTS_DIR,
    common_args,
    context_texts,
    decode_catopter,
    default_seeds,
    encode_catopter,
    generate_secret_bits,
    load_json,
    load_model,
    save_json,
    set_seed,
    stats,
    suppress_tqdm,
)

suppress_tqdm()

TOPS = [0.8, 0.9, 0.95, 1.0]
CS, T_BOUND, MSG_LEN = 1, 10.0, 4


def _run_model(path, contexts, seeds, device, ckpt, key):
    from catopter.prf import warmup_pool
    model, tokenizer = load_model(path, device)
    warmup_pool()          # one-time pool spawn must not pollute timings
    rows = []
    for top_p in TOPS:
        for ctx_idx, context in enumerate(contexts):
            for seed in seeds:
                k = ckpt.enc_key(key, ctx_idx, seed) + f"_{top_p}"
                if not ckpt.get_encode(k):
                    secret = generate_secret_bits(MSG_LEN, random.Random(seed))
                    set_seed(seed)
                    try:
                        enc = encode_catopter(model=model, tokenizer=tokenizer,
                                                context=context, secret_bits=secret,
                                                device=device, random_seed=seed,
                                                chunk_size=CS, T_bound=T_BOUND,
                                                top_p=top_p)
                    except Exception as e:
                        print(f"    [fail] p={top_p} seed={seed}: {e}", flush=True)
                        continue
                    if enc.get("n_embedded", 0) <= 0 or enc.get("n_tokens", 0) <= 0:
                        continue
                    ckpt.set_encode(k, enc)
                    ckpt.save()
                else:
                    enc = ckpt.get_encode(k)
                te = sum(enc["entropy_list"])
                # decode is LLM-free and Top-p independent; time it on THIS
                # record's own token sequence so every row reports its own
                # measurement instead of one number copied across rows.
                t0 = time.time()
                decode_catopter(enc["token_seq"], enc["n_embedded"],
                                  chunk_size=CS, T_bound=T_BOUND)
                dec_dt = time.time() - t0
                rows.append({"top_p": top_p,
                             "encode_tok_s": enc["n_tokens"] / enc["encoding_time"]
                             if enc["encoding_time"] > 0 else 0.0,
                             "decode_tok_s": enc["n_tokens"] / dec_dt
                             if dec_dt > 0 else 0.0,
                             "bits_per_token": enc["n_embedded"] / enc["n_tokens"],
                             "entropy_util": enc["n_embedded"] / te if te > 0 else 0.0,
                             "entropy": te / enc["n_tokens"],
                             "ppl": float(np.exp(np.mean(enc["nll_list"])))
                             if enc["nll_list"] else 0.0})
    del model, tokenizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def _run_model_only(key, path, contexts, seeds, device, out_dir, model_path):
    """One model in a fresh process: saves its rows/dec to {key}_data.json."""
    ckpt = Checkpoint("exp7", out_dir, model_path)
    rows = _run_model(path, contexts, seeds, device, ckpt, key)
    save_json({"rows": rows},
              os.path.join(out_dir, f"{key}_data.json"))


def _aggregate(data):
    """Per-(model, top-p) means; decode throughput comes from each row's own
    measurement (it is Top-p independent up to noise, which the spread shows)."""
    summary = {"rows": []}
    for key, d in data.items():
        for top_p in TOPS:
            sub = [r for r in d["rows"] if r["top_p"] == top_p]
            if not sub:
                continue
            decs = [r["decode_tok_s"] for r in sub if r.get("decode_tok_s", 0) > 0]
            summary["rows"].append({"model": key, "top_p": top_p,
                                    "encode_tok_s": stats(
                                        [r["encode_tok_s"] for r in sub]),
                                    "decode_tok_s": stats(decs)
                                    if decs else [0.0, 0.0, 0.0],
                                    "bits_per_token": stats(
                                        [r["bits_per_token"] for r in sub]),
                                    "entropy_util": stats(
                                        [r["entropy_util"] for r in sub]),
                                    "entropy": stats([r["entropy"] for r in sub]),
                                    "ppl": stats([r["ppl"] for r in sub]),
                                    "n_samples": len(sub)})
    return summary


def print_table(summary):
    hdr = f"{'Model':<12}|{'top-p':<6}|{'enc tok/s':<12}|{'dec tok/s':<12}|{'bits/token':<12}|{'entr-util %':<14}|{'entropy':<10}|{'PPL':<10}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in summary["rows"]:
        print(f"{r['model']:<12}|{r['top_p']:<6}|{r['encode_tok_s'][0]:<12.1f}|"
              f"{r['decode_tok_s'][0]:<12.1f}|{r['bits_per_token'][0]:<12.4f}|"
              f"{r['entropy_util'][0]*100:<14.1f}|{r['entropy'][0]:<10.3f}|{r['ppl'][0]:<10.3f}")


def run(model_path, models_arg, samples, seeds, device, output_dir,
        lang="en"):
    out_dir = output_dir or os.path.join(RESULTS_DIR, "exp7")
    os.makedirs(out_dir, exist_ok=True)
    t0_all = time.time()

    specs = []
    for tok in models_arg.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            specs.append((k, os.path.join(PROJECT_ROOT, v)))
    if not specs:
        specs = [("model", model_path)]

    data = {}
    for key, path in specs:
        print(f"\n===== {key} ({path}) =====", flush=True)
        # Each model runs in its own subprocess so GPU memory is fully freed
        # between 8-9B models: in-process `del`+`empty_cache()` leaks for
        # trust_remote_code models and the 4th model OOMs.
        sub = subprocess.run(
            [sys.executable, "-m", "exp.exp7", "--model_only", f"{key}={path}",
             "--model_path", model_path, "--samples", str(samples),
             "--seeds"] + [str(s) for s in seeds] +
            ["--device", device, "--output_dir", out_dir, "--lang", lang],
        )
        if sub.returncode != 0:
            print(f"    [SKIP] {key} failed (exit {sub.returncode}) — retry later", flush=True)
            continue
        data[key] = load_json(os.path.join(out_dir, f"{key}_data.json"))

    if not data:
        raise RuntimeError(
            "exp7: every model failed (GPU unavailable / wrong model path?) "
            "— refusing to write an empty summary")

    save_json(data, os.path.join(out_dir, "exp7_records.json"))     # persist first

    summary = _aggregate(data)
    p = os.path.join(out_dir, "exp7_summary.json")
    save_json(summary, p)                                           # persist first
    print(f"\nSaved summary → {p}")

    print_table(summary)

    print(f"\n[exp7 done] {time.time()-t0_all:.1f}s", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description="exp7 — cross-model availability")
    common_args(p)
    p.add_argument("--models", type=str, default="",
                   help='"key1=path1 key2=path2"')
    p.add_argument("--model_only", type=str, default=None,
                   help="internal: run a single model (key=path) in a subprocess")
    a = p.parse_args()
    if a.model_only:
        key, path = a.model_only.split("=", 1)
        out_dir = a.output_dir or os.path.join(RESULTS_DIR, "exp7")
        os.makedirs(out_dir, exist_ok=True)
        _run_model_only(key, os.path.join(PROJECT_ROOT, path),
                        context_texts(a.lang)[:a.samples],
                        a.seeds or default_seeds(), a.device, out_dir, a.model_path)
        return
    run(a.model_path, a.models, a.samples, a.seeds or default_seeds(),
        a.device, a.output_dir, a.lang)


if __name__ == "__main__":
    main()
