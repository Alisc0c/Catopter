"""exp6 — Catopter hash-strategy table (plan §4 exp6)"""

import argparse
import os
import random
import time

import numpy as np
import torch

from .common import (
    Checkpoint,
    RESULTS_DIR,
    common_args,
    context_texts,
    default_seeds,
    encode_catopter,
    evaluate_attack_trial,
    generate_secret_bits,
    load_model,
    make_decoder,
    save_json,
    set_seed,
    suppress_tqdm,
)

suppress_tqdm()

RATES = [0.01, 0.03, 0.05, 0.10]
CS, T_BOUND, MSG_LEN = 1, 10.0, 16
# Soft boundary strategy table: (hash_w, hash_e, hash_k).  (None,None,None) = no hash.
# hash_e=0 ⇒ whole window must vote yes (strictest); hash_e≥1 is the soft default regime.
COMBOS = [(None, None, None), (3, 1, 1), (3, 1, 3), (4, 2, 2), (5, 2, 2)]


def print_table(rows, rates):
    hdr = f"{'strategy':<10}|{'bits/token':<12}|{'entr-util':<12}|{'enc tok/s':<12}|{'dec tok/s':<12}|" + \
          "|".join(f"{f'correct@{r*100:.0f}%':<10}" for r in rates)
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        row = f"{r['label']:<10}|{r['bits_per_token']:<12.4f}|{r['entropy_util']*100:<12.1f}|" \
              f"{r['encode_tok_s']:<12.1f}|{r['decode_tok_s']:<12.1f}|"
        row += "|".join(f"{r['correct_rate'][str(x)]:<10.3f}" for x in rates)
        print(row)


def run(model_path, samples, seeds, rates, trials, device, output_dir,
        lang="en"):
    model, tokenizer = load_model(model_path, device)
    contexts = context_texts(lang)[:samples]
    out_dir = output_dir or os.path.join(RESULTS_DIR, "exp6")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = Checkpoint("exp6", out_dir, model_path)
    t0_all = time.time()

    rows = []
    for (hw, he, hk) in COMBOS:
        label = "no_hash" if hw is None else f"w{hw}_e{he}_k{hk}"
        print(f"\n===== {label} =====", flush=True)
        encs = []
        for ctx_idx, context in enumerate(contexts):
            for seed in seeds:
                key = ckpt.enc_key(label, ctx_idx, seed)
                if not ckpt.get_encode(key):
                    ml = MSG_LEN
                    secret = generate_secret_bits(ml, random.Random(seed))
                    set_seed(seed)
                    try:
                        enc = encode_catopter(
                            model=model, tokenizer=tokenizer, context=context,
                            secret_bits=secret, device=device, random_seed=seed,
                            chunk_size=CS, T_bound=T_BOUND,
                            use_hash_boundary=hw is not None, hash_w=hw or 3,
                            hash_e=he or 1, hash_k=hk or 1)
                    except Exception as e:
                        print(f"    [encode fail] {label} seed={seed}: {e}",
                              flush=True)
                        continue
                    if enc.get("n_embedded", 0) <= 0 or enc.get("n_tokens", 0) <= 0:
                        continue
                    enc["secret_bits"] = secret
                    ckpt.set_encode(key, enc)
                    ckpt.save()
                else:
                    enc = ckpt.get_encode(key)
                encs.append(enc)
        if not encs:
            continue

        dec_ts = []
        for enc in encs:
            fn = make_decoder("Catopter", enc, model, tokenizer, contexts[0],
                                device, 0.95, 42)
            try:
                _, dt = fn(list(enc["token_seq"]))
                dec_ts.append(dt)
            except Exception:
                dec_ts.append(0.0)

        per_rate = {r: [] for r in rates}
        for (ctx_idx, context) in enumerate(contexts):
            for seed in seeds:
                k = ckpt.enc_key(label, ctx_idx, seed)
                enc = ckpt.get_encode(k)
                if not enc:
                    continue
                fn = make_decoder("Catopter", enc, model, tokenizer, contexts[0],
                                    device, 0.95, 42)
                ref = enc["secret_bits"][: enc["n_embedded"]]
                for rate in rates:
                    ak = ckpt.atk_key(label, ctx_idx, seed, "edit", rate, trials)
                    if ckpt.get_attack(ak):
                        per_rate[rate] += ckpt.get_attack(ak)
                        continue
                    n = trials if rate > 0 else 1
                    vals = [evaluate_attack_trial(
                        enc["token_seq"], ref, tokenizer, "edit", rate, fn)["correct_rate"]
                        for _ in range(n)]
                    ckpt.set_attack(ak, vals)
                    ckpt.save()
                    per_rate[rate] += vals

        nt = np.mean([e["n_tokens"] for e in encs])
        nb = np.mean([e["n_embedded"] for e in encs])
        te = sum(sum(e["entropy_list"]) for e in encs)
        rows.append({
            "label": label, "hw": hw, "he": he, "hk": hk,
            "bits_per_token": nb / nt if nt > 0 else 0.0,
            "entropy_util": (nb * len(encs)) / te if te > 0 else 0.0,
            "encode_tok_s": nt / np.mean([e["encoding_time"] for e in encs])
            if encs else 0.0,
            "decode_tok_s": nt / np.mean(dec_ts) if np.mean(dec_ts) > 0 else 0.0,
            "correct_rate": {str(r): float(np.mean(per_rate[r])) for r in rates}})
        print(f"  bpt={rows[-1]['bits_per_token']:.3f} "
              f"enc={rows[-1]['encode_tok_s']:.1f} dec={rows[-1]['decode_tok_s']:.1f} "
              f"correct={ {str(r): round(rows[-1]['correct_rate'][str(r)], 3) for r in rates} }",
              flush=True)

    save_json(rows, os.path.join(out_dir, "exp6_rows.json"))        # persist first
    p = os.path.join(out_dir, "exp6_summary.json")
    save_json(rows, p)                                              # persist first
    print(f"\nSaved summary → {p}")

    print_table(rows, rates)

    del model, tokenizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"\n[exp6 done] {time.time()-t0_all:.1f}s", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description="exp6 — hash strategy table (Catopter)")
    common_args(p)
    p.add_argument("--rates", type=float, nargs="+", default=RATES)
    p.add_argument("--trials", type=int, default=20)
    a = p.parse_args()
    run(a.model_path, a.samples, a.seeds or default_seeds(), a.rates,
        a.trials, a.device, a.output_dir, a.lang)


if __name__ == "__main__":
    main()
