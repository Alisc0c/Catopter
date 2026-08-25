"""exp1 — token-level edit robustness (plan §1 exp1)"""

import argparse
import os
import random
import time

import numpy as np
import torch

from .common import (
    Checkpoint,
    FIGURES_DIR,
    METHODS,
    METHOD_ORDER,
    RESULTS_DIR,
    common_args,
    context_texts,
    default_seeds,
    evaluate_attack_trial,
    generate_secret_bits,
    load_model,
    make_decoder,
    plot_outcome_bars,
    save_json,
    set_seed,
    stats,
    suppress_tqdm,
)

suppress_tqdm()

# ══════════════════════════════════════════════════════════════════════
#  Experiment parameters — designed per experiment; CLI overrides these.
#  (Message length design: Catopter/ARS = 4 bits, others = 128 bits.)
# ══════════════════════════════════════════════════════════════════════
DEFAULT_RATES = [0.01, 0.03, 0.05, 0.07, 0.10]     # edit error rates
TRIALS = 20                                  # attack trials per record
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
    parser.add_argument("--prefix_len", type=int, default=CAT["prefix_len"])
    parser.add_argument("--sk", type=str, default=CAT["sk"])
    parser.add_argument("--top_k", type=int, default=CAT["top_k"])
    parser.add_argument("--max_n", type=int, default=CAT["max_n"])
    parser.add_argument("--top_p", type=float, default=CAT["top_p"])
    parser.add_argument("--error_bound", type=float, default=ARS["error_bound"])
    parser.add_argument("--ars_max_n", type=int, default=ARS["max_n"])
    return parser


def build_params(args):
    """Per-method encode kwargs from CLI (unlisted knobs keep design values)."""
    return {
        "Catopter": dict(chunk_size=args.chunk_size, T_bound=args.T_bound,
                         prefix_len=args.prefix_len, sk=args.sk,
                         top_k=args.top_k, max_n=args.max_n,
                         use_hash_boundary=CAT["use_hash_boundary"],
                         hash_w=args.hash_w, hash_e=args.hash_e, hash_k=args.hash_k,
                         top_p=args.top_p),
        "ARS": dict(error_bound=args.error_bound, max_n=args.ars_max_n),
        "SparSamp": dict(SPARSAMP),
        "AnStega": dict(ANSTEGA),
    }


# ── data generation (exp_a style: generate → store) ──────────────────────

def generate_data(model, tokenizer, method, contexts, seeds, device,
                  ckpt, params):
    """Encode one record per (context, seed), store in checkpoint."""
    print(f"\n--- Starting {method} Experiment ---")
    ml = ML[method]
    kw = dict(params[method])
    for ctx_idx, context in enumerate(contexts):
        for seed in seeds:
            key = ckpt.enc_key(method, ctx_idx, seed)
            if ckpt.get_encode(key):
                continue
            set_seed(seed)
            secret = generate_secret_bits(ml, random.Random(seed))
            kw2 = dict(kw, random_seed=seed)
            try:
                enc = METHODS[method]["encode"](
                    model=model, tokenizer=tokenizer, context=context,
                    secret_bits=secret, device=device, **kw2)
                if enc.get("n_embedded", 0) <= 0 or enc.get("n_tokens", 0) <= 0:
                    continue
                enc["secret_bits"] = secret
                ckpt.set_encode(key, enc)
                ckpt.save()          # persist before printing
            except Exception as e:
                print(f"    [encode fail] {method} ctx={ctx_idx} seed={seed}: {e}",
                      flush=True)
    encs = list(ckpt.encode_records([method], len(contexts), seeds))
    print(f"  encoded {len(encs)} records")
    return encs


# ── attack + decode (exp_a style: per attack_rate, aggregate) ────────────

def run_attacks(ckpt, method, encs, model, tokenizer, device, rate, trials,
                contexts):
    """One attack_rate over all records; returns per-trial metrics list."""
    recs = []
    for (ctx_idx, context) in enumerate(contexts):
        for seed, enc in ((s, e) for (m, ci, s, e) in encs if ci == ctx_idx):
            key = ckpt.atk_key(method, ctx_idx, seed, "edit", rate, trials)
            if ckpt.get_attack(key):
                recs += ckpt.get_attack(key)
                continue
            fn = make_decoder(method, enc, model, tokenizer, context,
                                device, 0.95, seed)
            ref = enc["secret_bits"][: enc["n_embedded"]]
            n = trials if rate > 0 else 1   # rate=0 is a pure roundtrip check
            rs = [evaluate_attack_trial(enc["token_seq"], ref, tokenizer,
                                          "edit", rate, fn)
                  for _ in range(n)]
            ckpt.set_attack(key, rs)
            ckpt.save()              # persist before printing
            recs += rs
    return recs


def aggregate(records):
    if not records:
        return None
    a = {k: np.array([r[k] for r in records])
         for k in ("correct", "wrong", "lost", "correct_rate", "success")}
    return {"n_embedded": float(np.mean([r["n_embedded"] for r in records])),
            **{k: stats(v) for k, v in a.items()},
            "success_rate": stats(a["success"])}


# ── main driver ───────────────────────────────────────────────────────────

def run(model_path, samples, seeds, rates, trials, device, output_dir,
        plot=False, lang="en", params=None):
    model, tokenizer = load_model(model_path, device)
    contexts = context_texts(lang)[:samples]
    out_dir = output_dir or os.path.join(RESULTS_DIR, "exp1")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = Checkpoint("exp1", out_dir, model_path)
    t0_all = time.time()

    for method in METHOD_ORDER:
        encs = generate_data(model, tokenizer, method, contexts, seeds,
                             device, ckpt, params)
        for rate in rates:
            recs = run_attacks(ckpt, method, encs, model, tokenizer, device,
                               rate, trials, contexts)
            agg = aggregate(recs)
            if agg:
                print(f"  rate={rate}: N_emb={agg['n_embedded']:.0f} "
                      f"correct={agg['correct_rate'][0]:.3f} "
                      f"succ={agg['success_rate'][0]:.3f} "
                      f"c/w/l={agg['correct'][0]:.0f}/"
                      f"{agg['wrong'][0]:.0f}/{agg['lost'][0]:.0f}", flush=True)

    # build & persist summary
    summary = build_summary(ckpt, len(contexts), seeds, rates, trials)
    p = os.path.join(out_dir, "exp1_summary.json")
    save_json(summary, p)          # persist first
    print(f"\nSaved summary → {p}")
    print_table(summary, rates)

    if plot:
        _plot(summary, rates, os.path.join(FIGURES_DIR, "exp1"))

    del model, tokenizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"\n[exp1 done] {time.time()-t0_all:.1f}s", flush=True)
    return summary


def plot(summary, rates, out_dir=None, show=True):
    figures_dir = out_dir or os.path.join(FIGURES_DIR, "exp1")
    _plot(summary, rates, figures_dir, show=show)
    return figures_dir


def build_summary(ckpt, ctx_count, seeds, rates, trials):
    s = {}
    for method in METHOD_ORDER:
        encs = list(ckpt.encode_records([method], ctx_count, seeds))
        s[method] = {
            "mean_n_tokens": float(np.mean([e["n_tokens"] for _, _, _, e in encs]))
            if encs else 0.0,
            "mean_n_embedded": float(np.mean([e["n_embedded"]
                                              for _, _, _, e in encs]))
            if encs else 0.0,
            "rates": {}}
        for rate in rates:
            rs = [r for _, _, _, recs in ckpt.attack_records(
                [method], ctx_count, seeds, "edit", rate, trials) for r in recs]
            agg = aggregate(rs)
            if agg:
                s[method]["rates"][str(rate)] = {k: agg[k] for k in
                                                 ("n_embedded", "correct",
                                                  "wrong", "lost", "correct_rate",
                                                  "success_rate")}
    return s


def print_table(summary, rates):
    hdr = "Method".ljust(10) + "|"
    for r in rates:
        hdr += (f"correct@{r*100:.0f}%").ljust(12) + "succ".ljust(8)
    print("\n" + hdr)
    print("-" * len(hdr))
    for m in METHOD_ORDER:
        s = summary.get(m, {})
        row = m.ljust(10) + "|"
        for r in rates:
            d = s.get("rates", {}).get(str(r), {})
            if d:
                row += (f"{d['correct_rate'][0]:.3f}").ljust(12) + \
                       (f"{d['success_rate'][0]:.3f}").ljust(8)
            else:
                row += "-".ljust(12) + "-".ljust(8)
        print(row)


def _plot(summary, rates, figures_dir, show=False):
    os.makedirs(figures_dir, exist_ok=True)
    for rate in rates:
        labels, emb, corr, wrong, lost, succ = [], [], [], [], [], []
        for m in METHOD_ORDER:
            d = summary[m]["rates"].get(str(rate))
            if not d:
                continue
            labels.append(m)
            emb.append(d["n_embedded"])
            corr.append(d["correct"][0])
            wrong.append(d["wrong"][0])
            lost.append(d["lost"][0])
            succ.append(d["success_rate"][0])

        # normalise to percentages (per method, correct+wrong+lost = 100%)
        corr = np.array(corr) / np.array(emb) * 100
        wrong = np.array(wrong) / np.array(emb) * 100
        lost = np.array(lost) / np.array(emb) * 100
        succ = np.array(succ) * 100

        plot_outcome_bars(labels, corr, wrong, lost, succ,
                                 figures_dir, f"edit_rate_{rate:.2f}", show=show)


def main():
    p = argparse.ArgumentParser(description="exp1 — token-level edit robustness")
    common_args(p)
    add_params(p)
    p.add_argument("--attack_rates", type=float, nargs="+", default=DEFAULT_RATES)
    p.add_argument("--trials", type=int, default=TRIALS)
    a = p.parse_args()
    params = build_params(a)
    run(a.model_path, a.samples, a.seeds or default_seeds(), a.attack_rates,
        a.trials, a.device, a.output_dir, a.plot, a.lang,
        params)


if __name__ == "__main__":
    main()
