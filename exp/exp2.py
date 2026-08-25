"""exp2 — robustness under real-world channel perturbations (plan §1 exp2)"""

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

DEFAULT_SCENES = ["homoglyph", "invisible_char", "social_platform",
                  "mobile_im", "document_reformat"]
# per-word scenes fixed at 5% so all five scenarios align (one figure each);
# one-shot channel scenes apply once.
PER_WORD_RATE = [0.05]
ONE_SHOT_RATE = [0.0]
RATE_SEMANTICS = {"homoglyph": "per-word", "invisible_char": "per-word",
                  "social_platform": "one-shot", "mobile_im": "one-shot",
                  "document_reformat": "one-shot"}

# ══════════════════════════════════════════════════════════════════════
#  Experiment parameters — designed per experiment; CLI overrides these.
#  (Message length design: Catopter/ARS = 4 bits, others = 128 bits.)
# ══════════════════════════════════════════════════════════════════════
TRIALS = 20
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


def scene_rates(scene, rates=None):
    return ONE_SHOT_RATE if RATE_SEMANTICS[scene] == "one-shot" \
        else (rates if rates else PER_WORD_RATE)


def encode_one(method, model, tokenizer, context, seed, device, params=None):
    ml = ML[method]
    kw = dict(params[method], random_seed=seed)
    secret = generate_secret_bits(ml, random.Random(seed))
    try:
        enc = METHODS[method]["encode"](model=model, tokenizer=tokenizer,
                                          context=context, secret_bits=secret,
                                          device=device, **kw)
        if enc.get("n_embedded", 0) <= 0 or enc.get("n_tokens", 0) <= 0:
            return None
        enc["secret_bits"] = secret
        return enc
    except Exception as e:
        print(f"    [encode fail] {method} seed={seed}: {e}", flush=True)
        return None


def _agg(records):
    if not records:
        return None
    a = {k: np.array([r[k] for r in records])
         for k in ("correct", "wrong", "lost", "correct_rate", "success")}
    return {"n_embedded": float(np.mean([r["n_embedded"] for r in records])),
            **{k: stats(v) for k, v in a.items()},
            "success_rate": stats(a["success"])}


def print_table(summary, scenes=DEFAULT_SCENES):
    for scene in scenes:
        rows = []
        for m in METHOD_ORDER:
            d = summary.get(m, {}).get(scene, {})
            if not d:
                continue
            r0 = list(d.values())[0]
            rows.append((m, r0["n_embedded"], r0["correct_rate"][0],
                         r0["success_rate"][0]))
        if not rows:
            continue
        hdr = f"{'Method':<10}|{'N_emb':<8}|{'correct':<10}|{'succ':<10}"
        print(f"\n=== {scene} ===")
        print(hdr)
        print("-" * len(hdr))
        for m, n, mt, sc in rows:
            print(f"{m:<10}|{n:<8.0f}|{mt:<10.3f}|{sc:<10.3f}")


def run(model_path, samples, seeds, scenes, rates, trials, device, output_dir,
        plot=False, lang="en", params=None, methods=None):
    model, tokenizer = load_model(model_path, device)
    contexts = context_texts(lang)[:samples]
    out_dir = output_dir or os.path.join(RESULTS_DIR, "exp2")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = Checkpoint("exp2", out_dir, model_path)
    t0_all = time.time()
    methods = [m for m in METHOD_ORDER if methods is None or m in methods]

    for method in methods:
        print(f"\n===== {method} =====", flush=True)
        for ctx_idx, context in enumerate(contexts):
            for seed in seeds:
                k = ckpt.enc_key(method, ctx_idx, seed)
                if not ckpt.get_encode(k):
                    set_seed(seed)
                    enc = encode_one(method, model, tokenizer, context, seed,
                                     device, params)
                    if enc:
                        ckpt.set_encode(k, enc)
                        ckpt.save()

        for scene in scenes:
            # one-shot channel attacks ignore rate (single pass); per-word
            # attacks keep the rate sweep.
            rates_for_scene = scene_rates(scene)
            for rate in rates_for_scene:
                recs = []
                for ctx_idx, context in enumerate(contexts):
                    for seed in seeds:
                        k = ckpt.enc_key(method, ctx_idx, seed)
                        enc = ckpt.get_encode(k)
                        if not enc:
                            continue
                        ak = ckpt.atk_key(method, ctx_idx, seed, scene, rate,
                                          trials)
                        if ckpt.get_attack(ak):
                            recs += ckpt.get_attack(ak)
                            continue
                        fn = make_decoder(method, enc, model, tokenizer,
                                            context, device, 0.95, seed)
                        ref = enc["secret_bits"][: enc["n_embedded"]]
                        n = trials if rate > 0 else 1
                        rs = [evaluate_attack_trial(enc["token_seq"], ref,
                                                      tokenizer, scene, rate,
                                                      fn) for _ in range(n)]
                        ckpt.set_attack(ak, rs)
                        ckpt.save()               # persist before printing
                        recs += rs
                agg = _agg(recs)
                if agg:
                    print(f"  {scene}@{rate}: N={agg['n_embedded']:.0f} "
                          f"correct={agg['correct_rate'][0]:.3f} "
                          f"succ={agg['success_rate'][0]:.3f}", flush=True)

    summary = {}
    for method in methods:
        summary[method] = {}
        for scene in scenes:
            summary[method][scene] = {}
            rates_for_scene = scene_rates(scene)
            for rate in rates_for_scene:
                r = [r for _, _, _, rs in ckpt.attack_records(
                    [method], len(contexts), seeds, scene, rate, trials)
                     for r in rs]
                a = _agg(r)
                if a:
                    summary[method][scene][str(rate)] = a
    p = os.path.join(out_dir, "exp2_summary.json")
    save_json(summary, p)                       # persist before printing
    print(f"\nSaved summary → {p}")

    if plot:
        _plot(summary, scenes, rates, os.path.join(FIGURES_DIR, "exp2"))

    del model, tokenizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"\n[exp2 done] {time.time()-t0_all:.1f}s", flush=True)
    return summary


def plot(summary, scenes, rates, out_dir=None, show=True):
    figures_dir = out_dir or os.path.join(FIGURES_DIR, "exp2")
    _plot(summary, scenes, rates, figures_dir, show=show)
    return figures_dir


def _plot(summary, scenes, rates, figures_dir, show=False):
    os.makedirs(figures_dir, exist_ok=True)
    for scene in scenes:
        rates_for_scene = scene_rates(scene)
        for rate in rates_for_scene:
            labels, emb, corr, wrong, lost, succ = [], [], [], [], [], []
            for m in METHOD_ORDER:
                d = summary.get(m, {}).get(scene, {}).get(str(rate))
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
                                     figures_dir, f"{scene}_{rate:.2f}", show=show)


def main():
    p = argparse.ArgumentParser(description="exp2 — real-world scene attacks")
    common_args(p)
    add_params(p)
    p.add_argument("--scenes", type=str, nargs="+", default=DEFAULT_SCENES,
                   choices=DEFAULT_SCENES)
    p.add_argument("--rates", type=float, nargs="+", default=PER_WORD_RATE)
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--methods", type=str, nargs="+", default=None,
                   choices=METHOD_ORDER,
                   help="subset of methods to run (default: all)")
    a = p.parse_args()
    params = build_params(a)
    run(a.model_path, a.samples, a.seeds or default_seeds(), a.scenes,
        a.rates, a.trials, a.device, a.output_dir, a.plot,
        a.lang, params, a.methods)


if __name__ == "__main__":
    main()
