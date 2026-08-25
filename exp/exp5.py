"""exp5 — Catopter capacity×robustness Pareto (cs=1 only, T swept).

Single chunk_size=1, sweep T_bound (with hash boundary). Each config is one
point: x = correct rate (%), y = bits/token (%). Five edit rates. Points are
colored by T_bound so the Pareto frontier (robustness vs capacity) is visible.
All per-trial data checkpointed.
"""

import argparse
import os
import random
import time

import numpy as np
import torch

from .common import (
    Checkpoint,
    FIGURES_DIR,
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
    set_mpl_style,
    set_seed,
    stats,
    suppress_tqdm,
)

suppress_tqdm()

# cs=1 only; T swept to find where m@0 is guaranteed WITHOUT the boundary
GRID = [dict(chunk_size=1, T_bound=T) for T in (3, 5, 10, 15, 20, 25, 30)]
HASH_W, HASH_E, HASH_K = 3, 1, 1
RATES = [0.0, 0.01, 0.05, 0.10, 0.20]
MSG_LEN = 1
TRIALS = 12
TOP_P = 1.0                    # sampling filter; 1.0 = full vocab (higher entropy)

# color ramp for T_bound (low→high capacity vs robustness tradeoff)
T_COLORS = {3: "#D55E00", 5: "#56B4E9", 10: "#E69F00", 15: "#009E73",
            20: "#CC79A7", 25: "#F0E442", 30: "#999999"}
T_MARKERS = {3: "X", 5: "o", 10: "s", 15: "^", 20: "D", 25: "v", 30: "P"}


def cfg_key(cfg, top_p, hash_k):
    return f"cs1_T{cfg['T_bound']}_p{top_p}_k{hash_k}"


def encode_point(model, tokenizer, cfg, context, seed, device, top_p, top_k=None,
                 hash_k=HASH_K, prefix_len=1, use_hash_boundary=True):
    secret = generate_secret_bits(MSG_LEN, random.Random(seed))
    set_seed(seed)
    random.seed(seed)
    enc = encode_catopter(
        model=model, tokenizer=tokenizer, context=context, secret_bits=secret,
        device=device, random_seed=seed, chunk_size=cfg["chunk_size"],
        T_bound=cfg["T_bound"], top_p=top_p, top_k=top_k, prefix_len=prefix_len,
        use_hash_boundary=use_hash_boundary, hash_w=HASH_W, hash_e=HASH_E,
        hash_k=hash_k)
    if enc["n_embedded"] <= 0 or enc["n_tokens"] <= 0:
        return None
    enc["secret_bits"] = secret
    return enc


def print_table(summary):
    hdr = f"{'T':<6}|{'bits/token':<12}|{'entr-util %':<14}|{'m@0':<8}|{'m@5%':<8}|{'m@20%':<8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for pt in summary["points"]:
        T = pt["config"]["T_bound"]
        print(f"{T:<6}|{pt['capacity'][0]:<12.3f}|{pt['entropy_util'][0]*100:<14.1f}|"
              f"{pt['0.0'][0]:<8.3f}|{pt['0.05'][0]:<8.3f}|{pt['0.2'][0]:<8.3f}")


def run(model_path, samples, seeds, trials, device, output_dir,
        make_plot=False, lang="en", top_p=TOP_P,
        top_k=None, hash_k=HASH_K, prefix_len=1, use_hash_boundary=True):
    model, tokenizer = load_model(model_path, device)
    contexts = context_texts(lang)[:samples]
    out_dir = output_dir or os.path.join(RESULTS_DIR, "exp5")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = Checkpoint("exp5", out_dir, model_path)
    t0_all = time.time()

    # ── encode phase (per config × context × seed) ──
    for cfg in GRID:
        for ctx_idx, context in enumerate(contexts):
            for seed in seeds:
                key = ckpt.enc_key(cfg_key(cfg, top_p, hash_k), ctx_idx, seed)
                if ckpt.get_encode(key):
                    continue
                enc = encode_point(model, tokenizer, cfg, context, seed, device,
                                   top_p, top_k, hash_k, prefix_len,
                                   use_hash_boundary)
                if enc:
                    ckpt.set_encode(key, enc)
                    ckpt.save()

    # ── attack phase (per config × rate, checkpointed) ──
    for cfg in GRID:
        for ctx_idx, context in enumerate(contexts):
            for seed in seeds:
                ekey = ckpt.enc_key(cfg_key(cfg, top_p, hash_k), ctx_idx, seed)
                enc = ckpt.get_encode(ekey)
                if not enc:
                    continue
                for rate in RATES:
                    ak = ckpt.atk_key(cfg_key(cfg, top_p, hash_k), ctx_idx, seed,
                                      "edit", rate, trials)
                    if ckpt.get_attack(ak):
                        continue
                    fn = make_decoder("Catopter", enc, model, tokenizer, context,
                                      device, top_p, seed)
                    ref = enc["secret_bits"][: enc["n_embedded"]]
                    n = trials if rate > 0 else 1
                    rs = [evaluate_attack_trial(enc["token_seq"], ref, tokenizer,
                                                "edit", rate, fn) for _ in range(n)]
                    ckpt.set_attack(ak, rs)
                    ckpt.save()

    # ── aggregate ──
    summary = {"points": [], "rates": RATES}
    for cfg in GRID:
        key = cfg_key(cfg, top_p, hash_k)
        encs = list(ckpt.encode_records([key], len(contexts), seeds))
        if not encs:
            continue
        # capacity = bits/token; entropy utilisation = bits / total entropy
        bpt = [e["n_embedded"] / e["n_tokens"] for _, _, _, e in encs]
        ent_util = [e["n_embedded"] / max(1e-9, sum(e["entropy_list"]))
                    for _, _, _, e in encs]
        point = {"config": cfg, "capacity": stats(bpt),
                 "entropy_util": stats(ent_util)}
        for rate in RATES:
            rs = [r for _, _, _, recs in ckpt.attack_records(
                [key], len(contexts), seeds, "edit", rate, trials) for r in recs]
            point[str(rate)] = stats([r["correct_rate"] for r in rs])
        summary["points"].append(point)
        print(f"  {key}: cap={point['capacity'][0]:.3f} "
              f"eu={point['entropy_util'][0]*100:.1f}% "
              f"m@0={point['0.0'][0]:.3f} m@5%={point['0.05'][0]:.3f} "
              f"m@20%={point['0.2'][0]:.3f}", flush=True)

    p = os.path.join(out_dir, "exp5_summary.json")
    save_json(summary, p)          # persist first
    print(f"\nSaved summary → {p}")

    if make_plot:
        plot(summary)

    del model, tokenizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"\n[exp5 done] {time.time()-t0_all:.1f}s", flush=True)
    return summary


def plot(summary, out_dir=None, show=True):
    """Scatter per edit rate: x = correct rate(%), y = bits/token."""
    figures_dir = out_dir or os.path.join(FIGURES_DIR, "exp5")
    os.makedirs(figures_dir, exist_ok=True)
    plt = set_mpl_style()
    for rate in summary["rates"]:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        for pt in summary["points"]:
            cfg = pt["config"]
            T = cfg["T_bound"]
            x = pt[str(rate)]
            y = pt["capacity"]
            if any(np.isnan(v) for v in (x[0], y[0])):
                continue
            xerr = [[(x[0] - x[1]) * 100], [(x[2] - x[0]) * 100]]
            yerr = [[(y[0] - y[1])], [(y[2] - y[0])]]
            ax.errorbar(x[0] * 100, y[0], xerr=xerr, yerr=yerr,
                        fmt=T_MARKERS.get(T, "o"),
                        color=T_COLORS.get(T, "#333333"),
                        ms=10, alpha=0.95, capsize=3, capthick=1.0,
                        elinewidth=1.0, markeredgecolor="white",
                        markeredgewidth=0.6, zorder=5)
        ax.set_xlabel("Correct Rate (%)", fontsize=13)
        ax.set_ylabel("Bits per Token", fontsize=13)
        ax.set_xlim(-2, 102)
        ax.set_ylim(0, None)
        ax.tick_params(labelsize=11)
        # clean legend: marker-only handles (no error bars)
        handles = [plt.Line2D([], [], color=T_COLORS[T],
                              marker=T_MARKERS[T], linestyle="none",
                              ms=10, markeredgecolor="white",
                              markeredgewidth=0.5, label=f"T={T}")
                   for T in sorted(T_COLORS)]
        ax.legend(handles=handles, loc="lower left",
                  fontsize=11, frameon=False)
        f = f"pareto_edit_{rate:.2f}"
        fig.savefig(os.path.join(figures_dir, f + ".png"))
        fig.savefig(os.path.join(figures_dir, f + ".pdf"))
        if not show:
            plt.close(fig)
        print(f"  figure {f}", flush=True)
    return figures_dir


def main():
    p = argparse.ArgumentParser(description="exp5 — Catopter Pareto (cs=1, T sweep)")
    common_args(p)
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--top_p", type=float, default=TOP_P,
                   help="sampling filter; 1.0 = full vocab (higher entropy)")
    p.add_argument("--top_k", type=int, default=None,
                   help="hard top-k truncation (None = p-threshold filter)")
    p.add_argument("--hash_k", type=int, default=HASH_K,
                   help="boundary marker rarity: 2^-hash_k per-token vote")
    p.add_argument("--prefix_len", type=int, default=1,
                   help="PRF prefix context length")
    p.add_argument("--use_hash_boundary", type=int, default=1, choices=(0, 1),
                   help="enable/disable the hash boundary marker")
    a = p.parse_args()
    run(a.model_path, a.samples, a.seeds or default_seeds(), a.trials,
        a.device, a.output_dir, a.plot, a.lang, a.top_p,
        a.top_k, a.hash_k, a.prefix_len, bool(a.use_hash_boundary))


if __name__ == "__main__":
    main()
