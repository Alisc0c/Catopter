"""Shared helpers for all exp experiments (metrics, IO, methods, wrappers)."""

import hashlib
import json
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, is_dataclass

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXP_DIR)
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(EXP_DIR, "data")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
FIGURES_DIR = os.path.join(EXP_DIR, "figures")
for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "llm", "Qwen3-0.6B")


# ── tqdm suppression (library code prints progress bars) ────────────────

def suppress_tqdm():
    """Silence tqdm progress bars.  Subclass the real class so every class
    method (format_interval, set_lock, ...) still exists — this keeps
    transformers' lazy imports and huggingface_hub's tqdm wrapper working."""
    try:
        from tqdm import tqdm as _RealTqdm
        import tqdm.auto, tqdm.std, tqdm.autonotebook  # noqa: F401
    except ImportError:
        return

    class _Silent(_RealTqdm):
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
            self.n = 0
            self.total = kwargs.get("total", None)

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter([])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def update(self, n=1):
            self.n += n

        def close(self):
            pass

        def refresh(self, *a, **k):
            pass

        def display(self, *a, **k):
            pass

        def set_description(self, *a, **k):
            pass

        def set_postfix(self, *a, **k):
            pass

        def __len__(self):
            return self.n if self.total is None else self.total

    for modname in ("tqdm", "tqdm.auto", "tqdm.std", "tqdm.autonotebook",
                    "tqdm.notebook"):
        m = sys.modules.get(modname)
        if m is not None:
            m.tqdm = _Silent


# ── Basics ───────────────────────────────────────────────────────────────

def get_device(prefer="cuda"):
    return "cpu" if prefer == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model(model_path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = dict(trust_remote_code=True, local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto",
                                                     **kw).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kw)
    except Exception as e:
        warnings.warn(f"load {model_path} failed ({e}); fallback gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2", **kw).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "openai-community/gpt2", **kw)
    model.eval()
    return model, tokenizer


def vocab_size(model, tokenizer=None):
    """Robust vocab size across architectures (Gemma4 stores it in a
    sub-config and has no model.vocab_size)."""
    try:
        return int(model.config.vocab_size)
    except AttributeError:
        pass
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for name in ("text_config", "language_config", "decoder"):
            sub = getattr(cfg, name, None)
            if sub is not None and hasattr(sub, "vocab_size"):
                return int(sub.vocab_size)
    if tokenizer is not None:
        return int(tokenizer.vocab_size)
    raise AttributeError("cannot determine vocab size")


def process_context(tokenizer, context):
    """Tokenize context (chat template first, plain encode as fallback).

    ``enable_thinking=False``: on Qwen-family models this keeps the model in
    pure-text answer mode — measured 281 vs 359 tokens for 32 bits, and the
    output carries no <think> block.  Default thinking mode crushes entropy.
    """
    try:
        tmpl = [{"role": "user", "content": context}]
        s = tokenizer.apply_chat_template(tmpl, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
        return tokenizer(s, return_tensors="pt").input_ids.squeeze().numpy()
    except Exception:
        return tokenizer.encode(context, return_tensors="pt",
                                add_special_tokens=False).squeeze().numpy()


def process_message(msg):
    return "".join(c for c in msg if c in "01")


def generate_secret_bits(num, rng=None):
    rng = rng if rng is not None else random
    return "".join(rng.choice("01") for _ in range(num))


# ── Bit metrics (reference length = N_embedded) ──────────────────────────

def _to_bits(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return "".join(str(int(b)) for b in x)
    return str(x)


def compare_bits(original, reconstructed):
    """Classify bits: correct/wrong/lost vs reference N (extra recon ignored)."""
    orig, recon = _to_bits(original), _to_bits(reconstructed)
    if not orig:
        return 0, 0, 0
    correct = sum(1 for a, b in zip(orig, recon) if a == b)
    compared = min(len(orig), len(recon))
    return correct, compared - correct, len(orig) - compared


def compute_correct_rate(original, reconstructed):
    c, _, _ = compare_bits(original, reconstructed)
    n = len(_to_bits(original))
    return c / n if n else 0.0


def decode_success(original, reconstructed):
    orig, recon = _to_bits(original), _to_bits(reconstructed)
    return bool(orig) and recon[: len(orig)] == orig


def nats_to_bits(x):
    return float(x) / np.log(2)


def bootstrap_ci(values, n_boot=4000, alpha=0.05, seed=1234):
    """mean ± bootstrap 95% CI → (mean, (lo, hi)), deterministic."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    if v.size == 1:
        return float(v[0]), (float(v[0]), float(v[0]))
    rng = np.random.RandomState(seed)
    means = np.array([rng.choice(v, size=v.size, replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), (float(lo), float(hi))


# ── Persistence ───────────────────────────────────────────────────────────

class _JsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if is_dataclass(o):
            return asdict(o)
        return str(o)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=_JsonEncoder, indent=2, ensure_ascii=False)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stats(vals):
    """mean + bootstrap CI → [mean, lo, hi]."""
    m, (lo, hi) = bootstrap_ci(np.asarray(vals, dtype=float))
    return [m, lo, hi]


# ── Incremental checkpoint (persist-then-print; skip done, resume) ────────

class Checkpoint:
    def __init__(self, exp, out_dir, model_path):
        self.exp = exp
        self.out_dir = out_dir
        self.model = os.path.basename(os.path.normpath(model_path))
        self.path = os.path.join(out_dir, f"{exp}_checkpoint.json")
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                d = load_json(self.path)
                if "encode" in d and "attack" in d:
                    return d
            except Exception:
                pass
        return {"encode": {}, "attack": {}}

    def save(self):
        save_json(self.data, self.path)

    @staticmethod
    def _k(*parts):
        return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:16]

    def enc_key(self, method, ctx_idx, seed):
        return self._k("enc", method, ctx_idx, seed, self.model)

    def atk_key(self, method, ctx_idx, seed, attack, rate, trials):
        return self._k("atk", method, ctx_idx, seed, attack, rate, trials)

    def get_encode(self, k):
        return self.data["encode"].get(k)

    def set_encode(self, k, rec):
        self.data["encode"][k] = rec

    def get_attack(self, k):
        return self.data["attack"].get(k)

    def set_attack(self, k, recs):
        self.data["attack"][k] = recs

    def encode_records(self, methods, ctx_count, seeds):
        for m in methods:
            for ci in range(ctx_count):
                for seed in seeds:
                    rec = self.get_encode(self.enc_key(m, ci, seed))
                    if rec is not None:
                        yield m, ci, seed, rec

    def attack_records(self, methods, ctx_count, seeds, attack, rate, trials):
        for m in methods:
            for ci in range(ctx_count):
                for seed in seeds:
                    recs = self.get_attack(self.atk_key(m, ci, seed, attack,
                                                        rate, trials))
                    if recs is not None:
                        yield m, ci, seed, recs


# ── Method registry (after all encode/decode defs) ────────────────────────
# ml = message bits offered (≈100-token budget); n_embedded is the true count.

# Experiment parameters live in each exp file (top of file, overridable via
# CLI).  common.py keeps only shared wrappers, metrics and IO.


def context_texts(lang="en", limit=None):
    """Load contexts from exp/data/prompts.json (fallback: 5 built-ins)."""
    p = os.path.join(EXP_DIR, "data", "prompts.json")
    if os.path.exists(p):
        import json as _json
        with open(p, encoding="utf-8") as f:
            prompts = _json.load(f)
        texts = prompts.get(lang) or prompts.get("en") or []
    else:
        texts = [
            "Can you tell me a long story about a beautiful sunrise over the ocean, where the light slowly spreads across the waves and the world wakes up?",
            "Write a short essay explaining why biodiversity matters for the future of the planet and how conservation efforts can make a real difference.",
            "Imagine a detective investigating a mysterious theft in a quiet lakeside town, where every clue seems to contradict the last one.",
            "Describe the process of brewing traditional tea, from selecting the leaves to the final warm cup, with attention to every small detail.",
            "What would a day in the life of a space station botanist look like, growing fresh vegetables in zero gravity far from Earth?",
        ]
    return texts[:limit] if limit else texts


# ── encode/decode wrappers for every method ───────────────────────────────

def _record(token_seq, n_embedded, nll, ent, enc_time, ok=True):
    return {"ok": ok, "token_seq": list(token_seq), "n_embedded": int(n_embedded),
            "n_tokens": len(token_seq), "nll_list": list(nll),
            # every method already reports per-token Shannon entropy in BITS
            # (np.log2 / scipy base=2 / torch.log2); do NOT re-convert.
            "entropy_list": [float(e) for e in ent],
            "encoding_time": enc_time}


def encode_catopter(model, tokenizer, context, secret_bits, device, top_p=0.95,
                    random_seed=42, chunk_size=2, T_bound=10.0, prefix_len=1,
                    sk="42", top_k=None, max_n=1024, use_hash_boundary=True,
                    hash_w=3, hash_e=1, hash_k=1):
    from catopter.catopter import encode
    ctx = process_context(tokenizer, context).tolist()
    set_seed(random_seed)
    random.seed(random_seed)
    bits = [int(b) for b in process_message(secret_bits)]
    t0 = time.time()
    tok, remain, nll, ent = encode(model=model, history=ctx, secret_bits=bits,
                                   chunk_size=chunk_size, T_bound=T_bound,
                                   top_p=top_p, top_k=top_k, sk=sk, max_n=max_n,
                                   prefix_len=prefix_len, device=device,
                                   use_hash_boundary=use_hash_boundary,
                                   hash_w=hash_w, hash_k=hash_k, hash_e=hash_e)
    rec = _record(tok, len(bits) - remain, nll, ent, time.time() - t0)
    rec["enc_kwargs"] = dict(chunk_size=chunk_size, T_bound=T_bound,
                             prefix_len=prefix_len, sk=sk,
                             use_hash_boundary=use_hash_boundary,
                             hash_w=hash_w, hash_e=hash_e, hash_k=hash_k)
    return rec


def decode_catopter(token_seq, num_bits, chunk_size=2, T_bound=10.0, sk="42",
                    prefix_len=1, force_output=False, use_hash_boundary=True,
                    hash_w=3, hash_e=1, hash_k=1):
    from catopter.catopter import decode
    t0 = time.time()
    out = decode(tokens=list(token_seq), num_bits=int(num_bits),
                 chunk_size=chunk_size, T_bound=T_bound, sk=sk,
                 prefix_len=prefix_len, force_output=force_output,
                 use_hash_boundary=use_hash_boundary, hash_w=hash_w,
                 hash_k=hash_k, hash_e=hash_e)
    bits = "".join(str(int(b)) for b in out[0])
    return {"bits": bits, "n_decoded": len(bits), "decoding_time": time.time() - t0}


def encode_ars(model, tokenizer, context, secret_bits, device, random_seed=42,
               error_bound=0.2, max_n=512):
    """ARS has no top-p (its encode() takes error_bound only)."""
    from others import ars_encode
    ctx = process_context(tokenizer, context).tolist()
    set_seed(random_seed)
    random.seed(random_seed)
    bits = [int(b) for b in process_message(secret_bits)]
    t0 = time.time()
    tok, remain, nll, ent = ars_encode(model=model, history=ctx,
                                       secret_bits=bits, error_bound=error_bound,
                                       random_seed=random_seed, max_n=max_n,
                                       device=device)
    rec = _record(tok, len(bits) - remain, nll, ent, time.time() - t0)
    rec["enc_kwargs"] = dict(error_bound=error_bound, max_n=max_n)
    return rec


def decode_ars(token_seq, vocab_size, error_bound=0.2, random_seed=42):
    from others import ars_decode
    t0 = time.time()
    out = ars_decode(token_sequence=list(token_seq), vocab_size=int(vocab_size),
                     error_bound=error_bound, random_seed=random_seed)
    return {"bits": "".join(str(int(b)) for b in out), "n_decoded": len(out),
            "decoding_time": time.time() - t0}


def encode_sparsamp(model, tokenizer, context, secret_bits, device,
                    top_p=1.0, random_seed=42, block_size=32):
    """Zero-pad message to a block_size multiple; n_embedded = full blocks."""
    from others import sparsamp_encode
    msg = process_message(secret_bits)
    set_seed(random_seed)
    random.seed(random_seed)
    padded = msg + "0" * ((-len(msg)) % block_size)
    ctx = torch.tensor(process_context(tokenizer, context), device=device, dtype=torch.long).unsqueeze(0)
    t0 = time.time()
    gids, enc_msg, ent, nll, _s, _m = sparsamp_encode(
        model=model, context=ctx, message_bits=padded, device=device,
        block_size=block_size, top_p=top_p, random_seed=random_seed)
    rec = _record(gids, len(enc_msg) * block_size, nll, ent, time.time() - t0)
    rec["enc_kwargs"] = dict(block_size=block_size)
    return rec


def decode_sparsamp(model, tokenizer, token_seq, context, device,
                    random_seed=42, block_size=32, top_p=1.0):
    from others import sparsamp_decode
    ctx = torch.tensor(process_context(tokenizer, context), device=device, dtype=torch.long).unsqueeze(0)
    random.seed(random_seed)
    t0 = time.time()
    try:
        blocks = sparsamp_decode(model=model, generated_ids=list(token_seq),
                                 context=ctx, device=device, block_size=block_size,
                                 top_p=top_p, random_seed=random_seed)
    except Exception:
        # attacked token outside the top-p set → whole trial lost
        blocks = []
    bits = "".join(blocks)
    return {"bits": bits, "n_decoded": len(bits),
            "decoding_time": time.time() - t0}


def _context_tensor(tokenizer, context, device):
    ids = process_context(tokenizer, context)
    return torch.from_numpy(ids).to(device).unsqueeze(0)


def encode_anstega(model, tokenizer, context, secret_bits, device,
                   top_p=0.95, random_seed=42, token_num_need_generated=100,
                   precision=32, ans_t=16):
    """n_embedded = message prefix consumed (init-state bit + per-token chunks)."""
    from others import anstega_encode
    msg = process_message(secret_bits)
    ctx = _context_tensor(tokenizer, context, device)
    set_seed(random_seed)
    random.seed(random_seed)
    t0 = time.time()
    gids, fs, chunks, te, _s, _m, nll, pulls = anstega_encode(
        model=model, context=ctx, message=msg,
        token_num_need_generated=token_num_need_generated, device=device,
        precision=precision, ans_t=ans_t, top_p=top_p, seed=random_seed)
    n_emb = min(len("".join(chunks)) + pulls, len(msg))
    rec = _record(gids, n_emb, nll, [te] if te else [], time.time() - t0)
    rec["final_state"] = int(fs)
    rec["initial_pulls"] = int(pulls)
    rec["enc_kwargs"] = dict(precision=precision, ans_t=ans_t)
    return rec


def decode_anstega(model, tokenizer, token_seq, context, device, final_state,
                   initial_pulls, random_seed=42, precision=32, ans_t=16,
                   top_p=0.95):
    from others import anstega_decode
    ctx = _context_tensor(tokenizer, context, device)
    random.seed(random_seed)
    t0 = time.time()
    try:
        bits = anstega_decode(model=model, context=ctx, generated_ids=list(token_seq),
                              final_state=final_state, device=device,
                              precision=precision, ans_t=ans_t, top_p=top_p,
                              seed=random_seed, initial_pulls=initial_pulls)
    except Exception:
        # attacked token outside the top-p set → whole trial lost
        bits = ""
    return {"bits": bits, "n_decoded": len(bits),
            "decoding_time": time.time() - t0}


METHODS = {
    "Catopter": dict(key="Catopter", encode=encode_catopter, decode=decode_catopter),
    "SparSamp": dict(key="SparSamp", encode=encode_sparsamp, decode=decode_sparsamp),
    "AnStega": dict(key="AnStega", encode=encode_anstega, decode=decode_anstega),
    "ARS": dict(key="ARS", encode=encode_ars, decode=decode_ars),
}
METHOD_ORDER = ["SparSamp", "AnStega", "ARS", "Catopter"]


def make_decoder(method, enc, model, tokenizer, context, device, top_p, seed):
    """Return fn(attacked_tokens) -> (bits, decode_time).

    Decode params come from the record's own ``enc_kwargs`` (each exp /
    ablation point embeds its own params at encode time).
    """
    d = enc["enc_kwargs"]
    if method == "Catopter":
        def fn(tok):
            out = decode_catopter(tok, enc["n_embedded"], d["chunk_size"],
                                  d["T_bound"], d["sk"], d["prefix_len"],
                                  use_hash_boundary=d["use_hash_boundary"],
                                  hash_w=d["hash_w"], hash_e=d["hash_e"],
                                  hash_k=d["hash_k"])
            return out["bits"], out["decoding_time"]
    elif method == "ARS":
        vs = vocab_size(model)

        def fn(tok):
            out = decode_ars(tok, vs, d["error_bound"], seed)
            return out["bits"], out["decoding_time"]
    elif method == "SparSamp":
        def fn(tok):
            # top_p must match encode's default 1.0 (top-p truncation would
            # drop the stego token → whole trial lost)
            out = decode_sparsamp(model, tokenizer, tok, context, device, seed,
                                  d["block_size"], 1.0)
            return out["bits"], out["decoding_time"]
    elif method == "AnStega":
        n_emb = enc["n_embedded"]

        def fn(tok):
            out = decode_anstega(model, tokenizer, tok, context, device,
                                 enc["final_state"], enc["initial_pulls"],
                                 seed, d["precision"], d["ans_t"], top_p)
            # ANStega's decode returns [state prefix] + [message]; the embedded
            # message occupies the LAST n_embedded bits.
            bits = out["bits"][-n_emb:] if len(out["bits"]) >= n_emb else out["bits"]
            return bits, out["decoding_time"]
    else:
        raise KeyError(method)
    return fn


def evaluate_attack_trial(token_seq, ref_bits, tokenizer, attack_type, rate,
                          decode_fn):
    """One attack→decode trial; decode_fn error → whole trial lost."""
    from catopter.edits import apply_attack
    try:
        attacked = apply_attack(token_list=list(token_seq), attack_type=attack_type,
                                rate=rate, tokenizer=tokenizer)
    except Exception:
        attacked = list(token_seq)
    try:
        recon, dt = decode_fn(attacked)
    except Exception:
        recon, dt = "", 0.0
    ref = _to_bits(ref_bits)
    c, w, l = compare_bits(ref, recon)
    return {"n_embedded": len(ref), "correct": float(c), "wrong": float(w),
            "lost": float(l), "correct_rate": float(compute_correct_rate(ref, recon)),
            "success": float(decode_success(ref, recon)), "decoding_time": float(dt)}


# ── Plotting ──────────────────────────────────────────────────────────────

CATEGORICAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
               "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
               "violet": "#4a3aa7", "red": "#e34948"}
SERIES = [CATEGORICAL[c] for c in
          ("blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red")]


def set_mpl_style():
    import matplotlib
    try:
        import IPython
        _ip = IPython.get_ipython()
        if _ip is not None:          # running in a notebook → inline display
            matplotlib.use("module://matplotlib_inline.backend_inline")
        else:
            matplotlib.use("Agg")
    except Exception:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "font.size": 12,
                         "axes.titlesize": 14, "axes.labelsize": 13,
                         "axes.labelweight": "bold", "legend.fontsize": 11,
                         "axes.spines.top": True, "axes.spines.right": True,
                         "axes.linewidth": 0.8, "axes.grid": True,
                         "grid.color": "#dcdcdc", "grid.linestyle": "--",
                         "grid.alpha": 0.6, "grid.linewidth": 0.6})
    return plt


def plot_outcome_bars(labels, corr, wrong, lost, succ, figures_dir, fname, show=False):
    """Shared stacked-bar (Correct/Wrong/Lost) + twinx Success plot.

    Used by exp1 (per edit rate) and exp2 (per scene) so the two robustness
    figures stay visually identical.  ``corr/wrong/lost`` are per-method
    percentages (sum to 100); ``succ`` is per-method success rate (0–100).
    """
    plt = set_mpl_style()
    outcome = [("Correct", "#56B4E9"), ("Wrong", "#E69F00"),
               ("Lost", "#C8CDD0")]
    success_color = "#CC79A7"

    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.18, top=0.86)
    bottom = np.zeros(len(labels))
    for (lab, col), vals in zip(outcome, (corr, wrong, lost)):
        ax.bar(range(len(labels)), vals, bottom=bottom, label=lab,
               color=col, width=0.6, edgecolor="white", linewidth=1.0,
               zorder=3)
        bottom += vals
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=13)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="y", zorder=0)

    ax2 = ax.twinx()
    ax2.plot(np.arange(len(labels)), succ, "o-", color=success_color,
             ms=6, lw=2, markeredgecolor="white", markeredgewidth=0.6,
             zorder=4, label="Success")
    ax2.set_ylabel("Success Rate (%)", color=success_color, fontsize=13)
    ax2.tick_params(axis="y", labelcolor=success_color, labelsize=11)
    ax2.set_ylim(0, 100)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center",
              bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False, fontsize=10)
    fig.savefig(os.path.join(figures_dir, fname + ".png"))
    fig.savefig(os.path.join(figures_dir, fname + ".pdf"))
    if show:
        return fig
    plt.close(fig)
    print(f"  figure {fname}", flush=True)
    return None


def common_args(parser):
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--lang", type=str, default="en",
                        choices=["en", "zh", "es", "fr", "de", "ja"],
                        help="prompts.json language subset (en has most prompts).")
    return parser


def default_seeds():
    return [42, 43, 44, 45, 46]
