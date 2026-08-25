"""Keyed PRF for Catopter: u ∈ [0,1) = trunc64(HMAC-SHA256(sk, payload)).

Two evaluation modes share ONE reference function so their outputs are
bitwise identical by construction:

  token_hash(sk, tid, prefix)     — one candidate (decoder hot path)
  token_hash_batch(sk, ids, pref) — every candidate of one encoder step,
                                    spread over persistent fork workers
                                    connected by pipes (~5ms/step at
                                    vocabulary scale vs ~30ms serial).

Workers only touch hashlib — never torch, never CUDA — which makes
fork-after-CUDA-init safe in practice (DataLoader-style usage).

Any failure (dead worker, recv timeout) permanently downgrades this process
to the inline path: correctness never depends on the pool.
"""

import hashlib
import hmac
import os

import numpy as np


# ---------------------------------------------------------------------------
#  Reference PRF (single value) — HMAC-SHA256 truncated to 64 bits
# ---------------------------------------------------------------------------

def _prf_hash(sk: str, tokens: list) -> bytes:
    """HMAC-SHA256 keyed by *sk* over a token-id list."""
    payload = "_".join(str(t) for t in tokens)
    return hmac.new(sk.encode(), payload.encode(), hashlib.sha256).digest()


def token_hash(sk: str, token_id: int, prefix: tuple) -> float:
    """Deterministic PRF u in [0, 1) keyed by (sk, prefix)."""
    hash_bytes = _prf_hash(sk, list(prefix) + [token_id])
    hash_int = int.from_bytes(hash_bytes[:8], "big")        # 64 bits
    return hash_int / float(1 << 64)                        # → [0, 1)


def check_boundary(sk: str, tokens: list,
                   hash_w: int = 3, hash_e: int = 1, hash_k: int = 1) -> bool:
    """True when at least hash_w - hash_e of the last hash_w tokens hit."""
    assert hash_w >= 1, "hash_w must be >= 1"
    assert 0 <= hash_e < hash_w, "hash_e must be in [0, hash_w)"
    assert 1 <= hash_k <= 8, "hash_k must be in [1, 8]"

    window = tokens[-hash_w:]
    mask = (1 << hash_k) - 1
    votes = 0
    for t in window:
        h = _prf_hash(sk, ["b", int(t)])
        if (h[0] & mask) == 0:
            votes += 1
    return votes >= (hash_w - hash_e)


# ---------------------------------------------------------------------------
#  Fast evaluator: identical digests, cached key schedule (HMAC ipad/opad
#  compressed once per sk; each candidate pays copy()+update()+digest()).
# ---------------------------------------------------------------------------

_HMAC_BASE_CACHE = {}


def _hmac_base(sk: str):
    base = _HMAC_BASE_CACHE.get(sk)
    if base is None:
        base = hmac.new(sk.encode(), digestmod=hashlib.sha256)
        _HMAC_BASE_CACHE[sk] = base
    return base


def token_hash_fast(sk: str, prefix, ids) -> list:
    """Float trunc64 values — identical to token_hash per item."""
    pre = "_".join(str(t) for t in prefix)
    if pre:
        pre += "_"
    base = _hmac_base(sk)
    out = []
    for t in ids:
        h = base.copy()
        h.update((pre + str(int(t))).encode())
        out.append(int.from_bytes(h.digest()[:8], "big") / float(1 << 64))
    return out


# ---------------------------------------------------------------------------
#  Persistent fork workers connected by pipes
# ---------------------------------------------------------------------------

_WORKERS = []                # [(proc, conn), ...] created lazily
_POOL_BROKEN = False         # any failure → permanent inline fallback
_PARALLEL_MIN = 512          # below this the inline loop is already fast
_MAX_WORKERS = 48
_RECV_TIMEOUT = 120          # seconds; a silent worker must not hang us


def _worker_loop(conn):
    while True:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt, SystemExit):
            return
        if msg is None:
            return
        sk, prefix, ids = msg
        conn.send(token_hash_fast(sk, prefix, ids))


def _start_workers(n):
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    procs = []
    for _ in range(n):
        parent, child = ctx.Pipe(duplex=True)
        p = ctx.Process(target=_worker_loop, args=(child,), daemon=True)
        p.start()
        child.close()
        procs.append((p, parent))
    return procs


def _get_workers():
    """Start the FULL pool once; never shrink (a small warmup batch used to
    freeze the pool size and throttle every later large batch)."""
    global _WORKERS
    if not _WORKERS:
        _WORKERS = _start_workers(max(1, min(_MAX_WORKERS,
                                             os.cpu_count() or 1)))
    return _WORKERS


def _shutdown_workers():
    global _WORKERS
    for proc, conn in _WORKERS:
        try:
            conn.send(None)
        except Exception:
            pass
        proc.terminate()
    _WORKERS = []


def token_hash_batch(sk: str, token_ids, prefix) -> np.ndarray:
    """u for many candidates at once — identical values to token_hash."""
    arr = np.asarray(token_ids, dtype=np.int64).ravel()
    n = arr.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    if n < _PARALLEL_MIN or _POOL_BROKEN or _MAX_WORKERS <= 1:
        return np.array(token_hash_fast(sk, prefix, arr), dtype=np.float64)

    try:
        workers = _get_workers()
        subs = np.array_split(arr, len(workers))
        for (_, conn), s in zip(workers, subs):
            conn.send((sk, tuple(prefix), s.tolist()))
        parts = []
        for _, conn in workers:
            if not conn.poll(_RECV_TIMEOUT):
                raise TimeoutError("PRF worker did not respond")
            parts.extend(conn.recv())
        u = np.array(parts, dtype=np.float64)
    except Exception:
        globals()["_POOL_BROKEN"] = True       # never retry within a run
        try:
            _shutdown_workers()
        except Exception:
            pass
        globals()["_WORKERS"] = []
        u = np.array(token_hash_fast(sk, prefix, arr), dtype=np.float64)

    return u                                    # dispatch order preserved


def warmup_pool():
    """Start the workers OUTSIDE timed regions so startup cost never lands
    inside a record's encoding_time."""
    if _POOL_BROKEN or _MAX_WORKERS <= 1:
        return
    try:
        token_hash_batch("42", np.arange(max(_PARALLEL_MIN,
                                             _MAX_WORKERS * 256)), (0,))
    except Exception:
        pass


def close_pool():
    """Release the workers (call once at the end of a run)."""
    try:
        _shutdown_workers()
    except Exception:
        pass
