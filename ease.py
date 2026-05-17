# ============================================================
# models/ease.py — EASE (Embarrassingly Shallow Autoencoders)
# Steck, H. (2019). Embarrassingly shallow autoencoders for sparse data.
# ============================================================

import time
import numpy as np
import torch
import gc

from config import CACHE_DIR, CANDIDATE_K, K, EASE_CONFIGS
from data_utils import build_sparse, fill_to_k
from metrics import evaluate, fmt
from cache_utils import save_pkl, load_pkl


def run_ease(X_train, X_seen, probe_users, ground_truth, pop_ord,
             lam=200.0, min_count=2, candidate_k=3000, label="EASE"):
    """
    EASE item-item weight matrix via Cholesky inversion on GPU.

    Args:
        X_train     : CSR sparse training matrix
        X_seen      : CSR sparse seen-items matrix (for masking)
        probe_users : array of user indices to generate recs for
        ground_truth: dict {user: set(relevant_items)} for evaluation
        pop_ord     : item popularity order for fallback filling
        lam         : L2 regularisation strength
        min_count   : minimum item interaction count to include in weight matrix
        candidate_k : number of candidates to generate per user
        label       : display label

    Returns:
        res, recs, scores_store
    """
    t0 = time.time()

    Xcsc        = X_train.tocsc()
    item_counts = np.diff(Xcsc.indptr)
    keep        = np.where(item_counts >= min_count)[0]
    n_keep      = len(keep)
    X_red       = Xcsc[:, keep].toarray().astype("float32")

    print(f"[{label}] items={n_keep:,}  gram={n_keep**2*4/1e9:.2f} GB")

    # Gram matrix on GPU
    X_gpu = torch.tensor(X_red, device="cuda", dtype=torch.float32)
    G_gpu = X_gpu.T @ X_gpu
    del X_gpu; torch.cuda.empty_cache()

    # Regularisation + Cholesky inversion
    G_gpu.diagonal().add_(lam)
    L     = torch.linalg.cholesky(G_gpu, upper=False)
    del G_gpu; torch.cuda.empty_cache()

    I     = torch.eye(n_keep, device="cuda", dtype=torch.float32)
    Y     = torch.linalg.solve_triangular(L, I, upper=False)
    P     = torch.linalg.solve_triangular(L.mT, Y, upper=True)
    del L, Y, I; torch.cuda.empty_cache()

    diag_P = P.diagonal().clone()
    W      = P / (-diag_P.unsqueeze(0))
    W.diagonal().fill_(0.0)
    del P; torch.cuda.empty_cache()

    # Chunked inference
    X_gpu = torch.tensor(X_red, device="cuda", dtype=torch.float32)
    CHUNK = 2000
    recs, scores_store = {}, {}

    for start in range(0, len(probe_users), CHUNK):
        batch = probe_users[start:start + CHUNK]
        S_np  = (X_gpu[batch] @ W).cpu().numpy()

        for i, u in enumerate(batch):
            seen_items = X_seen[u].indices
            local_idx  = np.searchsorted(keep, seen_items)
            in_bounds  = local_idx < n_keep
            matched    = np.zeros_like(in_bounds, dtype=bool)
            matched[in_bounds] = keep[local_idx[in_bounds]] == seen_items[in_bounds]
            S_np[i, local_idx[matched]] = -np.inf

            top_local_i  = np.argsort(-S_np[i])[:candidate_k]
            top_global_i = keep[top_local_i]
            ts           = S_np[i, top_local_i]
            mask         = ts > -np.inf
            vi, vs       = top_global_i[mask], ts[mask]

            seen = set(seen_items)
            scores_store[u] = {int(v): float(s) for v, s in zip(vi, vs)}
            recs[u] = fill_to_k(list(vi), seen, pop_ord, candidate_k)

    del X_gpu, W; torch.cuda.empty_cache(); gc.collect()

    res = evaluate(recs, ground_truth, K)
    fmt(label, res, time.time() - t0)
    return res, recs, scores_store


def run_all_ease_variants(train_df, n_users, n_items, probe_users, gt, pop_ord, X_seen):
    """
    Run EASE for all configs defined in EASE_CONFIGS, with caching.
    Used for exploratory baseline benchmarking only — not part of final pipeline.
    """
    results = {}
    for name, cfg in EASE_CONFIGS.items():
        cache_path = CACHE_DIR / f"{name}_lam{cfg['lam']}_min{cfg['min_count']}_ck{CANDIDATE_K}.pkl"
        cached = load_pkl(cache_path)

        if cached is not None:
            print(f"[CACHE HIT] {name}  HM={cached['res']['hm']:.6f}")
            results[name] = cached
            continue

        X_var = build_sparse(train_df, n_users, n_items, cfg["col"], drop_zeros=True)
        res, recs, scores = run_ease(
            X_var, X_seen, probe_users, gt, pop_ord,
            lam=float(cfg["lam"]), min_count=cfg["min_count"],
            candidate_k=CANDIDATE_K, label=name,
        )
        payload = {"res": res, "recs": recs, "scores": scores}
        save_pkl(payload, cache_path)
        print(f"[CACHE SAVED] {cache_path}")
        results[name] = payload
        del X_var; gc.collect(); torch.cuda.empty_cache()

    return results
