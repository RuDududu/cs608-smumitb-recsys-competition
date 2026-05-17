# ============================================================
# models/admm_slim.py — ADMM-SLIM
# Steck, H., et al. (2020). Admm slim: Sparse recommendations for
# many users. WSDM 2020.
# ============================================================

import time
import numpy as np
import torch
import gc

from config import CACHE_DIR, CANDIDATE_K, K
from data_utils import build_sparse, fill_to_k
from metrics import evaluate, fmt
from cache_utils import save_pkl, load_pkl


def soft_threshold(x, threshold):
    """L1 proximal operator — elementwise soft thresholding."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def run_admm_slim(X_train, X_seen, probe_users, gt, pop_ord,
                  lambda1=1.0, lambda2=500.0, rho=1000.0,
                  n_iter=100, min_count=2, candidate_k=3000, label="ADMM-SLIM"):
    """
    ADMM-SLIM: sparse item-item weight matrix via ADMM optimisation.

    L1 penalty (lambda1) controls sparsity — higher values retain only the
    strongest item-item co-occurrence edges, improving early-rank precision (NCRR).
    Best config: lambda1=0.50, lambda2=200, rho=40, n_iter=20, min_count=1.

    Args:
        X_train    : CSR sparse training matrix (implicit_all signal)
        X_seen     : CSR sparse seen-items matrix (for masking)
        probe_users: array of user indices
        gt         : ground truth dict {user: set(relevant_items)}
        pop_ord    : item popularity order for fallback filling
        lambda1    : L1 penalty (sparsity)
        lambda2    : L2 penalty (magnitude)
        rho        : ADMM step size
        n_iter     : number of ADMM iterations
        min_count  : minimum item interaction count
        candidate_k: candidates per user
        label      : display label

    Returns:
        res, recs, scores_store
    """
    t0 = time.time()

    Xcsc        = X_train.tocsc()
    item_counts = np.diff(Xcsc.indptr)
    keep        = np.where(item_counts >= min_count)[0]
    n_keep      = len(keep)
    X_red       = Xcsc[:, keep].toarray().astype("float32")

    print(f"[{label}] items={n_keep:,} lambda1={lambda1} lambda2={lambda2} rho={rho}")

    # Gram matrix + regularised inverse on GPU
    X_gpu = torch.tensor(X_red, device="cuda", dtype=torch.float32)
    G_gpu = X_gpu.T @ X_gpu
    XtX   = G_gpu.clone()

    G_gpu.diagonal().add_(lambda2 + rho)
    L     = torch.linalg.cholesky(G_gpu, upper=False)
    del G_gpu; torch.cuda.empty_cache()

    I_gpu = torch.eye(n_keep, device="cuda", dtype=torch.float32)
    Y     = torch.linalg.solve_triangular(L, I_gpu, upper=False)
    P_gpu = torch.linalg.solve_triangular(L.mT, Y, upper=True)
    del L, Y, I_gpu; torch.cuda.empty_cache()

    PXtX = P_gpu @ XtX
    del XtX; torch.cuda.empty_cache()

    print(f"[{label}] Gram inversion done ({time.time()-t0:.0f}s). "
          f"Running {n_iter} ADMM iterations...")

    # ADMM variables
    W = np.zeros((n_keep, n_keep), dtype="float32")
    Z = np.zeros((n_keep, n_keep), dtype="float32")
    U = np.zeros((n_keep, n_keep), dtype="float32")

    P_np = P_gpu.cpu().numpy()
    del P_gpu; torch.cuda.empty_cache()

    for it in range(1, n_iter + 1):
        # W update
        ZU    = torch.tensor(Z - U, device="cuda", dtype=torch.float32)
        W_gpu = torch.tensor(PXtX.cpu().numpy(), device="cuda", dtype=torch.float32) \
                + rho * (torch.tensor(P_np, device="cuda", dtype=torch.float32) @ ZU)
        W     = W_gpu.cpu().numpy()
        del W_gpu, ZU; torch.cuda.empty_cache()
        np.fill_diagonal(W, 0.0)

        # Z update (soft threshold + non-negativity)
        Z = np.maximum(soft_threshold(W + U, lambda1 / rho), 0.0)
        np.fill_diagonal(Z, 0.0)

        # Dual update
        U = U + W - Z

        if it % 10 == 0:
            residual  = np.abs(W - Z).mean()
            w_nonzero = int((Z != 0).sum())
            w_mean    = float(Z[Z != 0].mean()) if w_nonzero > 0 else 0.0
            print(f"[{label}] iter={it:3d}/{n_iter}  "
                  f"residual={residual:.6f}  "
                  f"nnz={w_nonzero:,}  "
                  f"W_mean={w_mean:.4f}  "
                  f"({time.time()-t0:.0f}s)")

    del PXtX; gc.collect(); torch.cuda.empty_cache()

    W_final = Z
    print(f"[{label}] W sparsity: {(W_final == 0).mean()*100:.1f}% zeros  "
          f"nnz={int((W_final != 0).sum()):,}")

    # GPU scoring
    W_gpu = torch.tensor(W_final, device="cuda", dtype=torch.float32)
    X_gpu = torch.tensor(X_red,   device="cuda", dtype=torch.float32)
    del W_final; torch.cuda.empty_cache()

    CHUNK = 2000
    recs, scores_store = {}, {}

    for start in range(0, len(probe_users), CHUNK):
        batch = probe_users[start:start + CHUNK]
        S_np  = (X_gpu[batch] @ W_gpu).cpu().numpy()

        for i, u in enumerate(batch):
            seen_items = X_seen[u].indices
            local_idx  = np.searchsorted(keep, seen_items)
            in_bounds  = local_idx < n_keep
            matched    = np.zeros_like(in_bounds, dtype=bool)
            matched[in_bounds] = keep[local_idx[in_bounds]] == seen_items[in_bounds]
            S_np[i, local_idx[matched]] = -np.inf

        top_local  = np.argsort(-S_np, axis=1)[:, :candidate_k]
        top_global = keep[top_local]

        for i, u in enumerate(batch):
            seen = set(X_seen[u].indices)
            tv, ts = top_global[i], S_np[i, top_local[i]]
            mask = ts > -np.inf
            vi, vs = tv[mask], ts[mask]
            scores_store[u] = {int(v): float(s) for v, s in zip(vi, vs)}
            recs[u] = fill_to_k(list(vi), seen, pop_ord, candidate_k)

    del X_gpu, W_gpu; torch.cuda.empty_cache(); gc.collect()

    res = evaluate(recs, gt, K)
    fmt(label, res, time.time() - t0)
    return res, recs, scores_store


# Global config reference for cached wrapper
ADMM_SLIM_CONFIG = None


def run_admm_slim_cached(train_df, n_users, n_items, X_seen, probe_users, gt, pop_ord):
    """
    Cached wrapper for run_admm_slim using the global ADMM_SLIM_CONFIG.
    Set ADMM_SLIM_CONFIG before calling.
    """
    cfg        = ADMM_SLIM_CONFIG
    cache_path = CACHE_DIR / (
        f"admm_slim_l1{cfg['lambda1']}_l2{cfg['lambda2']}"
        f"_rho{cfg['rho']}_iter{cfg['n_iter']}"
        f"_min{cfg['min_count']}_ck{CANDIDATE_K}.pkl"
    )
    cached = load_pkl(cache_path)
    if cached is not None:
        print(f"[CACHE HIT] ADMM-SLIM  HM={cached['res']['hm']:.6f}")
        return cached

    X_var = build_sparse(train_df, n_users, n_items, "implicit_all", drop_zeros=True)
    res, recs, scores = run_admm_slim(
        X_var, X_seen, probe_users, gt, pop_ord,
        lambda1=cfg["lambda1"], lambda2=cfg["lambda2"],
        rho=cfg["rho"], n_iter=cfg["n_iter"],
        min_count=cfg["min_count"], candidate_k=CANDIDATE_K,
    )
    payload = {"res": res, "recs": recs, "scores": scores}
    save_pkl(payload, cache_path)
    print(f"[CACHE SAVED] {cache_path}")
    del X_var; gc.collect(); torch.cuda.empty_cache()
    return payload
