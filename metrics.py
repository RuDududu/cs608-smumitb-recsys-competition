# ============================================================
# metrics.py — Evaluation metrics (NDCG, NCRR, Recall, HM)
# ============================================================

import numpy as np
from config import K


def ndcg_at_k(ranked, rel, k):
    """Normalised Discounted Cumulative Gain at k."""
    dcg  = sum(1 / np.log2(r + 2) for r, i in enumerate(ranked[:k]) if i in rel)
    idcg = sum(1 / np.log2(r + 2) for r in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0


def ncrr_at_k(ranked, rel, k):
    """Normalised Cumulative Reciprocal Rank at k."""
    crr  = sum(1 / (r + 1) for r, i in enumerate(ranked[:k]) if i in rel)
    icrr = sum(1 / (r + 1) for r in range(min(len(rel), k)))
    return crr / icrr if icrr else 0.0


def recall_at_k(ranked, rel, k):
    """Recall at k."""
    return sum(1 for i in ranked[:k] if i in rel) / len(rel) if rel else 0.0


def evaluate(recs, gt, k=50):
    """
    Compute mean NDCG@k, NCRR@k, Recall@k, and HM over all users with ground truth.

    Returns:
        dict with keys: ndcg, ncrr, recall, hm, n
    """
    ns, cs, rs = [], [], []
    for u, rel in gt.items():
        if u not in recs or not rel:
            continue
        ns.append(ndcg_at_k(recs[u], rel, k))
        cs.append(ncrr_at_k(recs[u], rel, k))
        rs.append(recall_at_k(recs[u], rel, k))

    mn, mc, mr = float(np.mean(ns)), float(np.mean(cs)), float(np.mean(rs))
    hm = 3 / (1/mn + 1/mc + 1/mr) if (mn > 0 and mc > 0 and mr > 0) else 0.0
    return {"ndcg": mn, "ncrr": mc, "recall": mr, "hm": hm, "n": len(ns)}


def fmt(label, res, elapsed=0):
    """Pretty-print evaluation results."""
    print(f"\n  ── {label}  ({elapsed:.0f}s)")
    print(f"  NDCG@{K}   : {res['ndcg']:.6f}")
    print(f"  NCRR@{K}   : {res['ncrr']:.6f}")
    print(f"  Recall@{K} : {res['recall']:.6f}")
    print(f"  HM        : {res['hm']:.6f}")
