# ============================================================
# submission.py — Two-stage pipeline and submission writer
# ============================================================

import os
import zipfile
import numpy as np

from config import SUBMISSION_TXT, SUBMISSION_ZIP, K
from data_utils import fill_to_k


def two_stage_candidate_union_submission(
    base_scores, aux_scores, X_seen, users, pop_ord,
    aux_boost=0.29, k=50,
):
    """
    Two-stage candidate union pipeline.

    Stage 1 (Retrieval): Union of ADMM-SLIM candidates and BPR candidates.
    Stage 2 (Reranking): Rescore using ADMM-SLIM scores exclusively.
      - ADMM items: admm_norm(item) + aux_boost * bpr_norm(item)
      - BPR-unique items: aux_boost * bpr_norm(item)

    BPR scores are discarded after candidate generation. The additive boost
    on BPR-unique items prevents them from defaulting to the bottom of the
    ranked list without reordering ADMM-SLIM's own candidates.

    This design exploits the 20.5% unique recall finding: relevant items
    exclusively retrievable by BPR can appear in the final top-50, while
    ADMM-SLIM's precise item-item weights retain full ranking control.

    Args:
        base_scores : dict {user: {item: score}} from ADMM-SLIM (full union)
        aux_scores  : dict {user: {item: score}} from BPR (full union)
        X_seen      : CSR sparse seen-items matrix
        users       : array of all user indices
        pop_ord     : item popularity order for fallback filling
        aux_boost   : weight applied to BPR scores (best: 0.29)
        k           : final recommendation list length

    Returns:
        dict {user: [item_idx, ...]} of length k per user
    """
    final = {}

    for u in users:
        seen = set(X_seen[u].indices)
        base = base_scores.get(u, {})
        aux  = aux_scores.get(u, {})

        def norm(d):
            if not d:
                return {}
            vals = np.array(list(d.values()), dtype=np.float32)
            lo, hi = vals.min(), vals.max()
            if hi == lo:
                return {item: 1.0 for item in d}
            return {item: float((v - lo) / (hi - lo)) for item, v in d.items()}

        base_n = norm(base)
        aux_n  = norm(aux)

        all_candidates = set(base_n.keys()) | set(aux_n.keys())
        scored = {}

        for item in all_candidates:
            if item in seen:
                continue
            if item in base_n:
                scored[item] = base_n[item] + aux_boost * aux_n.get(item, 0.0)
            else:
                scored[item] = aux_boost * aux_n[item]

        ranked   = sorted(scored, key=scored.get, reverse=True)
        final[u] = fill_to_k(ranked, seen, pop_ord, k)

    return final


def save_submission(recs, X_seen, idx_to_item, n_users, pop_ord, k=50):
    """
    Write recommendations to submission txt and zip.
    Validates format (n_users lines, k items per line) before returning.

    Args:
        recs        : dict {user_idx: [item_idx, ...]}
        X_seen      : CSR sparse seen-items matrix
        idx_to_item : dict {item_idx: original_item_id}
        n_users     : total number of users
        pop_ord     : item popularity order for fallback filling
        k           : recommendation list length
    """
    with open(SUBMISSION_TXT, "w") as f:
        for u in range(n_users):
            seen  = set(X_seen[u].indices)
            items = fill_to_k(recs.get(u, []), seen, pop_ord, k)
            f.write(" ".join(str(idx_to_item[v]) for v in items) + "\n")

    with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(SUBMISSION_TXT)

    with open(SUBMISSION_TXT) as f:
        lines = f.readlines()

    assert len(lines) == n_users and all(len(l.split()) == k for l in lines), \
        "Submission format check failed"

    print(f"\n✓ {SUBMISSION_ZIP}  ({os.path.getsize(SUBMISSION_ZIP)/1e3:.0f} KB)")
    print(f"  {len(lines):,} users × {k} items")
    print(f"  First: {lines[0][:70].strip()}")
    print(f"  Last : {lines[-1][:70].strip()}")
