# ============================================================
# data_utils.py — Data loading and preprocessing utilities
# ============================================================

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from config import TRAIN_PATH, PROBE_PATH, REL_THRESHOLD, K


def load_data(holdout_frac=0.15, holdout_seed=42):
    """
    Union strategy: pool train + probe CSVs into one dataset, then carve
    out a per-user stratified holdout for local evaluation.

    A 15% per-user stratified holdout is carved from the union for local
    evaluation. The remaining 85% is used for training. For final submission,
    holdout_frac=0.0 is used to train on the full union.

    Args:
        holdout_frac : fraction of each user's interactions to hold out
        holdout_seed : RNG seed for reproducibility

    Returns:
        train_df, holdout_df, n_users, n_items, idx_to_item
    """
    train_raw = pd.read_csv(TRAIN_PATH)
    probe_raw = pd.read_csv(PROBE_PATH)

    for df in [train_raw, probe_raw]:
        df.columns = df.columns.str.strip().str.lower()

    # Pool train and probe into a single union dataset
    union = pd.concat([train_raw, probe_raw], ignore_index=True)
    union = union.drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)

    print(f"train_raw : {len(train_raw):,}  |  probe_raw : {len(probe_raw):,}")
    print(f"Union (after dedup): {len(union):,}")

    # Build global ID maps from the union
    all_users   = union.user_id.unique()
    all_items   = union.item_id.unique()
    user_map    = {u: i for i, u in enumerate(sorted(all_users))}
    item_map    = {v: i for i, v in enumerate(sorted(all_items))}
    idx_to_item = {i: v for v, i in item_map.items()}

    union["u"] = union.user_id.map(user_map)
    union["v"] = union.item_id.map(item_map)

    n_users = len(user_map)
    n_items = len(item_map)

    # Per-user stratified split
    rng          = np.random.default_rng(holdout_seed)
    holdout_mask = np.zeros(len(union), dtype=bool)

    for u_idx, grp in union.groupby("u"):
        n      = len(grp)
        n_hold = int(np.floor(n * holdout_frac))
        if n_hold == 0:
            continue
        hold_indices              = rng.choice(grp.index.values, size=n_hold, replace=False)
        holdout_mask[hold_indices] = True

    train_df   = union[~holdout_mask].reset_index(drop=True)
    holdout_df = union[holdout_mask].reset_index(drop=True)

    print(f"\nSplit summary (holdout_frac={holdout_frac}):")
    print(f"  Train (new) : {len(train_df):,}  interactions")
    print(f"  Holdout     : {len(holdout_df):,}  interactions")
    print(f"  Users       : {n_users:,}")
    print(f"  Items       : {n_items:,}")
    sparsity = 1 - len(train_df) / (n_users * n_items)
    print(f"  Sparsity    : {sparsity:.6f}")
    print(f"  Avg interactions/user (train): {len(train_df)/train_df.u.nunique():.1f}")
    print(f"\nTrain rating distribution:")
    print(train_df.rating.value_counts().sort_index())
    print(f"\nHoldout users with relevant items (rating>={REL_THRESHOLD}): "
          f"{(holdout_df.rating >= REL_THRESHOLD).groupby(holdout_df.u).any().sum():,}")

    return train_df, holdout_df, n_users, n_items, idx_to_item


def add_signal_columns(df):
    """Add interaction signal columns used by EASE and ADMM-SLIM."""
    df = df.copy()

    df["implicit_all"] = 1.0
    df["high_conf"]    = (df["rating"] >= 4).astype("float32")
    df["rating_wt"]    = (df["rating"] / 5.0).astype("float32")

    user_mean = df.groupby("u")["rating"].transform("mean")
    df["user_mean_centered"] = (df["rating"] - user_mean).clip(lower=0).astype("float32")

    item_mean = df.groupby("v")["rating"].transform("mean")
    df["item_deviation"] = (df["rating"] - item_mean).clip(lower=0).astype("float32")

    df["relative_preference"] = (
        (df["rating"] - user_mean) / (5.0 - user_mean + 1e-10)
    ).clip(lower=0).astype("float32")

    return df


def build_sparse(df, n_users, n_items, col, drop_zeros=True):
    """Build a CSR sparse interaction matrix from a signal column."""
    tmp = df[["u", "v", col]].copy().rename(columns={col: "w"})
    tmp["w"] = tmp["w"].astype("float32")
    if drop_zeros:
        tmp = tmp[tmp["w"] > 0]
    return csr_matrix(
        (tmp.w.values, (tmp.u.values, tmp.v.values)),
        shape=(n_users, n_items),
    )


def get_ground_truth(probe_df, threshold=4.0):
    """Build ground truth dict {user_idx: set(relevant_item_idxs)} from holdout."""
    rel = probe_df[probe_df.rating >= threshold]
    return rel.groupby("u")["v"].apply(set).to_dict()


def item_pop(X):
    """Item popularity vector (interaction counts)."""
    return np.asarray(X.sum(axis=0)).ravel()


def pop_order(X):
    """Item indices sorted by descending popularity."""
    return np.argsort(-item_pop(X))


def fill_to_k(items, seen, pop_ord, k=50):
    """
    Fill recommendation list to exactly k items.
    Falls back to popularity-ordered unseen items if candidates are exhausted.
    """
    result, used = [], set()
    for v in items:
        if v not in seen and v not in used:
            result.append(v)
            used.add(v)
        if len(result) == k:
            return result
    for v in pop_ord:
        if v not in seen and v not in used:
            result.append(v)
            used.add(v)
        if len(result) == k:
            return result
    return result
