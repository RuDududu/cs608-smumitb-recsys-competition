# ============================================================
# submit.py — Final two-stage submission pipeline
#
# Best result: HM 0.074641 (2nd place)
# Config: ADMM-SLIM (l1=0.50, min_count=1) + BPR (r4, lr=0.001, ep=300)
#         two-stage candidate union, boost=0.29
# ============================================================

import random
import gc
import shutil
import numpy as np
import torch
from pathlib import Path

from config import (
    SEED, CACHE_DIR, CANDIDATE_K, SUBMISSION_ZIP,
    ADMM_MIN1_CONFIG, BPR_FULL_PARAMS, BEST_BOOST,
)
from data_utils import load_data, add_signal_columns, build_sparse, \
    item_pop, pop_order
from cache_utils import save_pkl, load_pkl
from models.admm_slim import run_admm_slim
from models.bpr import train_bprmf_min_rating, eval_bprmf
from submission import two_stage_candidate_union_submission, save_submission


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# Step 1: Load full union data
print("Reloading data with holdout_frac=0.0 (full union)...")
train_full, holdout_full, n_users_f, n_items_f, idx_to_item_f = load_data(
    holdout_frac=0.0,
    holdout_seed=42,
)

assert len(holdout_full) == 0, f"Expected 0 holdout got {len(holdout_full):,}"
assert len(train_full) == 238951, f"Expected 238,951 got {len(train_full):,}"
print(f"Full union verified: {len(train_full):,} interactions")

train_full  = add_signal_columns(train_full)
X_seen_full = build_sparse(train_full, n_users_f, n_items_f, "implicit_all")
i_pop_f     = item_pop(X_seen_full)
pop_ord_f   = pop_order(X_seen_full)
all_users_f = np.arange(n_users_f)

# Step 2: Train/load ADMM-SLIM (min_count=1, l1=0.50)
# Requires ~63GB VRAM (80GB A100 recommended)
admm_min1_cache = CACHE_DIR / (
    f"admm_slim_FULLUNION"
    f"_l1{ADMM_MIN1_CONFIG['lambda1']}"
    f"_l2{ADMM_MIN1_CONFIG['lambda2']}"
    f"_rho{ADMM_MIN1_CONFIG['rho']}"
    f"_iter{ADMM_MIN1_CONFIG['n_iter']}"
    f"_min{ADMM_MIN1_CONFIG['min_count']}"
    f"_ck{CANDIDATE_K}.pkl"
)

cached_admm = load_pkl(admm_min1_cache)

if cached_admm is not None:
    print(f"[CACHE HIT] ADMM min_count=1 full union")
    admm_min1_scores = cached_admm["scores"]
else:
    print("[CACHE MISS] Training ADMM-SLIM l1=0.50 min_count=1 on full union...")
    X_full = build_sparse(train_full, n_users_f, n_items_f, "implicit_all", drop_zeros=True)

    res_admm, recs_admm, scores_admm = run_admm_slim(
        X_full, X_seen_full, all_users_f, {}, pop_ord_f,
        lambda1=ADMM_MIN1_CONFIG["lambda1"],
        lambda2=ADMM_MIN1_CONFIG["lambda2"],
        rho=ADMM_MIN1_CONFIG["rho"],
        n_iter=ADMM_MIN1_CONFIG["n_iter"],
        min_count=ADMM_MIN1_CONFIG["min_count"],
        candidate_k=CANDIDATE_K,
    )
    payload = {
        "res":          res_admm,
        "recs":         recs_admm,
        "scores":       scores_admm,
        "config":       ADMM_MIN1_CONFIG,
        "n_users":      n_users_f,
        "n_items":      n_items_f,
        "train_size":   len(train_full),
        "holdout_frac": 0.0,
    }
    save_pkl(payload, admm_min1_cache)
    print(f"[CACHE SAVED] {admm_min1_cache}")
    admm_min1_scores = scores_admm
    del X_full; gc.collect(); torch.cuda.empty_cache()

# Step 3: Train/load BPR (min_rating=4, lr=0.001, ep=300)
bpr_full_cache_name = (
    f"BPRMF_FULLUNION"
    f"_r{BPR_FULL_PARAMS['min_rating']}"
    f"_f{BPR_FULL_PARAMS['factors']}"
    f"_lr{BPR_FULL_PARAMS['lr']}"
    f"_reg{BPR_FULL_PARAMS['reg']}"
    f"_ep{BPR_FULL_PARAMS['epochs']}"
    f"_bs{BPR_FULL_PARAMS['batch_size']}"
    f"_ck{CANDIDATE_K}"
    f"_seed{SEED}.pkl"
)
bpr_full_cache_path = CACHE_DIR / bpr_full_cache_name
bpr_full_cached     = load_pkl(bpr_full_cache_path)

if bpr_full_cached is not None:
    print(f"[CACHE HIT] BPR full union: {bpr_full_cache_path.name}")
    bpr_full_scores = bpr_full_cached["scores"]
else:
    print("[CACHE MISS] Training BPR on full union...")
    set_seed(SEED)

    model = train_bprmf_min_rating(
        train_df=train_full,
        n_users=n_users_f,
        n_items=n_items_f,
        min_rating=BPR_FULL_PARAMS["min_rating"],
        factors=BPR_FULL_PARAMS["factors"],
        lr=BPR_FULL_PARAMS["lr"],
        reg=BPR_FULL_PARAMS["reg"],
        epochs=BPR_FULL_PARAMS["epochs"],
        batch_size=BPR_FULL_PARAMS["batch_size"],
    )
    _, recs_bpr_full, bpr_full_scores = eval_bprmf(
        model=model, X_seen=X_seen_full, probe_users=all_users_f,
        ground_truth={}, pop_ord=pop_ord_f, candidate_k=CANDIDATE_K,
    )
    payload = {
        "recs":         recs_bpr_full,
        "scores":       bpr_full_scores,
        "params":       BPR_FULL_PARAMS,
        "n_users":      n_users_f,
        "n_items":      n_items_f,
        "train_size":   len(train_full),
        "holdout_frac": 0.0,
    }
    save_pkl(payload, bpr_full_cache_path)
    print(f"[CACHE SAVED] {bpr_full_cache_path}")
    del model; gc.collect(); torch.cuda.empty_cache()

# Step 4: Two-stage candidate union
print(f"\nADMM min1 full union users : {len(admm_min1_scores):,}")
print(f"BPR ep300 full union users : {len(bpr_full_scores):,}")

print(f"\nRunning two-stage pipeline (boost={BEST_BOOST})...")
recs = two_stage_candidate_union_submission(
    base_scores=admm_min1_scores,
    aux_scores=bpr_full_scores,
    X_seen=X_seen_full,
    users=all_users_f,
    pop_ord=pop_ord_f,
    aux_boost=BEST_BOOST,
)

# Step 5: Save and download submission
label           = f"2stage_admmMIN1_l050_bpr_r4_lr001_ep300_b{BEST_BOOST}"
submission_path = Path(SUBMISSION_ZIP).parent / f"submission_{label}.zip"

save_submission(recs, X_seen_full, idx_to_item_f, n_users_f, pop_ord_f)
shutil.copy(SUBMISSION_ZIP, submission_path)
print(f"\n[SUBMISSION] {submission_path}")

try:
    from google.colab import files
    files.download(str(submission_path))
except ImportError:
    print("(Not in Colab — submission zip saved locally)")

print("\n Submission generated.")
