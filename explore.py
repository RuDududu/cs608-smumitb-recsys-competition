# ============================================================
# explore.py — Exploratory model sweeps and diagnostic
#
# Run this notebook-style: execute sections sequentially.
# All results are cached to CACHE_DIR to avoid retraining.
# ============================================================

import random
import gc
import numpy as np
import torch

from config import SEED, CACHE_DIR, CANDIDATE_K
from data_utils import load_data, add_signal_columns, build_sparse, \
    get_ground_truth, item_pop, pop_order
from cache_utils import load_pkl
from models.ease import run_all_ease_variants
from models.admm_slim import run_admm_slim_cached, ADMM_SLIM_CONFIG
import models.admm_slim as admm_module
from models.bpr import train_bprmf_min_rating, eval_bprmf


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

REL_THRESHOLD = 4.0

# Step 1: Load data (85% holdout for local eval)
train_df, holdout_df, n_users, n_items, idx_to_item = load_data(
    holdout_frac=0.15,
    holdout_seed=42,
)
train_df = add_signal_columns(train_df)
X_seen   = build_sparse(train_df, n_users, n_items, "implicit_all")
gt          = get_ground_truth(holdout_df, REL_THRESHOLD)
probe_users = np.array(holdout_df.u.unique())
i_pop       = item_pop(X_seen)
pop_ord     = pop_order(X_seen)

# Step 2: EASE lambda sweep (exploratory baseline)
# Note: EASE was not part of the final pipeline. Included for ablation only.
print("=" * 60)
print("EASE LAMBDA SWEEP")
print("=" * 60)

ease_results = run_all_ease_variants(
    train_df, n_users, n_items, probe_users, gt, pop_ord, X_seen
)

print("\nEASE lambda sweep summary:")
best_name, best_hm = None, -1
for name, payload in ease_results.items():
    r      = payload["res"]
    marker = " ◄ best" if r["hm"] > best_hm else ""
    if r["hm"] > best_hm:
        best_hm, best_name = r["hm"], name
    print(f"  {name:<20} HM={r['hm']:.6f}  NDCG={r['ndcg']:.6f}  "
          f"NCRR={r['ncrr']:.6f}  Recall={r['recall']:.6f}{marker}")

print(f"\nBest: {best_name}  HM={best_hm:.6f}")
print(f"Projected leaderboard: {best_hm * 1.56:.6f}")

# Step 3: ADMM-SLIM lambda sweep (min_count=2)
ADMM_SWEEP = [
    {"lambda1": 0.10, "lambda2": 100,  "rho": 40, "n_iter": 20, "min_count": 2},
    {"lambda1": 0.10, "lambda2": 500,  "rho": 40, "n_iter": 20, "min_count": 2},
    {"lambda1": 0.50, "lambda2": 200,  "rho": 40, "n_iter": 20, "min_count": 2},
    {"lambda1": 0.10, "lambda2": 50,   "rho": 40, "n_iter": 20, "min_count": 2},
    {"lambda1": 0.10, "lambda2": 1000, "rho": 40, "n_iter": 20, "min_count": 2},
]

admm_sweep_results = {}
best_admm_hm, best_admm_cfg, best_admm_result = -1, None, None

print("=" * 60)
print("ADMM-SLIM LAMBDA SWEEP")
print("=" * 60)

for cfg in ADMM_SWEEP:
    admm_module.ADMM_SLIM_CONFIG = cfg
    label = (f"l1={cfg['lambda1']} l2={cfg['lambda2']} "
             f"rho={cfg['rho']} iter={cfg['n_iter']}")
    print(f"\ {label}")

    result = run_admm_slim_cached(
        train_df, n_users, n_items, X_seen, probe_users, gt, pop_ord
    )

    hm     = result["res"]["hm"]
    marker = " <<< best" if hm > best_admm_hm else ""
    if hm > best_admm_hm:
        best_admm_hm, best_admm_cfg, best_admm_result = hm, label, result

    admm_sweep_results[label] = result
    print(f"  HM={hm:.6f}  NDCG={result['res']['ndcg']:.6f}  "
          f"NCRR={result['res']['ncrr']:.6f}  "
          f"Recall={result['res']['recall']:.6f}  "
          f"projected LB={hm*1.56:.6f}{marker}")

# Step 4: ADMM-SLIM min_count=1 benchmark
admm_module.ADMM_SLIM_CONFIG = {
    "lambda1": 0.5, "lambda2": 200, "rho": 40, "n_iter": 20, "min_count": 1
}

print("=" * 60)
print("ADMM-SLIM min_count=1 benchmark")
print("=" * 60)

admm_min1_result = run_admm_slim_cached(
    train_df, n_users, n_items, X_seen, probe_users, gt, pop_ord
)
r = admm_min1_result["res"]
print(f"\n  HM={r['hm']:.6f}  NDCG={r['ndcg']:.6f}  "
      f"NCRR={r['ncrr']:.6f}  Recall={r['recall']:.6f}  "
      f"projected LB={r['hm']*1.56:.6f}")

# Step 5: BPR (85% union, used for recall diagnostic)
# min_rating=4 outperforms 3; lr=0.003/ep=100 used here for diagnostic only.
# Final submission uses lr=0.001/ep=300 on full union (see submit.py).

BPR_PARAMS = {
    "min_rating": 4,
    "factors":    512,
    "lr":         0.003,
    "reg":        1e-5,
    "epochs":     100,
    "batch_size": 8192,
}

from config import BPR_CACHE_DIR
bpr_cache_name = (
    f"BPRMF_r{BPR_PARAMS['min_rating']}"
    f"_f{BPR_PARAMS['factors']}"
    f"_lr{BPR_PARAMS['lr']}"
    f"_reg{BPR_PARAMS['reg']}"
    f"_ep{BPR_PARAMS['epochs']}"
    f"_bs{BPR_PARAMS['batch_size']}"
    f"_ck{CANDIDATE_K}"
    f"_seed{SEED}.pkl"
)
bpr_cache_path = BPR_CACHE_DIR / bpr_cache_name

print("\n" + "=" * 60)
print(f"[BPRMF] f={BPR_PARAMS['factors']} lr={BPR_PARAMS['lr']}")
print("=" * 60)

cached = load_pkl(bpr_cache_path)

if cached is not None:
    print(f"[CACHE HIT] Loaded BPR from {bpr_cache_path}")
    best_bpr = cached
else:
    print("[CACHE MISS] Training BPR")
    set_seed(SEED)

    model = train_bprmf_min_rating(
        train_df=train_df,
        n_users=n_users,
        n_items=n_items,
        min_rating=BPR_PARAMS["min_rating"],
        factors=BPR_PARAMS["factors"],
        lr=BPR_PARAMS["lr"],
        reg=BPR_PARAMS["reg"],
        epochs=BPR_PARAMS["epochs"],
        batch_size=BPR_PARAMS["batch_size"],
    )
    res, recs, scores = eval_bprmf(
        model=model, X_seen=X_seen, probe_users=probe_users,
        ground_truth=gt, pop_ord=pop_ord, candidate_k=CANDIDATE_K,
    )
    best_bpr = {
        "label": f"BPRMF f={BPR_PARAMS['factors']} lr={BPR_PARAMS['lr']}",
        "params": BPR_PARAMS, "candidate_k": CANDIDATE_K, "seed": SEED,
        "res": res, "recs": recs, "scores": scores,
    }
    from cache_utils import save_pkl
    save_pkl(best_bpr, bpr_cache_path)
    print(f"[CACHE SAVED] {bpr_cache_path}")
    del model; torch.cuda.empty_cache(); gc.collect()

print(
    f"\n  {best_bpr['label']}\n"
    f"  NDCG@50   : {best_bpr['res']['ndcg']:.6f}\n"
    f"  NCRR@50   : {best_bpr['res']['ncrr']:.6f}\n"
    f"  Recall@50 : {best_bpr['res']['recall']:.6f}\n"
    f"  HM        : {best_bpr['res']['hm']:.6f}\n"
)

# Step 6: Unique recall diagnostic
# Quantifies how many relevant items BPR retrieves that ADMM-SLIM misses entirely.
# This motivates the two-stage candidate union pipeline in submit.py.

admm_85  = load_pkl(CACHE_DIR / "admm_slim_l10.5_l2200_rho40_iter20_min2_ck3000.pkl")
bpr_85   = load_pkl(BPR_CACHE_DIR / bpr_cache_name)

admm_scores_85 = admm_85["scores"]
bpr_scores_85  = bpr_85["scores"]

bpr_unique_hits = []
for u in probe_users:
    admm_items_u = set(admm_scores_85.get(u, {}).keys())
    bpr_items_u  = set(bpr_scores_85.get(u, {}).keys())
    relevant     = gt.get(u, set())
    if not relevant:
        continue
    bpr_only = bpr_items_u - admm_items_u
    hits     = len(bpr_only & relevant)
    bpr_unique_hits.append(hits / len(relevant))

print(f"BPR unique recall contribution : {np.mean(bpr_unique_hits):.4f}")
print(f"i.e. {np.mean(bpr_unique_hits)*100:.1f}% of relevant items only BPR finds")

all_bpr_items  = set().union(*[set(bpr_scores_85.get(u, {}).keys()) for u in probe_users])
all_admm_items = set().union(*[set(admm_scores_85.get(u, {}).keys()) for u in probe_users])
bpr_only_items = all_bpr_items - all_admm_items
print(f"\nTotal BPR items considered    : {len(all_bpr_items):,}")
print(f"BPR-unique items (not in ADMM): {len(bpr_only_items):,}")
