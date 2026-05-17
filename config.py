# ============================================================
# config.py — Global constants and paths
# ============================================================

from pathlib import Path

SEED = 42

# ── Data paths (Google Drive) ─────────────────────────────────────────────────
TRAIN_PATH = "/content/drive/MyDrive/Colab Notebooks/cs608_ip_train_v3.csv"
PROBE_PATH = "/content/drive/MyDrive/Colab Notebooks/cs608_ip_probe_v3.csv"

# ── Cache directory ───────────────────────────────────────────────────────────
CACHE_DIR     = Path("/content/drive/MyDrive/Colab Notebooks/recsys_cache/trainplusprobe")
BPR_CACHE_DIR = CACHE_DIR  # BPR caches stored in same directory

# ── Evaluation ────────────────────────────────────────────────────────────────
K             = 50           # recommendation list length
CANDIDATE_K   = 3000         # candidate pool size per user
REL_THRESHOLD = 4.0          # minimum rating to count as relevant

# ── Submission output ─────────────────────────────────────────────────────────
SUBMISSION_TXT = "submission_v2.txt"
SUBMISSION_ZIP = "submission_v2.zip"

# ── EASE configurations ───────────────────────────────────────────────────────
# Only ease_all used in final pipeline; ease_lam200_min1 retained for ablation
EASE_CONFIGS = {
    "ease_all"         : {"col": "implicit_all", "lam": 200, "min_count": 2},
    "ease_lam200_min1" : {"col": "implicit_all", "lam": 200, "min_count": 1},
}

# ── Best model configs ────────────────────────────────────────────────────────
# ADMM-SLIM: l1=0.50 selected via lambda sweep (see explore.py)
# min_count=1 marginally outperforms min_count=2 on full union
ADMM_MIN1_CONFIG = {
    "lambda1":   0.5,
    "lambda2":   200.0,
    "rho":       40.0,
    "n_iter":    20,
    "min_count": 1,
}

# BPR: min_rating=4 > 3 (cleaner pairwise signal)
#      lr=0.001 > 0.003 (more refined convergence, plateaus ~ep250)
#      epochs=300 confirmed optimal via leaderboard sweep
BPR_FULL_PARAMS = {
    "min_rating": 4,
    "factors":    512,
    "lr":         0.001,
    "reg":        1e-5,
    "epochs":     300,
    "batch_size": 8192,
}

# ── Two-stage pipeline ────────────────────────────────────────────────────────
# boost=0.29 is the optimal value; non-monotonic around optimum (0.28 < 0.27 < 0.29)
BEST_BOOST = 0.29
