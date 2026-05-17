# CS608 Individual Project — Recommender System

**Final Result: 2nd Place | HM 0.074641**

| Metric | Score |
|---|---|
| Harmonic Mean | 0.074641 |
| NDCG@50 | 0.070900 |
| NCRR@50 | 0.060204 |
| Recall@50 | 0.105510 |

**Note:** This code was originally developed and run on Google Colab (A100 GPU).
The Python files here are a clean refactoring of the original notebook for reference purposes.
Path constants in `config.py` point to Google Drive locations and will need to be updated
for local use.

## Pipeline

Two-stage candidate union:
1. **ADMM-SLIM** (Steck et al., 2020) — primary reranker, l1=0.50, l2=200, min_count=1, full union
2. **BPR** (Rendle et al., 2009) — candidate retrieval only, min_rating≥4, lr=0.001, ep=300, full union
3. **Union** of candidate pools → rescore with ADMM-SLIM scores exclusively → boost=0.29 for BPR-unique items

BPR contributes 20.5% of relevant items invisible to ADMM-SLIM (10,524 unique items out of 41,402 considered).

## File Structure

```
cs608_recsys/
├── config.py          # All constants, paths, and best hyperparameters
├── data_utils.py      # Data loading, sparse matrix, signal columns
├── metrics.py         # NDCG, NCRR, Recall, HM evaluation
├── cache_utils.py     # Pickle cache save/load helpers
├── submission.py      # Two-stage pipeline and submission writer
├── explore.py         # Exploratory sweeps (EASE, ADMM, BPR diagnostic)
├── submit.py          # Final submission — runs the best pipeline end-to-end
└── models/
    ├── ease.py        # EASE model (exploratory baseline)
    ├── admm_slim.py   # ADMM-SLIM model (final reranker)
    └── bpr.py         # BPR-MF model (candidate retrieval)
```

## Usage

### Reproduce final submission
```bash
python submit.py
```
Requires 80GB A100 GPU for ADMM-SLIM min_count=1 training (~25 min).
Subsequent runs load from cache (<5 min).

### Run exploratory sweeps
```bash
python explore.py
```
Runs EASE lambda sweep, ADMM-SLIM lambda sweep, BPR training, and unique recall diagnostic.

## Key Hyperparameters

| Component | Parameter | Best Value |
|---|---|---|
| ADMM-SLIM | λ₁ (L1 penalty) | 0.50 |
| ADMM-SLIM | λ₂ (L2 penalty) | 200 |
| ADMM-SLIM | min_count | 1 |
| BPR | Min. rating (positives) | ≥4 |
| BPR | Learning rate | 0.001 |
| BPR | Epochs | 300 |
| Pipeline | Boost (aux_boost) | 0.29 |

## Data Strategy

The probe CSV (73,943 interactions) was pooled with the training CSV (165,008 interactions) into a full union of 238,951 interactions (after deduplication). A 15% per-user stratified holdout was carved from the union for local evaluation. The full union was used for final submissions.

## Requirements

```
torch>=2.0
numpy
pandas
scipy
```

## References

- Steck, H. (2019). Embarrassingly shallow autoencoders for sparse data. *WWW 2019*.
- Steck, H., et al. (2020). ADMM SLIM: Sparse recommendations for many users. *WSDM 2020*.
- Rendle, S., et al. (2009). BPR: Bayesian personalized ranking from implicit feedback. *UAI 2009*.
- Ning, X., & Karypis, G. (2011). SLIM: Sparse linear methods for top-n recommender systems. *ICDM 2011*.
