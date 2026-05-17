# ============================================================
# models/bpr.py — Bayesian Personalised Ranking (BPR-MF)
# Rendle, S., et al. (2009). BPR: Bayesian personalized ranking
# from implicit feedback. UAI 2009.
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

from config import SEED, K, CANDIDATE_K
from data_utils import fill_to_k
from metrics import evaluate, fmt


class BPRMF(nn.Module):
    """Matrix factorisation model trained with BPR pairwise loss."""

    def __init__(self, n_users, n_items, factors=512):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, factors)
        self.item_emb = nn.Embedding(n_items, factors)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, users, pos_items, neg_items):
        u = self.user_emb(users)
        i = self.item_emb(pos_items)
        j = self.item_emb(neg_items)
        pos = (u * i).sum(dim=1)
        neg = (u * j).sum(dim=1)
        return pos, neg


def train_bprmf_min_rating(
    train_df,
    n_users,
    n_items,
    min_rating=4,
    factors=512,
    lr=0.001,
    reg=1e-5,
    epochs=300,
    batch_size=8192,
):
    """
    Train BPRMF using interactions with rating >= min_rating as positives.

    Key hyperparameter decisions (via leaderboard sweep):
      - min_rating=4 > 3: high-confidence interactions provide cleaner pairwise signal
      - lr=0.001 > 0.003: slower learning finds a sharper loss minimum
      - epochs=300: loss plateaus ~ep250; ep400+ shows no improvement

    BPR is used exclusively as a candidate retrieval model in the two-stage pipeline.
    Its scores are discarded after candidate generation; ADMM-SLIM controls all ranking.

    Args:
        train_df   : training DataFrame with columns [u, v, rating]
        n_users    : total number of users
        n_items    : total number of items
        min_rating : minimum rating to treat as a positive interaction
        factors    : embedding dimension
        lr         : learning rate
        reg        : L2 regularisation (weight decay)
        epochs     : number of training epochs
        batch_size : mini-batch size

    Returns:
        Trained BPRMF model (on CUDA)
    """
    pos_df   = train_df[train_df["rating"] >= min_rating][["u", "v"]].drop_duplicates()
    user_pos = pos_df.groupby("u")["v"].apply(set).to_dict()
    users_np = pos_df["u"].values.astype(np.int64)
    pos_np   = pos_df["v"].values.astype(np.int64)

    model = BPRMF(n_users, n_items, factors=factors).cuda()
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

    n   = len(users_np)
    rng = np.random.default_rng(SEED)

    for epoch in range(1, epochs + 1):
        perm   = rng.permutation(n)
        losses = []

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            u   = users_np[idx]
            i   = pos_np[idx]
            j   = rng.integers(0, n_items, size=len(idx))

            # Resample negatives that are actually positives
            for t in range(len(j)):
                while j[t] in user_pos.get(u[t], set()):
                    j[t] = rng.integers(0, n_items)

            u_t = torch.tensor(u, dtype=torch.long, device="cuda")
            i_t = torch.tensor(i, dtype=torch.long, device="cuda")
            j_t = torch.tensor(j, dtype=torch.long, device="cuda")

            pos_s, neg_s = model(u_t, i_t, j_t)
            loss = -F.logsigmoid(pos_s - neg_s).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"[BPRMF >= {min_rating}] epoch={epoch} loss={np.mean(losses):.5f}")

    return model


def eval_bprmf(model, X_seen, probe_users, ground_truth, pop_ord, candidate_k=3000):
    """
    Generate top-k candidates per user using trained BPRMF.

    Args:
        model        : trained BPRMF model
        X_seen       : CSR sparse seen-items matrix
        probe_users  : array of user indices
        ground_truth : dict {user: set(relevant_items)} — pass {} for full union
        pop_ord      : item popularity order for fallback filling
        candidate_k  : candidates per user

    Returns:
        res, recs, scores_store
    """
    model.eval()
    item_factors = model.item_emb.weight.detach()
    recs, scores_store = {}, {}
    CHUNK = 1000

    with torch.no_grad():
        for start in range(0, len(probe_users), CHUNK):
            batch = probe_users[start:start + CHUNK]
            u_t   = torch.tensor(batch, dtype=torch.long, device="cuda")
            U     = model.user_emb(u_t)
            S     = (U @ item_factors.T).cpu().numpy()

            for i, u in enumerate(batch):
                seen = set(X_seen[u].indices)
                if seen:
                    S[i, list(seen)] = -np.inf

                top  = np.argpartition(-S[i], candidate_k)[:candidate_k]
                top  = top[np.argsort(-S[i, top])]
                vals = S[i, top]
                mask = vals > -np.inf

                items = top[mask]
                vs    = vals[mask]

                scores_store[u] = {int(v): float(s) for v, s in zip(items, vs)}
                recs[u] = fill_to_k(list(items), seen, pop_ord, candidate_k)

    res = evaluate(recs, ground_truth, K)
    fmt("BPRMF", res, 0)
    return res, recs, scores_store
