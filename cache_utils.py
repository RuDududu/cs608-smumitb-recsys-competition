# ============================================================
# cache_utils.py — Pickle cache save/load helpers
# ============================================================

import pickle
from pathlib import Path


def save_pkl(obj, path):
    """Serialise obj to path using highest pickle protocol."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    """
    Load pickled object from path.
    Returns None if file does not exist.
    """
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)
