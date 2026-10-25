"""Validation protocols (Section IV-E and Sections V-VII of the paper).

All R2 values are computed as 1 - SSE/SST on the log1p-transformed
evaluation set, where SSE is the sum of squared residuals of the model and
SST is the sum of squared deviations of the evaluation outcomes from their
own mean.  Negative values therefore indicate performance worse than
predicting the evaluation-set mean.

Protocols implemented here:

  (1) shuffled 80/20 split (interpolation)
  (2) chronological 80/20 holdout (forecasting)
  (3) N-block walk-forward validation (train block i, predict block i+1),
      with bootstrap 95% confidence intervals over test rows
  (4) naive rolling-mean persistence baselines (k = 10, 30)
  (5) recency-exclusion check (drop posts within b days of collection end)
  (6) per-account refits (fit per account, test on that account's posts)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import get_models
from .selection import select_features


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    if sst == 0:
        return float("nan")
    return 1.0 - sse / sst


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_resamples: int = 100,
                 seed: int = 42) -> tuple:
    """Bootstrap 95% CI of R2 by resampling test rows with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        vals.append(r2(y_true[idx], y_pred[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _fit_predict(model, X_tr, y_tr, X_te):
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def _rolling_mean_baseline(y: np.ndarray, test_mask: np.ndarray,
                           k: int) -> np.ndarray:
    """Walk-forward rolling-mean persistence baseline.

    For a test post at position t, predict the mean of the log1p-engagement
    values of the k most recent posts strictly before t that are NOT part of
    the test block (so the baseline never touches the test block's own
    outcomes).  The window advances with t and uses only information
    available at prediction time.
    """
    preds = np.full(int(test_mask.sum()), np.nan)
    test_idx = np.where(test_mask)[0]
    for out_i, t in enumerate(test_idx):
        history = np.where(~test_mask & (np.arange(len(y)) < t))[0]
        if len(history) == 0:
            preds[out_i] = np.mean(y[:t]) if t > 0 else 0.0
        else:
            recent = history[-k:]
            preds[out_i] = np.mean(y[recent])
    return preds


def shuffled_split(X, y, models, seed=42, test_size=0.2):
    """(1) Shuffled 80/20 interpolation split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    out = {}
    for name, model in models.items():
        y_pred = _fit_predict(model, X.iloc[train_idx], y[train_idx], X.iloc[test_idx])
        out[name] = r2(y[test_idx], y_pred)
    return out


def chronological_holdout(X, y, models, test_size=0.2):
    """(2) Chronological 80/20 holdout: train on the first 80% by time."""
    n_test = int(len(y) * test_size)
    train_idx = np.arange(len(y) - n_test)
    test_idx = np.arange(len(y) - n_test, len(y))
    out = {}
    for name, model in models.items():
        y_pred = _fit_predict(model, X.iloc[train_idx], y[train_idx], X.iloc[test_idx])
        out[name] = r2(y[test_idx], y_pred)
    return out


def walk_forward(X, y, models, n_blocks=5, n_resamples=100, seed=42,
                 baselines=(10, 30)):
    """(3) N-block walk-forward validation.

    The series is split into `n_blocks` contiguous, non-overlapping blocks
    of equal post count in chronological order.  Models train on block i and
    predict block i+1, giving n_blocks - 1 transitions.  Returns a list of
    per-transition dicts with model R2, bootstrap CIs, and naive-baseline R2.
    """
    n = len(y)
    block_size = n // n_blocks
    bounds = [i * block_size for i in range(n_blocks)] + [n]
    transitions = []
    for i in range(n_blocks - 1):
        tr = np.arange(bounds[i], bounds[i + 1])
        te = np.arange(bounds[i + 1], bounds[i + 2])
        X_tr, y_tr = X.iloc[tr], y[tr]
        X_te, y_te = X.iloc[te], y[te]

        selected = select_features(X_tr, y_tr, seed=seed)
        X_tr_s = X_tr[selected]
        X_te_s = X_te[selected]

        row = {"transition": f"T{i+1}", "train_range": (tr[0], tr[-1]),
               "test_range": (te[0], te[-1]), "n_train": int(len(tr)),
               "n_test": int(len(te))}
        for name, model in models.items():
            y_pred = _fit_predict(model, X_tr_s, y_tr, X_te_s)
            r = r2(y_te, y_pred)
            lo, hi = bootstrap_ci(y_te, y_pred, n_resamples=n_resamples, seed=seed)
            row[name] = r
            row[f"{name}_ci"] = (lo, hi)

        # naive baselines (never touch the test block's outcomes)
        te_mask = np.zeros(n, bool)
        te_mask[te] = True
        for k in baselines:
            y_pred = _rolling_mean_baseline(y, te_mask, k)
            row[f"naive_k{k}"] = r2(y_te, y_pred)
        transitions.append(row)
    return transitions


def block_sensitivity(X, y, models, n_blocks_list=(4, 6), seed=42):
    """(4) Block-count sensitivity with Random Forest only."""
    out = {}
    rf = {k: v for k, v in models.items() if k == "RF"}
    for nb in n_blocks_list:
        out[nb] = walk_forward(X, y, rf, n_blocks=nb, seed=seed)
    return out


def recency_exclusion(X, y, models, timestamps, buffers=(14, 30, 60), test_size=0.2):
    """(5) Recency-exclusion check: drop posts within `buffer` days of the
    collection end, then rerun the chronological 80/20 holdout."""
    collection_end = timestamps.max()
    out = {}
    for buf in buffers:
        keep = timestamps <= (collection_end - pd.Timedelta(days=buf))
        idx = np.where(keep)[0]
        if len(idx) < 30:
            out[buf] = None
            continue
        X_k = X.iloc[idx]
        y_k = y[idx]
        n_test = int(len(y_k) * test_size)
        tr = np.arange(len(y_k) - n_test)
        te = np.arange(len(y_k) - n_test, len(y_k))
        row = {}
        for name, model in models.items():
            y_pred = _fit_predict(model, X_k.iloc[tr], y_k[tr], X_k.iloc[te])
            row[name] = r2(y_k[te], y_pred)
        out[buf] = row
    return out


def per_account_refits(X, y, accounts, models, hard_test_mask):
    """(6) Per-account refits: fit separately on each account's training
    posts and test on that account's posts in the hard block."""
    out = {}
    for acct in accounts.unique():
        mask = (accounts == acct).values
        tr = mask & ~hard_test_mask
        te = mask & hard_test_mask
        if tr.sum() < 20 or te.sum() == 0:
            continue
        row = {"account": acct, "n_train": int(tr.sum()), "n_test": int(te.sum())}
        for name, model in models.items():
            y_pred = _fit_predict(model, X.iloc[tr], y[tr], X.iloc[te])
            row[name] = r2(y[te], y_pred)
        out[acct] = row
    return out
