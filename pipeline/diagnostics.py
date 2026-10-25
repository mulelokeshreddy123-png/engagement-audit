"""Diagnostics (Section VIII of the paper).

  A. Outcome non-stationarity: two-sample Kolmogorov-Smirnov tests comparing
     the first and second halves of each series on the target and on key
     features.
  B. Change-point localization: PELT (ruptures) on weekly series of mean
     engagement and of the primary-account posting share.
  E. Attribution stability: SHAP TreeExplainer on the Random Forest, with a
     bootstrap procedure (100 resamples) counting how often each feature
     ranks first by mean absolute SHAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

try:
    import ruptures
except Exception:  # pragma: no cover
    ruptures = None

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

from sklearn.ensemble import RandomForestRegressor


def ks_first_second_half(y: np.ndarray, feature_df: pd.DataFrame, feature_names):
    """Two-sample KS between the first and second halves of the series."""
    n = len(y)
    half = n // 2
    out = {"y": _ks(y[:half], y[half:])}
    for f in feature_names:
        if f in feature_df.columns:
            out[f] = _ks(feature_df[f].values[:half], feature_df[f].values[half:])
    return out


def _ks(a, b):
    stat, p = stats.ks_2samp(a, b)
    return {"stat": float(stat), "p": float(p)}


def pelt_change_points(weekly_series: pd.Series, min_size: int = 2, pen: float = 5.0):
    """PELT change-point detection on a weekly series.  Returns the indices
    (positions) of detected change points."""
    if ruptures is None:
        raise ImportError("ruptures is required for PELT change-point detection")
    values = weekly_series.fillna(0.0).values.reshape(-1, 1).astype(float)
    algo = ruptures.Pelt(model="rbf", min_size=min_size, jump=1).fit(values)
    bkps = algo.predict(pen=pen)
    # bkps includes the last index; drop it to get true change points
    return [int(b) for b in bkps[:-1]]


def shap_attribution(X_train: pd.DataFrame, y_train: np.ndarray,
                     X_test: pd.DataFrame, n_resamples: int = 100, seed: int = 42):
    """SHAP TreeExplainer on a Random Forest; bootstrap the test rows and
    count how often each feature ranks first by mean absolute SHAP."""
    if shap is None:
        raise ImportError("shap is required for attribution analysis")
    rng = np.random.default_rng(seed)
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    explainer = shap.TreeExplainer(rf)

    counts = {c: 0 for c in X_test.columns}
    mean_abs = {c: [] for c in X_test.columns}
    n = len(X_test)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sh = explainer.shap_values(X_test.iloc[idx])
        mabs = np.abs(sh).mean(axis=0)
        top = X_test.columns[int(np.argmax(mabs))]
        counts[top] += 1
        for c, v in zip(X_test.columns, mabs):
            mean_abs[c].append(float(v))

    return {
        "top1_counts": counts,
        "mean_abs_shap": {c: float(np.mean(v)) for c, v in mean_abs.items()},
    }
