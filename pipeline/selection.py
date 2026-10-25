"""Feature selection and decorrelation (Section IV-C of the paper).

Selection happens strictly inside each training split, never on test data:

  1. Drop constant / near-constant features.
  2. Cluster features with absolute pairwise correlation above 0.85 and keep
     only the higher-variance representative of each cluster.
  3. Compute permutation importance on the training split with a random
     forest (three repeats) and drop features with non-positive importance.

The function returns a list of selected feature column names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


def select_features(X_train: pd.DataFrame, y_train: np.ndarray, seed: int = 42) -> list:
    """Return the selected feature column names for a training split."""
    # 1: drop constant / near-constant features
    nunique = X_train.nunique()
    keep = nunique[nunique > 1].index.tolist()
    X = X_train[keep].copy()

    # 2: correlation clustering (|r| > 0.85), keep the higher-variance rep
    corr = X.corr().abs()
    representatives = []
    seen = set()
    for col in X.columns:
        if col in seen:
            continue
        cluster = [c for c in X.columns
                   if c not in seen and c != col and corr.loc[col, c] > 0.85]
        if cluster:
            rep = max([col] + cluster, key=lambda c: X[c].var())
        else:
            rep = col
        representatives.append(rep)
        seen.update(cluster)
        seen.add(col)
    # Guard against overlapping correlation clusters producing a repeated
    # representative (e.g. post_index) on newer pandas/scikit-learn versions.
    representatives = list(dict.fromkeys(representatives))
    X = X[representatives]

    # 3: permutation importance (Random Forest, three repeats)
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    rf.fit(X, y_train)
    perm = permutation_importance(rf, X, y_train, n_repeats=3,
                                  random_state=seed, scoring="neg_mean_squared_error")
    imp = pd.Series(perm.importances_mean, index=X.columns)
    selected = [c for c in X.columns if imp[c] > 0]

    # Always keep the account feature and the sequential time index if present.
    for mandatory in ["is_primary_account", "post_index"]:
        if mandatory in X.columns and mandatory not in selected:
            selected.append(mandatory)

    return selected
