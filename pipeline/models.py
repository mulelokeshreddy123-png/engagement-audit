"""Model configurations (Section IV-D of the paper).

Four learners with fixed, documented hyperparameters, held constant across
both cases, all blocks, and all experiments.  No hyperparameters are tuned;
the configurations were fixed a priori, so there is no tuning objective and
no validation-based model selection (leakage-safe by construction).

All random components use seed 42.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None


def get_models(seed: int = 42) -> dict:
    """Return {name: model} for the four learners."""
    models = {
        "RF": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, random_state=seed, n_jobs=-1
        ),
        "Ridge": Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=seed))]
        ),
    }
    if XGBRegressor is not None:
        models["XGB"] = XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=-1
        )
    if LGBMRegressor is not None:
        models["LGBM"] = LGBMRegressor(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            subsample=0.9, random_state=seed, n_jobs=-1, verbose=-1
        )
    return models
