# Engagement Forecasting Pipeline

Supplementary code for:

> **Engagement Forecasting Degrades Near an Election: A Provenance-Audited,
> Two-Case Temporal Validation Study of Tamil Nadu Party Instagram
> Operations** — Lokesh Reddy Mule

This package implements, end to end, the analysis described in the paper.
Every number in the paper is produced by running this code on the released
post-level datasets.  Nothing is hard-coded: the audit, features, models,
validation, baselines, and diagnostics all compute from the data.

## What it does

1. **Provenance audit** (Definition 1): groups records by the raw account
   identifier, reports volume / temporal range / engagement per account, the
   cross-account engagement gap, and the primary-account flag.
2. **Feature engineering** (Table VI): the 55-candidate set (temporal,
   media, text, content, account) with a `log1p(likes + comments)` target.
3. **Feature selection** (Section IV-C): constant-feature removal,
   correlation clustering (|r| > 0.85), and permutation-importance pruning,
   strictly inside each training split.
4. **Models** (Section IV-D): Random Forest (400 trees, `min_samples_leaf=3`),
   XGBoost (400 rounds, depth 5, lr 0.05, subsample 0.9), LightGBM (400
   rounds, 31 leaves, lr 0.05, subsample 0.9), Ridge (alpha 1.0).  Fixed
   hyperparameters, seed 42, no tuning.
5. **Validation** (Sections IV-E, V-VII): shuffled 80/20, chronological
   80/20, N-block walk-forward with bootstrap 95% CIs, block-count
   sensitivity (4/5/6), naive rolling-mean baselines (k = 10, 30),
   recency-exclusion check, per-account refits.
6. **Diagnostics** (Section VIII): KS first-vs-second-half, PELT
   change-point detection, SHAP attribution with bootstrap ranking.

All R2 values are `1 - SSE/SST` on the log1p-transformed evaluation set.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# smoke test with synthetic data (NOT the paper's dataset)
python -m pipeline.make_example_data --out example_data --posts 600

# full analysis on the real released datasets
python -m pipeline.run_all \
    --cases tvk=data/tvk.csv aiadmk=data/aiadmk.csv \
    --out results --election 2026-04-23
```

Results are written to `results/<case>/` as CSV tables (audit, split R2,
walk-forward, block sensitivity, recency exclusion, per-account refits, KS,
PELT, SHAP) plus a weekly-series PNG and a `summary.json`.

## Data format

See `data_format.md`.  The minimal five-column release is
`post_id, account_id, timestamp, media_type, likes, comments`; caption-derived
feature columns are released hashed alongside it.

## Notes on the naive baseline

The rolling-mean persistence forecaster predicts, for each test post, the
mean of the `log1p`-engagement values of the `k` most recent posts strictly
before it that are **not** part of the test block, so the baseline never
touches the test block's own outcomes.  The window advances with the test
post and uses only information available at prediction time.

## Reproducibility

Exact package versions used for the paper are pinned in the released
environment lockfile; `requirements.txt` lists the package set.
