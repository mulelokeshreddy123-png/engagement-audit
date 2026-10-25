"""End-to-end runner for the paper's analysis.

Usage:

    python -m pipeline.run_all --cases tvk=path/to/tvk.csv aiadmk=path/to/aiadmk.csv \
        --out results/ [--dmk path/to/dmk.csv] [--election 2026-04-23]

For every case this runs, in order:

  1. data loading + schema validation
  2. provenance audit (accounts, volumes, ranges, engagement gap)
  3. feature engineering (55-candidate set, log1p target)
  4. shuffled 80/20 and chronological 80/20 splits
  5. five-block walk-forward with bootstrap CIs and naive baselines
  6. block-count sensitivity (4 and 6 blocks, Random Forest)
  7. recency-exclusion check (Case 1 / TVK)
  8. per-account refits on the hard block (Case 1 / TVK)
  9. diagnostics: KS first-vs-second half, PELT change points, SHAP attribution

All results are written to the output directory as CSV tables; the weekly
series figure is written as PNG.  The numbers printed and saved are computed
from the data, not hard-coded.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import audit as audit_mod
from . import data as data_mod
from . import diagnostics as diag_mod
from . import features as feat_mod
from . import validate as val_mod
from .models import get_models


def _save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {path}")


def run_case(name, path, out_dir, election_date, models, do_diagnostics=True):
    print(f"\n=== CASE: {name} ===")
    df = data_mod.load_data(path)
    print("  loaded", len(df), "posts")

    # 1-2. provenance audit
    aud = audit_mod.run_audit(df)
    df = aud["data"]
    _save(aud["accounts"], os.path.join(out_dir, name, "audit_accounts.csv"))
    print("  accounts:", len(aud["accounts"]), "| primary:", aud["primary_account"],
          "| gap:", aud["gap"])

    # 3. features
    feats = feat_mod.build_features(df, election_date=election_date)
    target_col = "y"
    X = feats.drop(columns=[target_col])
    y = feats[target_col].values.astype(float)
    print("  candidate features:", X.shape[1])

    models_ = get_models(seed=42)
    # subset to models actually available
    models_ = {k: v for k, v in models_.items() if k in models}

    # 4. shuffled + chronological splits
    shuf = val_mod.shuffled_split(X, y, models_)
    chron = val_mod.chronological_holdout(X, y, models_)
    _save(pd.DataFrame([{"split": "shuffled", **shuf}, {"split": "chronological", **chron}]),
          os.path.join(out_dir, name, "split_r2.csv"))
    print("  shuffled R2:", {k: round(v, 3) for k, v in shuf.items()})
    print("  chronological R2:", {k: round(v, 3) for k, v in chron.items()})

    # 5. five-block walk-forward
    wf = val_mod.walk_forward(X, y, models_, n_blocks=5)
    wf_rows = []
    for row in wf:
        flat = {"transition": row["transition"], "n_train": row["n_train"],
                "n_test": row["n_test"]}
        for k, v in row.items():
            if k in ("transition", "n_train", "n_test", "train_range", "test_range"):
                continue
            if isinstance(v, tuple):
                flat[f"{k}_lo"], flat[f"{k}_hi"] = round(v[0], 3), round(v[1], 3)
            else:
                flat[k] = round(v, 3) if isinstance(v, float) else v
        wf_rows.append(flat)
    _save(pd.DataFrame(wf_rows), os.path.join(out_dir, name, "walk_forward_5.csv"))
    print("  walk-forward (5 blocks):")
    for row in wf:
        print("   ", row["transition"], {k: round(v, 3) for k, v in row.items()
                                         if k not in ("train_range", "test_range", "transition")
                                         and not isinstance(v, tuple)})

    # 6. block-count sensitivity (RF)
    sens = val_mod.block_sensitivity(X, y, models_, n_blocks_list=(4, 6))
    sens_rows = []
    for nb, trans in sens.items():
        for row in trans:
            sens_rows.append({"blocks": nb, "transition": row["transition"],
                              "RF": round(row["RF"], 3) if isinstance(row["RF"], float) else row["RF"]})
    _save(pd.DataFrame(sens_rows), os.path.join(out_dir, name, "block_sensitivity.csv"))

    # 7. recency-exclusion check (Case 1 / TVK)
    if name.lower() == "tvk":
        rec = val_mod.recency_exclusion(X, y, models_, df["timestamp"])
        _save(pd.DataFrame([{"buffer_days": b, **row} for b, row in rec.items() if row]),
              os.path.join(out_dir, name, "recency_exclusion.csv"))
        print("  recency-exclusion:", {b: {k: round(v, 3) for k, v in row.items()}
                                       for b, row in rec.items() if row})

        # 8. per-account refits on the hard (final) block
        n = len(y)
        block_size = n // 5
        hard_mask = np.zeros(n, bool)
        hard_mask[4 * block_size:] = True
        refits = val_mod.per_account_refits(X, y, df["account_id"], models_, hard_mask)
        if refits:
            _save(pd.DataFrame([{"account": k, **v} for k, v in refits.items()]),
                  os.path.join(out_dir, name, "per_account_refits.csv"))
            print("  per-account refits:", {k: {kk: round(vv, 3) if isinstance(vv, float) else vv
                                                 for kk, vv in v.items()} for k, v in refits.items()})

    # 9. diagnostics
    if do_diagnostics:
        ks = diag_mod.ks_first_second_half(y, X, ["is_primary_account", "post_index",
                                                  "days_to_election", "hashtag_count"])
        _save(pd.DataFrame([{"feature": k, "stat": v["stat"], "p": v["p"]}
                            for k, v in ks.items()]),
              os.path.join(out_dir, name, "ks_diagnostics.csv"))
        print("  KS (first vs second half):",
              {k: f"{v['stat']:.3f} (p={v['p']:.1e})" for k, v in ks.items()})

        # weekly series for PELT + figure
        weekly = df.set_index("timestamp")["engagement"].resample("W").mean()
        try:
            cps = diag_mod.pelt_change_points(weekly)
            print("  PELT change points (weekly mean engagement):", cps)
            _save(pd.DataFrame({"change_point_index": cps}),
                  os.path.join(out_dir, name, "pelt_change_points.csv"))
        except Exception as exc:  # pragma: no cover
            print("  PELT skipped:", exc)

        # SHAP attribution on the chronological 80/20 split
        try:
            n_test = int(len(y) * 0.2)
            selected = __import__("pipeline.selection", fromlist=["select_features"]).select_features(
                X.iloc[:-n_test], y[:-n_test])
            shap_res = diag_mod.shap_attribution(X.iloc[:-n_test][selected], y[:-n_test],
                                                 X.iloc[-n_test:][selected])
            top = sorted(shap_res["top1_counts"].items(), key=lambda kv: -kv[1])[:5]
            print("  SHAP top-1 counts:", top)
            _save(pd.DataFrame([{"feature": k, "top1_count": v,
                                 "mean_abs_shap": shap_res["mean_abs_shap"][k]}
                                for k, v in shap_res["top1_counts"].items()]),
                  os.path.join(out_dir, name, "shap_attribution.csv"))
        except Exception as exc:  # pragma: no cover
            print("  SHAP skipped:", exc)

        # weekly series figure
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(weekly.index, weekly.values, lw=1.2)
        ax.set_title(f"{name}: weekly mean engagement")
        ax.set_ylabel("mean engagement")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name, "weekly_series.png"), dpi=150)
        plt.close(fig)

    return {"case": name, "n_posts": len(df), "candidate_features": X.shape[1]}


def main():
    ap = argparse.ArgumentParser(description="Engagement forecasting pipeline")
    ap.add_argument("--cases", nargs="+", required=True,
                    help="case=path pairs, e.g. tvk=data/tvk.csv aiadmk=data/aiadmk.csv")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--election", default="2026-04-23", help="election date")
    ap.add_argument("--models", nargs="+", default=["RF", "XGB", "LGBM", "Ridge"],
                    help="models to run")
    ap.add_argument("--no-diagnostics", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    models = get_models(seed=42)
    summary = []
    for pair in args.cases:
        name, path = pair.split("=", 1)
        summary.append(run_case(name, path, args.out, args.election, args.models,
                                do_diagnostics=not args.no_diagnostics))
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nDone. Results in", args.out)


if __name__ == "__main__":
    main()
