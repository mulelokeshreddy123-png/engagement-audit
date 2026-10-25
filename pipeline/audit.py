"""Provenance audit (Definition 1 in the paper).

Groups records by the raw account identifier and reports, for every group,
its volume, temporal range, and engagement distribution.  Also computes the
cross-account engagement gap and a flag marking the high-engagement
"primary" account (the leader or official account).
"""

from __future__ import annotations

import pandas as pd


def run_audit(df: pd.DataFrame) -> dict:
    """Audit a dataset.  Returns a dict with per-account statistics and a
    DataFrame with an 'is_primary_account' column added to the input."""
    groups = []
    for account, g in df.groupby("account_id"):
        groups.append(
            {
                "account_id": account,
                "n": int(len(g)),
                "date_min": g["timestamp"].min().date().isoformat(),
                "date_max": g["timestamp"].max().date().isoformat(),
                "median_engagement": float(g["engagement"].median()),
                "mean_engagement": float(g["engagement"].mean()),
                "share": float(len(g) / len(df)),
            }
        )
    accounts = pd.DataFrame(groups).sort_values("n", ascending=False)

    # Primary account = the highest median-engagement account with a
    # non-trivial share of the dataset (>= 1%).  This matches the paper's
    # "leader or official account" definition.
    substantial = accounts[accounts["share"] >= 0.01]
    if len(substantial) > 1:
        primary = substantial.sort_values("median_engagement", ascending=False).iloc[0]["account_id"]
    elif len(accounts) > 0:
        primary = accounts.iloc[0]["account_id"]
    else:
        primary = None

    df = df.copy()
    df["is_primary_account"] = (df["account_id"] == primary).astype(int)

    # Engagement gap between the primary (high-engagement) account and the
    # largest-volume account: reported as high/low, matching the paper's
    # "30.8x median gap" framing.
    gap = None
    if len(accounts) >= 2:
        big = accounts.iloc[0]
        prim_row = accounts[accounts["account_id"] == primary]
        if len(prim_row) == 1:
            p = prim_row.iloc[0]
            med_hi = max(p["median_engagement"], big["median_engagement"])
            med_lo = min(p["median_engagement"], big["median_engagement"])
            mean_hi = max(p["mean_engagement"], big["mean_engagement"])
            mean_lo = min(p["mean_engagement"], big["mean_engagement"])
            gap = {
                "median_x": round(med_hi / med_lo, 1) if med_lo > 0 else None,
                "mean_x": round(mean_hi / mean_lo, 1) if mean_lo > 0 else None,
            }

    return {"accounts": accounts, "primary_account": primary, "gap": gap, "data": df}
