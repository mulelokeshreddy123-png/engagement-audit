"""Generate synthetic example data for smoke-testing the pipeline.

This produces FAKE data that matches the schema the pipeline expects.  It is
intended ONLY to verify that the code runs end to end; it is NOT the paper's
dataset and its numbers are meaningless for the paper's claims.  Replace the
generated files with the real released datasets before running the analysis.

Usage:  python -m pipeline.make_example_data --out example_data --posts 600
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def make_case(name: str, start: str, end: str, n_posts: int, seed: int,
              accounts=(("primary", 0.2, 25000.0), ("wing", 0.8, 700.0))) -> pd.DataFrame:
    """Two-account synthetic case: a low-volume high-engagement primary
    account and a high-volume low-engagement wing account."""
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    span_days = (end_ts - start_ts).days
    times = sorted(start_ts + pd.to_timedelta(rng.uniform(0, span_days, n_posts), unit="D"))

    rows = []
    for i, t in enumerate(times):
        r = rng.random()
        acct = accounts[0][0] if r < accounts[0][1] else accounts[1][0]
        base = accounts[0][2] if acct == accounts[0][0] else accounts[1][2]
        engagement = float(rng.lognormal(mean=np.log(base), sigma=0.9))
        media = rng.choice(["Image", "Video", "Sidecar"], p=[0.6, 0.3, 0.1])
        likes = engagement * 0.9
        comments = engagement * 0.1
        caption = "#campaign #tamilnadu" if rng.random() < 0.5 else "vote"
        rows.append({"post_id": f"{name}_{i:05d}", "account_id": acct,
                     "timestamp": t, "media_type": media, "likes": int(likes),
                     "comments": int(comments), "caption": caption})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="example_data")
    ap.add_argument("--posts", type=int, default=600)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cases = {
        "tvk": ("2024-03-07", "2026-06-18", 11),
        "aiadmk": ("2025-08-01", "2026-08-06", 12),
    }
    for name, (start, end, seed) in cases.items():
        df = make_case(name, start, end, args.posts, seed)
        df.to_csv(f"{args.out}/{name}.csv", index=False)
        print("wrote", f"{args.out}/{name}.csv", len(df))


if __name__ == "__main__":
    main()
