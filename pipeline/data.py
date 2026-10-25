"""Data loading and schema validation.

Expected input format (one CSV per case, UTF-8):

    post_id,account_id,timestamp,media_type,likes,comments

  post_id     anonymized post identifier (string)
  account_id  raw account identifier as collected (string) -- used ONLY for
              the provenance audit and the account feature
  timestamp   publication time, ISO 8601 or 'YYYY-MM-DD HH:MM:SS'
  media_type  one of Image, Video, Sidecar
  likes       integer like count at collection time
  comments    integer comment count at collection time

Optional additional columns:

  caption     raw caption text (used to build the text features).  If the
              caption-derived feature columns are released precomputed
              (hashed), the loader also accepts them directly:
              cap_len_chars, cap_len_words, hashtag_count, mention_count,
              tamil_script, top25_hashtag_*, top10_mention_*.

  video_duration  video duration in seconds (0 for non-video).

If 'likes' and 'comments' are absent but an 'engagement' column is present,
engagement is read directly from it.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

# Election date used by the temporal features (announced 15 March 2026;
# polling 23 April 2026).  Kept as a module constant so it can be changed.
ELECTION_DATE = dt.date(2026, 4, 23)

REQUIRED_COLUMNS = ["post_id", "account_id", "timestamp", "media_type"]
ENGAGEMENT_PAIRS = (("likes", "comments"), ("engagement",))


def load_data(path: str) -> pd.DataFrame:
    """Load and validate a case CSV.  Returns a clean DataFrame with a
    datetime 'timestamp', an 'engagement' column (likes + comments), and
    a 'post_age_days' column (days from publication to the end of the
    collection window, i.e. the last observed timestamp)."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    for pair in ENGAGEMENT_PAIRS:
        if all(c in df.columns for c in pair):
            if len(pair) == 2:
                df["engagement"] = df[list(pair)].astype(float).sum(axis=1)
            else:
                df["engagement"] = df[pair[0]].astype(float)
            break
    else:
        raise ValueError("need 'likes'+'comments' or an 'engagement' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Post age at collection: time from publication to the end of the
    # collection window (the last observed publication time is a proxy for
    # the collection date).  Used by the age diagnostics.
    collection_end = df["timestamp"].max()
    df["post_age_days"] = (collection_end - df["timestamp"]).dt.days

    df["media_type"] = df["media_type"].astype(str).str.strip()
    df["media_type"] = df["media_type"].where(
        df["media_type"].isin(["Image", "Video", "Sidecar"]), "Image"
    )
    return df


def describe_data(df: pd.DataFrame, name: str) -> dict:
    """Summary statistics used by the provenance audit and Table I."""
    return {
        "case": name,
        "n_posts": int(len(df)),
        "date_min": df["timestamp"].min().date().isoformat(),
        "date_max": df["timestamp"].max().date().isoformat(),
        "posts_per_day": round(len(df) / max((df["timestamp"].max() - df["timestamp"].min()).days, 1), 2),
    }
