"""Feature engineering (Table VI in the paper).

Builds the candidate feature set identically for every case:

  Temporal : hour, weekday, month, post_index, days_since_first,
             days_to_election, pre42_flag, post_flag, gap_prev
  Media    : media_type one-hot (Image/Video/Sidecar), video_duration
  Text     : cap_len_chars, cap_len_words, hashtag_count, mention_count,
             tamil_script
  Content  : top25_hashtag_* indicators, top10_mention_* indicators
  Account  : is_primary_account

No feature is derived from the engagement outcome, so none encodes the
target.  The election schedule (announced 15 March 2026, polling 23 April
2026) is known at prediction time, so the election flags are not look-ahead
leakage.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .data import ELECTION_DATE

TAMIL_RANGES = (
    (0x0B80, 0x0BFF),  # Tamil block
    (0x1CD0, 0x1CFF),  # Vedic extensions (used for Tamil)
    (0x1CD3, 0x1CD3),
)


def _is_tamil_script(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in TAMIL_RANGES):
            return 1
    return 0


def _hashtags(text: str) -> list:
    if not isinstance(text, str):
        return []
    return re.findall(r"#([\w]+)", text)


def _mentions(text: str) -> list:
    if not isinstance(text, str):
        return []
    return re.findall(r"@([\w]+)", text)


def build_features(df: pd.DataFrame, election_date=None) -> pd.DataFrame:
    """Return a feature matrix (one row per post) plus the target series.

    The returned DataFrame has one row per post, indexed like `df`, with
    the candidate features as columns and a 'y' column holding
    log1p(likes + comments).
    """
    if election_date is None:
        election_date = ELECTION_DATE

    ts = df["timestamp"]
    first = ts.min()

    feats = pd.DataFrame(index=df.index)
    feats["hour"] = ts.dt.hour.astype(int)
    feats["weekday"] = ts.dt.weekday.astype(int)
    feats["month"] = ts.dt.month.astype(int)
    feats["post_index"] = range(len(df))
    feats["days_since_first"] = (ts - first).dt.days.astype(int)
    feats["days_to_election"] = (pd.Timestamp(election_date) - ts).dt.days.astype(int)
    feats["pre42_flag"] = ((feats["days_to_election"] > 0) & (feats["days_to_election"] <= 42)).astype(int)
    feats["post_flag"] = (feats["days_to_election"] < 0).astype(int)
    feats["gap_prev"] = ts.diff().dt.days.fillna(0).astype(int)

    media = pd.get_dummies(df["media_type"], prefix="media")
    for col in ["media_Image", "media_Video", "media_Sidecar"]:
        if col not in media.columns:
            media[col] = 0
    feats = pd.concat([feats, media[["media_Image", "media_Video", "media_Sidecar"]]], axis=1)
    feats["video_duration"] = df.get("video_duration", pd.Series(0.0, index=df.index)).fillna(0.0)

    if "caption" in df.columns:
        caps = df["caption"].fillna("")
        feats["cap_len_chars"] = caps.str.len().astype(int)
        feats["cap_len_words"] = caps.str.split().str.len().astype(int)
        feats["hashtag_count"] = caps.map(lambda c: len(_hashtags(c))).astype(int)
        feats["mention_count"] = caps.map(lambda c: len(_mentions(c))).astype(int)
        feats["tamil_script"] = caps.map(_is_tamil_script).astype(int)

        all_hashtags = [h for c in caps for h in _hashtags(c)]
        top_hashtags = pd.Series(all_hashtags).value_counts().head(25).index.tolist()
        for h in top_hashtags:
            feats[f"top25_hashtag_{h}"] = caps.map(lambda c: int(h in _hashtags(c)))

        all_mentions = [m for c in caps for m in _mentions(c)]
        top_mentions = pd.Series(all_mentions).value_counts().head(10).index.tolist()
        for m in top_mentions:
            feats[f"top10_mention_{m}"] = caps.map(lambda c: int(m in _mentions(c)))
    else:
        # Precomputed caption-derived columns (released hashed alongside the
        # minimal columns).
        for col in ["cap_len_chars", "cap_len_words", "hashtag_count",
                    "mention_count", "tamil_script"]:
            feats[col] = df.get(col, 0)
        for col in df.columns:
            if col.startswith("top25_hashtag_") or col.startswith("top10_mention_"):
                feats[col] = df[col]

    feats["is_primary_account"] = df["is_primary_account"]

# Remove duplicate generated feature names before scikit-learn fitting.
    feats = feats.loc[:, ~feats.columns.duplicated()].copy()

    feats["y"] = np.log1p(df["engagement"].values)

    return feats
