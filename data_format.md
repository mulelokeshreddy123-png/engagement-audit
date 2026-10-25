# Released data format

The pipeline reads one UTF-8 CSV per case (TVK, AIADMK, DMK). Every row is
one public organizational post.

## Minimal five-column release

| column | type | meaning |
|---|-----------|---------|
| `post_id` | string | anonymized post identifier |
| `account_id` | string | raw account identifier as collected (used only for the provenance audit and the account feature) |
| `timestamp` | datetime | publication time (ISO 8601 or `YYYY-MM-DD HH:MM:SS`) |
| `media_type` | string | `Image`, `Video`, or `Sidecar` |
| `likes` | int | like count at collection time |
| `comments` | int | comment count at collection time |

If `likes` and `comments` are absent, a single `engagement` column
(`likes + comments`) is accepted instead.

## Caption-derived columns (released hashed alongside the minimal columns)

The full feature tables require caption-derived inputs. These are released
as additional columns (hashed where captions are retained):

* `caption` (optional): raw caption text, used to build `cap_len_chars`,
  `cap_len_words`, `hashtag_count`, `mention_count`, `tamil_script`, and the
  `top25_hashtag_*` / `top10_mention_*` indicators.  If `caption` is not
  present, the pipeline reads those columns directly if they are supplied.
* `video_duration` (optional): video duration in seconds (0 for non-video).

## Notes

* No follower-level or private data is included.
* `account_id` is retained only to perform the provenance audit; released
  post identifiers are anonymized.
* Engagement counts are snapshots taken at collection time; they carry no
  independent capture timestamp.  The pipeline derives `post_age_days` as
  the time from publication to the end of the collection window, which is
  used by the age diagnostics.
