# Disqualification Cautions

Use this checklist before every Dacon LG Aimers Phase 2 submission.

## Core Rule

Each test row must be predicted independently.

For a row A, the prediction may use only:

- input values inside row A
- row-wise derived features from row A
- official training data
- models, statistics, encoders, calibrators, and lookup maps fitted only from official training data

The prediction for row A must be the same when:

- `test.csv` contains only row A
- `test.csv` contains the full evaluation set

## Forbidden In `script.py`

Never use evaluation/test-set-wide information:

- `test.groupby(...)`
- `test.value_counts(...)`
- test-set mean, std, quantile, rank, frequency, or distribution
- rolling, expanding, cumulative, lag, or previous-row features over `test.csv`
- player/team/month/game aggregates computed from `test.csv`
- target encoding fitted on `test.csv`
- calibrating or shifting predictions based on all test rows
- using other rows that look earlier in time inside the same `test.csv`

## Allowed In `script.py`

Allowed operations:

- row-wise arithmetic and boolean features
- `asof_*` official columns
- missing-value fills learned from train
- category mappings learned from train
- target-encoding maps learned from train
- probability calibrators learned from train/validation seasons
- using `season` value from the current row with a train-fitted year trend
- model inference with saved local artifacts

## Required Pre-Submission Checks

Run these checks before uploading:

- zip top-level contains exactly `model/`, `script.py`, and `requirements.txt`
- `script.py` creates `output/submission.csv`
- probabilities are finite and clipped to `[0, 1]`
- row independence test has `max_abs_diff == 0.0` or only floating-point noise
- grep `script.py` for forbidden patterns:
  - `groupby`
  - `rolling`
  - `rank`
  - `value_counts`
  - `qcut`
  - `quantile`
  - `.mean(`
  - `.std(`
  - `.fit(`

## Current Submission Note

Current submission uses:

- calibrated baseline RF when `rf.pkl` loads
- calibrated custom model only as fallback
- `ensemble_weight = 1.0`, meaning RF-only when RF is available

The RF and custom calibrators are fitted from official training data only.
