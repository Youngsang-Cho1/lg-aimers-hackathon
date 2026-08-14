import argparse
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"

BASE_NUMERIC_COLS = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

LOW_CARD_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "balls_before",
    "strikes_before",
    "outs_before",
    "num_runners_on",
    "count_state",
    "hand_matchup",
    "runner_mask",
]

TE_SPECS = [
    {"name": "te_pitcher", "cols": ["pitcher_id"], "alpha": 250.0},
    {"name": "te_batter", "cols": ["batter_id"], "alpha": 300.0},
    {"name": "te_pitcher_batter_hand", "cols": ["pitcher_id", "batter_hand"], "alpha": 400.0},
    {"name": "te_batter_pitcher_hand", "cols": ["batter_id", "pitcher_hand"], "alpha": 450.0},
    {"name": "te_pitcher_count", "cols": ["pitcher_id", "count_state"], "alpha": 650.0},
    {"name": "te_batter_count", "cols": ["batter_id", "count_state"], "alpha": 750.0},
    {"name": "te_pitcher_base", "cols": ["pitcher_id", "base_state"], "alpha": 700.0},
    {"name": "te_pitcher_team", "cols": ["pitcher_team_id"], "alpha": 150.0},
    {"name": "te_batter_team", "cols": ["batter_team_id"], "alpha": 150.0},
    {"name": "te_team_matchup", "cols": ["pitcher_team_id", "batter_team_id"], "alpha": 180.0},
    {"name": "te_hand_count", "cols": ["hand_matchup", "count_state"], "alpha": 80.0},
    {"name": "te_base_count", "cols": ["base_state", "count_state"], "alpha": 90.0},
    {"name": "te_game_type", "cols": ["game_type"], "alpha": 80.0},
]

SMOOTH_SPECS = [
    ("asof_pitcher_success_rate", "asof_pitcher_n", 50.0),
    ("asof_pitcher_reverse_rate", "asof_pitcher_n", 100.0),
    ("asof_pitcher_middle_rate", "asof_pitcher_n", 100.0),
    ("asof_pitcher_ball_rate", "asof_pitcher_n", 100.0),
    ("asof_pitcher_strike_rate", "asof_pitcher_n", 100.0),
    ("asof_batter_success_rate", "asof_batter_n", 100.0),
    ("asof_batter_middle_rate", "asof_batter_n", 100.0),
    ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n", 100.0),
    ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n", 100.0),
    ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n", 100.0),
]


def sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def add_derived(df):
    out = df.copy()
    out["count_state"] = out["balls_before"].astype(str) + "_" + out["strikes_before"].astype(str)
    out["hand_matchup"] = out["pitcher_hand"].astype(str) + "_" + out["batter_hand"].astype(str)
    out["runner_mask"] = (
        out["runner_on_1b"].astype(np.int16)
        + 2 * out["runner_on_2b"].astype(np.int16)
        + 4 * out["runner_on_3b"].astype(np.int16)
    )
    out["is_full_count"] = ((out["balls_before"] == 3) & (out["strikes_before"] == 2)).astype(np.int8)
    out["is_two_strike"] = (out["strikes_before"] == 2).astype(np.int8)
    out["is_three_ball"] = (out["balls_before"] == 3).astype(np.int8)
    out["is_late_inning"] = (out["inning"] >= 7).astype(np.int8)
    out["is_extra_inning"] = (out["inning"] >= 10).astype(np.int8)
    out["is_risp"] = ((out["runner_on_2b"] == 1) | (out["runner_on_3b"] == 1)).astype(np.int8)
    out["is_loaded"] = (out["num_runners_on"] == 3).astype(np.int8)
    out["abs_score_diff_pitcher"] = out["score_diff_pitcher_team"].abs()
    out["abs_score_diff_home"] = out["score_diff_home"].abs()
    out["li_log1p"] = np.log1p(out["li"].clip(lower=0.0))
    out["pitcher_team_win_exp"] = np.where(
        out["top_bottom"].astype(str).eq("T"),
        out["home_win_expectancy"],
        out["away_win_expectancy"],
    )
    out["season_from_2024"] = out["season"] - 2024
    out["month_sin"] = np.sin(2.0 * np.pi * out["game_month"] / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out["game_month"] / 12.0)
    return out


def build_rate_priors(df):
    priors = {}
    for rate_col, _, _ in SMOOTH_SPECS:
        priors[rate_col] = float(df[rate_col].mean(skipna=True))
    return priors


def add_smoothed_rates(df, rate_priors):
    out = df.copy()
    for rate_col, n_col, m in SMOOTH_SPECS:
        prior = rate_priors[rate_col]
        rate = out[rate_col].fillna(prior).astype(float)
        n = out[n_col].fillna(0.0).astype(float).clip(lower=0.0)
        out[f"{rate_col}_smooth"] = (rate * n + prior * m) / (n + m)
    return out


def feature_numeric_cols():
    derived_cols = [
        "is_full_count",
        "is_two_strike",
        "is_three_ball",
        "is_late_inning",
        "is_extra_inning",
        "is_risp",
        "is_loaded",
        "abs_score_diff_pitcher",
        "abs_score_diff_home",
        "li_log1p",
        "pitcher_team_win_exp",
        "season_from_2024",
        "month_sin",
        "month_cos",
    ]
    smooth_cols = [f"{rate_col}_smooth" for rate_col, _, _ in SMOOTH_SPECS]
    te_cols = [spec["name"] for spec in TE_SPECS]
    return BASE_NUMERIC_COLS + derived_cols + smooth_cols + te_cols


def make_group_key_frame(df, cols):
    if len(cols) == 1:
        return df[cols[0]]
    return pd.MultiIndex.from_frame(df[cols])


def fit_te_map(df, cols, alpha, prior):
    agg = df.groupby(cols, dropna=False)[TARGET_COL].agg(["sum", "count"]).reset_index()
    values = (agg["sum"].to_numpy(dtype=float) + prior * alpha) / (agg["count"].to_numpy(dtype=float) + alpha)
    if len(cols) == 1:
        keys = agg[cols[0]].tolist()
    else:
        keys = list(map(tuple, agg[cols].itertuples(index=False, name=None)))
    return {k: float(v) for k, v in zip(keys, values)}


def apply_te_map(df, cols, mapping, default):
    n = len(df)
    if len(cols) == 1:
        vals = df[cols[0]].map(mapping).fillna(default).to_numpy(dtype=np.float32)
        return vals
    arrays = [df[c].to_numpy() for c in cols]
    return np.fromiter((mapping.get(tuple(key), default) for key in zip(*arrays)), dtype=np.float32, count=n)


def add_temporal_oof_te(df, specs, prior):
    out = df.copy()
    seasons = sorted(out["season"].unique())
    for spec in specs:
        out[spec["name"]] = np.float32(prior)
        cols = spec["cols"]
        alpha = spec["alpha"]
        group_cols = cols + ["season"]
        agg = out.groupby(group_cols, dropna=False)[TARGET_COL].agg(["sum", "count"]).reset_index()
        agg = agg.sort_values(cols + ["season"])
        agg["cum_sum"] = agg.groupby(cols, dropna=False)["sum"].cumsum() - agg["sum"]
        agg["cum_count"] = agg.groupby(cols, dropna=False)["count"].cumsum() - agg["count"]
        agg[spec["name"]] = (agg["cum_sum"] + prior * alpha) / (agg["cum_count"] + alpha)
        vals = agg[cols + ["season", spec["name"]]]
        out = out.merge(vals, on=cols + ["season"], how="left", suffixes=("", "_new"))
        out[spec["name"]] = out[f"{spec['name']}_new"].fillna(prior).astype(np.float32)
        out = out.drop(columns=[f"{spec['name']}_new"])
        # If a group first appears after the first season, cum_count is zero and the formula is prior.
        if not seasons:
            out[spec["name"]] = np.float32(prior)
    return out


def build_final_te_maps(df, specs, prior):
    maps = []
    for spec in specs:
        maps.append(
            {
                "name": spec["name"],
                "cols": spec["cols"],
                "alpha": spec["alpha"],
                "default": float(prior),
                "mapping": fit_te_map(df, spec["cols"], spec["alpha"], prior),
            }
        )
    return maps


def add_final_te(df, te_maps):
    out = df.copy()
    for spec in te_maps:
        out[spec["name"]] = apply_te_map(out, spec["cols"], spec["mapping"], spec["default"])
    return out


def fit_low_card_categories(df):
    categories = {}
    for col in LOW_CARD_COLS:
        vals = df[col].fillna("__MISSING__").astype(str).unique().tolist()
        categories[col] = sorted(vals)
    return categories


def build_matrix(df, artifact, fit_scaler=False):
    numeric_cols = artifact["numeric_cols"]
    categories = artifact["categories"]
    fill_values = artifact["fill_values"]

    parts = []
    numeric = df[numeric_cols].copy()
    for col in numeric_cols:
        numeric[col] = numeric[col].fillna(fill_values[col])
    x_num = numeric.to_numpy(dtype=np.float32)
    parts.append(x_num)

    for col in LOW_CARD_COLS:
        values = df[col].fillna("__MISSING__").astype(str).to_numpy()
        cats = categories[col]
        block = np.zeros((len(df), len(cats)), dtype=np.float32)
        pos = {cat: i for i, cat in enumerate(cats)}
        for i, value in enumerate(values):
            j = pos.get(value)
            if j is not None:
                block[i, j] = 1.0
        parts.append(block)

    x = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if fit_scaler:
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = x.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        artifact["scaler_mean"] = mean
        artifact["scaler_std"] = std

    x = (x - artifact["scaler_mean"]) / artifact["scaler_std"]
    return x.astype(np.float32, copy=False)


def prepare_train_features(df, prior=None):
    if prior is None:
        prior = float(df[TARGET_COL].mean())
    base = add_derived(df)
    rate_priors = build_rate_priors(base)
    base = add_smoothed_rates(base, rate_priors)
    base = add_temporal_oof_te(base, TE_SPECS, prior)
    te_maps = build_final_te_maps(base, TE_SPECS, prior)
    categories = fit_low_card_categories(base)
    numeric_cols = feature_numeric_cols()
    fill_values = {col: float(base[col].mean(skipna=True)) for col in numeric_cols}
    for col, value in list(fill_values.items()):
        if not np.isfinite(value):
            fill_values[col] = float(prior)

    artifact = {
        "version": 1,
        "global_prior": float(prior),
        "rate_priors": rate_priors,
        "te_maps": te_maps,
        "categories": categories,
        "numeric_cols": numeric_cols,
        "fill_values": fill_values,
        "scaler_mean": None,
        "scaler_std": None,
        "weights": None,
        "bias": None,
        "prob_calibrator": {"type": "identity", "slope": 1.0, "intercept": 0.0},
        "rf_prob_calibrator": {
            "type": "platt",
            "slope": 1.0832335020976482,
            "intercept": -0.0611618416230724,
        },
        "ensemble_weight": 1.0,
        "clip_min": 0.02,
        "clip_max": 0.98,
    }
    x = build_matrix(base, artifact, fit_scaler=True)
    y = base[TARGET_COL].to_numpy(dtype=np.float32)
    return x, y, artifact


def prepare_infer_features(df, artifact):
    base = add_derived(df)
    base = add_smoothed_rates(base, artifact["rate_priors"])
    base = add_final_te(base, artifact["te_maps"])
    return build_matrix(base, artifact, fit_scaler=False)


def fit_future_shift(df):
    rates = df.groupby("season")[TARGET_COL].mean().sort_index()
    anchor_season = int(rates.index.max())
    anchor_rate = float(rates.loc[anchor_season])
    recent = rates.tail(min(3, len(rates)))
    if len(recent) >= 2:
        coef = np.polyfit(recent.index.to_numpy(dtype=float), recent.to_numpy(dtype=float), 1)
        next_rate = float(np.polyval(coef, anchor_season + 1))
    else:
        next_rate = anchor_rate
    next_rate = float(np.clip(next_rate, anchor_rate - 0.035, anchor_rate + 0.025))
    next_rate = float(np.clip(next_rate, 0.35, 0.65))
    per_year_logit_shift = float(logit(next_rate) - logit(anchor_rate))
    return {
        "anchor_season": anchor_season,
        "anchor_rate": anchor_rate,
        "next_rate_forecast": next_rate,
        "per_year_logit_shift": per_year_logit_shift,
    }


def fit_logistic_adam(
    x,
    y,
    epochs=8,
    batch_size=65536,
    lr=0.015,
    l2=1e-4,
    seed=42,
    loss_name="bce",
    verbose=True,
):
    if loss_name not in {"bce", "brier"}:
        raise ValueError(f"Unsupported loss_name={loss_name}")
    rng = np.random.default_rng(seed)
    n, d = x.shape
    w = np.zeros(d, dtype=np.float32)
    b = float(logit(y.mean()))
    mw = np.zeros(d, dtype=np.float32)
    vw = np.zeros(d, dtype=np.float32)
    mb = 0.0
    vb = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = 0
    indices = np.arange(n)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        rng.shuffle(indices)
        total_loss = 0.0
        total_seen = 0
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = x[batch_idx]
            yb = y[batch_idx]
            z = xb @ w + b
            p = sigmoid(z).astype(np.float32)
            err = p - yb
            if loss_name == "brier":
                z_grad = 2.0 * err * p * (1.0 - p)
                grad_w = (xb.T @ z_grad) / len(batch_idx) + l2 * w
                grad_b = float(z_grad.mean())
            else:
                grad_w = (xb.T @ err) / len(batch_idx) + l2 * w
                grad_b = float(err.mean())

            step += 1
            mw = beta1 * mw + (1.0 - beta1) * grad_w
            vw = beta2 * vw + (1.0 - beta2) * (grad_w * grad_w)
            mb = beta1 * mb + (1.0 - beta1) * grad_b
            vb = beta2 * vb + (1.0 - beta2) * (grad_b * grad_b)
            mw_hat = mw / (1.0 - beta1**step)
            vw_hat = vw / (1.0 - beta2**step)
            mb_hat = mb / (1.0 - beta1**step)
            vb_hat = vb / (1.0 - beta2**step)

            w -= lr * mw_hat / (np.sqrt(vw_hat) + eps)
            b -= lr * mb_hat / (math.sqrt(vb_hat) + eps)

            p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
            if loss_name == "brier":
                loss = np.mean((p - yb) ** 2)
            else:
                loss = -np.mean(yb * np.log(p_clip) + (1.0 - yb) * np.log(1.0 - p_clip))
            total_loss += float(loss) * len(batch_idx)
            total_seen += len(batch_idx)
        if verbose:
            print(f"epoch={epoch} {loss_name}_loss={total_loss / total_seen:.6f} time={time.time() - t0:.1f}s")
    return w.astype(np.float32), float(b)


def predict_raw(x, artifact):
    return sigmoid(x @ artifact["weights"] + artifact["bias"]).astype(np.float32)


def apply_future_calibration(p, seasons, artifact):
    shift = artifact.get("future_shift", {})
    anchor = shift.get("anchor_season")
    per_year = shift.get("per_year_logit_shift", 0.0)
    if anchor is None or abs(per_year) < 1e-12:
        z = logit(p)
    else:
        year_delta = np.maximum(0.0, seasons.astype(float) - float(anchor))
        z = logit(p) + per_year * year_delta
    out = sigmoid(z)
    return np.clip(out, artifact["clip_min"], artifact["clip_max"])


def apply_probability_calibration(p, artifact):
    calibrator = artifact.get("prob_calibrator") or {"slope": 1.0, "intercept": 0.0}
    z = calibrator.get("slope", 1.0) * logit(p) + calibrator.get("intercept", 0.0)
    out = sigmoid(z)
    return np.clip(out, artifact["clip_min"], artifact["clip_max"])


def predict_calibrated(x, seasons, artifact):
    p = predict_raw(x, artifact)
    p = apply_probability_calibration(p, artifact)
    p = apply_future_calibration(p, seasons, artifact)
    return p


def brier_score(y, p):
    return float(np.mean((p - y) ** 2))


def log_loss_score(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def auc_score(y, p):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
    pos_rank_sum = ranks[y == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def classification_metrics(y, p, threshold=0.5):
    y = np.asarray(y, dtype=np.int8)
    pred = (np.asarray(p) >= threshold).astype(np.int8)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(1, len(y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return acc, precision, recall, float(pred.mean())


def calibration_ece(y, p, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    ece = 0.0
    rows = []
    for i, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        pred_mean = float(p[idx].mean())
        true_mean = float(y[idx].mean())
        weight = len(idx) / len(y)
        gap = abs(pred_mean - true_mean)
        ece += weight * gap
        rows.append((i, len(idx), pred_mean, true_mean, gap))
    return float(ece), rows


def report_metrics(name, y, p):
    p = np.asarray(p, dtype=float)
    brier = brier_score(y, p)
    r = float(np.mean(y))
    denom = r * (1.0 - r)
    bss = 1.0 - brier / denom if denom > 0 else float("nan")
    logloss = log_loss_score(y, p)
    auc = auc_score(y, p)
    acc, precision, recall, pred_pos_rate = classification_metrics(y, p)
    ece, _ = calibration_ece(y, p)
    print(
        f"{name}: brier={brier:.6f} bss={bss:.6f} logloss={logloss:.6f} "
        f"auc={auc:.6f} acc@0.5={acc:.6f} precision@0.5={precision:.6f} "
        f"recall@0.5={recall:.6f} pred_pos@0.5={pred_pos_rate:.6f} "
        f"ece10={ece:.6f} mean_y={r:.6f} mean_p={p.mean():.6f}"
    )


def fit_probability_calibrator(p, y, epochs=1500, lr=0.03, l2=1e-3):
    x = logit(np.asarray(p, dtype=float))
    y = np.asarray(y, dtype=float)
    slope = 1.0
    intercept = 0.0
    m_s = 0.0
    v_s = 0.0
    m_b = 0.0
    v_b = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for step in range(1, epochs + 1):
        pred = sigmoid(slope * x + intercept)
        err = pred - y
        grad_s = float(np.mean(err * x) + l2 * (slope - 1.0))
        grad_b = float(np.mean(err))
        m_s = beta1 * m_s + (1.0 - beta1) * grad_s
        v_s = beta2 * v_s + (1.0 - beta2) * grad_s * grad_s
        m_b = beta1 * m_b + (1.0 - beta1) * grad_b
        v_b = beta2 * v_b + (1.0 - beta2) * grad_b * grad_b
        slope -= lr * (m_s / (1.0 - beta1**step)) / (math.sqrt(v_s / (1.0 - beta2**step)) + eps)
        intercept -= lr * (m_b / (1.0 - beta1**step)) / (math.sqrt(v_b / (1.0 - beta2**step)) + eps)
    slope = float(np.clip(slope, 0.25, 3.0))
    intercept = float(np.clip(intercept, -2.0, 2.0))
    return {"type": "platt", "slope": slope, "intercept": intercept}


def run_validation(df, args):
    cal_season = args.calibration_season
    if cal_season is None:
        cal_season = args.valid_season - 1
    base_df = df[df["season"] < cal_season].copy()
    cal_df = df[df["season"] == cal_season].copy()
    eval_df = df[df["season"] == args.valid_season].copy()
    if len(base_df) and len(cal_df) and len(eval_df):
        print(
            "Temporal calibration validation "
            f"base={base_df.shape} cal{cal_season}={cal_df.shape} eval{args.valid_season}={eval_df.shape}"
        )
        x_base, y_base, base_artifact = prepare_train_features(base_df)
        w, b = fit_logistic_adam(
            x_base,
            y_base,
            epochs=args.valid_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            l2=args.l2,
            seed=args.seed,
            loss_name=args.loss,
        )
        base_artifact["weights"] = w
        base_artifact["bias"] = b

        x_cal = prepare_infer_features(cal_df, base_artifact)
        y_cal = cal_df[TARGET_COL].to_numpy(dtype=np.float32)
        p_cal_raw = predict_raw(x_cal, base_artifact)
        calibrator = fit_probability_calibrator(p_cal_raw, y_cal)
        print("fit prob_calibrator on calibration season:", calibrator)

        x_eval = prepare_infer_features(eval_df, base_artifact)
        y_eval = eval_df[TARGET_COL].to_numpy(dtype=np.float32)
        eval_seasons = eval_df["season"].to_numpy()
        p_eval_raw = predict_raw(x_eval, base_artifact)
        report_metrics("eval_raw", y_eval, p_eval_raw)
        base_artifact["prob_calibrator"] = calibrator
        p_eval_platt = apply_probability_calibration(p_eval_raw, base_artifact)
        report_metrics("eval_platt", y_eval, p_eval_platt)
        base_artifact["future_shift"] = fit_future_shift(df[df["season"] <= cal_season].copy())
        p_eval_platt_future = apply_future_calibration(p_eval_platt, eval_seasons, base_artifact)
        report_metrics("eval_platt_future", y_eval, p_eval_platt_future)
        print("temporal validation future_shift:", base_artifact["future_shift"])
    else:
        print("Temporal calibration validation skipped: missing base/cal/eval seasons.")

    train_df = df[df["season"] <= args.valid_train_max_season].copy()
    valid_df = df[df["season"] == args.valid_season].copy()
    if len(valid_df) == 0:
        print("Validation skipped: no rows for requested valid season.")
        return
    print(f"Validation train={train_df.shape} valid={valid_df.shape}")
    x_train, y_train, artifact = prepare_train_features(train_df)
    artifact["future_shift"] = fit_future_shift(train_df)
    w, b = fit_logistic_adam(
        x_train,
        y_train,
        epochs=args.valid_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        loss_name=args.loss,
    )
    artifact["weights"] = w
    artifact["bias"] = b
    x_valid = prepare_infer_features(valid_df, artifact)
    y_valid = valid_df[TARGET_COL].to_numpy(dtype=np.float32)
    p_raw = predict_raw(x_valid, artifact)
    artifact["prob_calibrator"] = fit_probability_calibrator(p_raw, y_valid)
    p_platt = apply_probability_calibration(p_raw, artifact)
    p_cal = apply_future_calibration(p_platt, valid_df["season"].to_numpy(), artifact)
    report_metrics("valid_raw", y_valid, p_raw)
    report_metrics("valid_platt_fit_on_same_year", y_valid, p_platt)
    report_metrics("valid_platt_future", y_valid, p_cal)
    print("validation future_shift:", artifact["future_shift"])


def fit_final_probability_calibrator(df, args):
    cal_season = args.final_calibration_season
    if cal_season is None:
        cal_season = int(df["season"].max())
    base_df = df[df["season"] < cal_season].copy()
    cal_df = df[df["season"] == cal_season].copy()
    if len(base_df) == 0 or len(cal_df) == 0:
        return {"type": "identity", "slope": 1.0, "intercept": 0.0}
    print(f"Fitting final probability calibrator with base<{cal_season} and cal={cal_season}")
    x_base, y_base, cal_artifact = prepare_train_features(base_df)
    w, b = fit_logistic_adam(
        x_base,
        y_base,
        epochs=args.calibrator_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        loss_name=args.loss,
        verbose=False,
    )
    cal_artifact["weights"] = w
    cal_artifact["bias"] = b
    x_cal = prepare_infer_features(cal_df, cal_artifact)
    y_cal = cal_df[TARGET_COL].to_numpy(dtype=np.float32)
    p_cal_raw = predict_raw(x_cal, cal_artifact)
    report_metrics("final_calibrator_raw_on_cal_season", y_cal, p_cal_raw)
    calibrator = fit_probability_calibrator(p_cal_raw, y_cal)
    cal_artifact["prob_calibrator"] = calibrator
    report_metrics("final_calibrator_platt_on_cal_season", y_cal, apply_probability_calibration(p_cal_raw, cal_artifact))
    print("final prob_calibrator:", calibrator)
    return calibrator


def save_artifact(artifact, out_path):
    artifact = make_pickle_compatible(artifact)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved artifact: {out_path} ({size_mb:.2f} MB)")


def make_pickle_compatible(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {make_pickle_compatible(k): make_pickle_compatible(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(make_pickle_compatible(v) for v in obj)
    if isinstance(obj, list):
        return [make_pickle_compatible(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Users/joyeongsang/Desktop/lg aimers/open/data")
    parser.add_argument("--out", default="submission/model/artifact.pkl")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--valid-train-max-season", type=int, default=2023)
    parser.add_argument("--valid-season", type=int, default=2024)
    parser.add_argument("--calibration-season", type=int, default=None)
    parser.add_argument("--final-calibration-season", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--valid-epochs", type=int, default=6)
    parser.add_argument("--calibrator-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["bce", "brier"], default="bce")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = Path(args.data_dir) / "train.csv"
    print(f"Loading {train_path}")
    df = pd.read_csv(train_path, nrows=args.nrows)
    print("train shape:", df.shape)
    print("target mean:", float(df[TARGET_COL].mean()))
    print(df.groupby("season")[TARGET_COL].agg(["size", "mean"]).to_string())

    if not args.skip_validation:
        run_validation(df, args)

    if args.skip_final:
        return

    final_calibrator = fit_final_probability_calibrator(df, args)

    print("Training final model...")
    x, y, artifact = prepare_train_features(df)
    artifact["future_shift"] = fit_future_shift(df)
    artifact["prob_calibrator"] = final_calibrator
    w, b = fit_logistic_adam(
        x,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        loss_name=args.loss,
    )
    artifact["weights"] = w
    artifact["bias"] = b
    print("final future_shift:", artifact["future_shift"])
    print("final prob_calibrator:", artifact["prob_calibrator"])
    p_train = predict_raw(x, artifact)
    p_train_platt = apply_probability_calibration(p_train, artifact)
    p_train_cal = apply_future_calibration(p_train_platt, df["season"].to_numpy(), artifact)
    report_metrics("train_in_sample_raw", y, p_train)
    report_metrics("train_in_sample_platt", y, p_train_platt)
    report_metrics("train_in_sample_cal", y, p_train_cal)
    save_artifact(artifact, args.out)


if __name__ == "__main__":
    main()
