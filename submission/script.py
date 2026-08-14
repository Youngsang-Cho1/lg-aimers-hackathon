import os
import pickle

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


def add_smoothed_rates(df, rate_priors):
    out = df.copy()
    for rate_col, n_col, m in SMOOTH_SPECS:
        prior = rate_priors[rate_col]
        rate = out[rate_col].fillna(prior).astype(float)
        n = out[n_col].fillna(0.0).astype(float).clip(lower=0.0)
        out[f"{rate_col}_smooth"] = (rate * n + prior * m) / (n + m)
    return out


def apply_te_map(df, cols, mapping, default):
    n = len(df)
    if len(cols) == 1:
        return df[cols[0]].map(mapping).fillna(default).to_numpy(dtype=np.float32)
    arrays = [df[c].to_numpy() for c in cols]
    return np.fromiter((mapping.get(tuple(key), default) for key in zip(*arrays)), dtype=np.float32, count=n)


def add_final_te(df, te_maps):
    out = df.copy()
    for spec in te_maps:
        out[spec["name"]] = apply_te_map(out, spec["cols"], spec["mapping"], spec["default"])
    return out


def build_matrix(df, artifact):
    numeric_cols = artifact["numeric_cols"]
    fill_values = artifact["fill_values"]
    parts = []

    numeric = df[numeric_cols].copy()
    for col in numeric_cols:
        numeric[col] = numeric[col].fillna(fill_values[col])
    parts.append(numeric.to_numpy(dtype=np.float32))

    categories = artifact["categories"]
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
    x = (x - artifact["scaler_mean"]) / artifact["scaler_std"]
    return x.astype(np.float32, copy=False)


def prepare_features(df, artifact):
    base = add_derived(df)
    base = add_smoothed_rates(base, artifact["rate_priors"])
    base = add_final_te(base, artifact["te_maps"])
    return build_matrix(base, artifact)


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


def apply_probability_calibration_with(p, calibrator, clip_min, clip_max):
    z = calibrator.get("slope", 1.0) * logit(p) + calibrator.get("intercept", 0.0)
    out = sigmoid(z)
    return np.clip(out, clip_min, clip_max)


def apply_probability_calibration(p, artifact):
    calibrator = artifact.get("prob_calibrator") or {"slope": 1.0, "intercept": 0.0}
    return apply_probability_calibration_with(
        p, calibrator, artifact["clip_min"], artifact["clip_max"]
    )


def predict(df, artifact):
    x = prepare_features(df, artifact)
    p = sigmoid(x @ artifact["weights"] + artifact["bias"]).astype(np.float32)
    p = apply_probability_calibration(p, artifact)
    return apply_future_calibration(p, df["season"].to_numpy(), artifact)


def normalize_artifact(artifact):
    artifact["weights"] = np.asarray(artifact["weights"], dtype=np.float32)
    artifact["scaler_mean"] = np.asarray(artifact["scaler_mean"], dtype=np.float32)
    artifact["scaler_std"] = np.asarray(artifact["scaler_std"], dtype=np.float32)
    artifact.setdefault("prob_calibrator", {"type": "identity", "slope": 1.0, "intercept": 0.0})
    artifact.setdefault("rf_prob_calibrator", {"type": "identity", "slope": 1.0, "intercept": 0.0})
    artifact.setdefault("ensemble_weight", 0.85)
    return artifact


def load_optional_rf(path):
    if not os.path.exists(path):
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception as exc:
        print(f"RF model unavailable, using custom model only: {exc}")
        return None


def predict_optional_rf(df, model):
    if model is None:
        return None
    x = df.drop(columns=[ID_COL])
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1].astype(np.float32)
    return None


def main():
    test_path = "./data/test.csv"
    model_path = "./model/artifact.pkl"
    rf_path = "./model/rf.pkl"
    out_path = "./output/submission.csv"

    test = pd.read_csv(test_path, encoding="utf-8-sig")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    artifact = normalize_artifact(artifact)

    custom_preds = predict(test, artifact)
    rf_model = load_optional_rf(rf_path)
    rf_preds = predict_optional_rf(test, rf_model)
    if rf_preds is None:
        preds = custom_preds
    else:
        rf_weight = float(artifact.get("ensemble_weight", 0.85))
        preds = rf_weight * rf_preds + (1.0 - rf_weight) * custom_preds
        preds = np.clip(preds, 0.0, 1.0)
    sub = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET_COL: preds})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path} rows={len(sub)}")


if __name__ == "__main__":
    main()
