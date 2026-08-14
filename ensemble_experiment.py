import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import train as core


ID_COL = core.ID_COL
TARGET_COL = core.TARGET_COL
RF_CAT_COLS = ["top_bottom", "game_type", "base_state"]


def rf_feature_cols(df):
    return [c for c in df.columns if c not in {ID_COL, TARGET_COL}]


def make_rf_pipeline(feature_cols):
    num_cols = [c for c in feature_cols if c not in RF_CAT_COLS]
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                RF_CAT_COLS,
            ),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=200,
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def apply_stored_calibration(p, calibrator, clip_min=0.02, clip_max=0.98):
    temp_artifact = {
        "prob_calibrator": calibrator,
        "clip_min": clip_min,
        "clip_max": clip_max,
    }
    return core.apply_probability_calibration(p, temp_artifact)


def apply_shift(p, seasons, future_shift, clip_min=0.02, clip_max=0.98):
    temp_artifact = {
        "future_shift": future_shift,
        "clip_min": clip_min,
        "clip_max": clip_max,
    }
    return core.apply_future_calibration(p, seasons, temp_artifact)


def sweep_weights(y, p_rf, p_custom, label):
    rows = []
    best = None
    for w in np.linspace(0.0, 1.0, 21):
        p = w * p_rf + (1.0 - w) * p_custom
        brier = brier_score_loss(y, p)
        row = (float(w), float(brier), float(p.mean()))
        rows.append(row)
        if best is None or brier < best[1]:
            best = row
    print(f"\n{label} weight sweep: w = RF weight")
    for w, brier, mean_p in rows:
        marker = "*" if w == best[0] else " "
        print(f"{marker} w={w:.2f} brier={brier:.6f} mean_p={mean_p:.6f}")
    print(f"{label} best: w={best[0]:.2f} brier={best[1]:.6f} mean_p={best[2]:.6f}")
    return best[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Users/joyeongsang/Desktop/lg aimers/open/data")
    parser.add_argument("--loss", choices=["bce", "brier"], default="brier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(Path(args.data_dir) / "train.csv")
    base_df = df[df["season"] <= 2022].copy()
    cal_df = df[df["season"] == 2023].copy()
    eval_df = df[df["season"] == 2024].copy()
    print("base/cal/eval:", base_df.shape, cal_df.shape, eval_df.shape)
    print(
        "rates:",
        {
            "base": float(base_df[TARGET_COL].mean()),
            "cal": float(cal_df[TARGET_COL].mean()),
            "eval": float(eval_df[TARGET_COL].mean()),
        },
    )

    feature_cols = rf_feature_cols(df)
    rf = make_rf_pipeline(feature_cols)
    print("\nTraining split RF...")
    rf.fit(base_df[feature_cols], base_df[TARGET_COL])
    p_rf_cal_raw = rf.predict_proba(cal_df[feature_cols])[:, 1]
    p_rf_eval_raw = rf.predict_proba(eval_df[feature_cols])[:, 1]
    core.report_metrics("rf_cal_raw", cal_df[TARGET_COL].to_numpy(), p_rf_cal_raw)
    core.report_metrics("rf_eval_raw", eval_df[TARGET_COL].to_numpy(), p_rf_eval_raw)

    rf_calibrator = core.fit_probability_calibrator(p_rf_cal_raw, cal_df[TARGET_COL].to_numpy())
    p_rf_cal_platt = apply_stored_calibration(p_rf_cal_raw, rf_calibrator)
    p_rf_eval_platt = apply_stored_calibration(p_rf_eval_raw, rf_calibrator)
    core.report_metrics("rf_cal_platt", cal_df[TARGET_COL].to_numpy(), p_rf_cal_platt)
    core.report_metrics("rf_eval_platt", eval_df[TARGET_COL].to_numpy(), p_rf_eval_platt)
    print("rf_calibrator:", rf_calibrator)

    print("\nTraining split custom model...")
    x_base, y_base, artifact = core.prepare_train_features(base_df)
    weights, bias = core.fit_logistic_adam(
        x_base,
        y_base,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        loss_name=args.loss,
    )
    artifact["weights"] = weights
    artifact["bias"] = bias
    artifact["prob_calibrator"] = {"type": "identity", "slope": 1.0, "intercept": 0.0}
    artifact["future_shift"] = {"anchor_season": None, "per_year_logit_shift": 0.0}

    x_cal = core.prepare_infer_features(cal_df, artifact)
    x_eval = core.prepare_infer_features(eval_df, artifact)
    p_custom_cal_raw = core.predict_raw(x_cal, artifact)
    p_custom_eval_raw = core.predict_raw(x_eval, artifact)
    custom_calibrator = core.fit_probability_calibrator(
        p_custom_cal_raw, cal_df[TARGET_COL].to_numpy()
    )
    artifact["prob_calibrator"] = custom_calibrator
    p_custom_cal_platt = core.apply_probability_calibration(p_custom_cal_raw, artifact)
    p_custom_eval_platt = core.apply_probability_calibration(p_custom_eval_raw, artifact)
    core.report_metrics("custom_cal_platt", cal_df[TARGET_COL].to_numpy(), p_custom_cal_platt)
    core.report_metrics("custom_eval_platt", eval_df[TARGET_COL].to_numpy(), p_custom_eval_platt)
    print("custom_calibrator:", custom_calibrator)

    future_shift = core.fit_future_shift(pd.concat([base_df, cal_df], axis=0))
    p_rf_eval_final = apply_shift(p_rf_eval_platt, eval_df["season"].to_numpy(), future_shift)
    p_custom_eval_final = apply_shift(
        p_custom_eval_platt, eval_df["season"].to_numpy(), future_shift
    )
    core.report_metrics("rf_eval_platt_future", eval_df[TARGET_COL].to_numpy(), p_rf_eval_final)
    core.report_metrics(
        "custom_eval_platt_future", eval_df[TARGET_COL].to_numpy(), p_custom_eval_final
    )

    y_cal = cal_df[TARGET_COL].to_numpy()
    y_eval = eval_df[TARGET_COL].to_numpy()
    best_w = sweep_weights(y_cal, p_rf_cal_platt, p_custom_cal_platt, "cal2023")
    p_eval_ens_platt = best_w * p_rf_eval_platt + (1.0 - best_w) * p_custom_eval_platt
    p_eval_ens_final = apply_shift(p_eval_ens_platt, eval_df["season"].to_numpy(), future_shift)
    core.report_metrics("ensemble_eval_platt_best_cal_w", y_eval, p_eval_ens_platt)
    core.report_metrics("ensemble_eval_platt_future_best_cal_w", y_eval, p_eval_ens_final)

    sweep_weights(y_eval, p_rf_eval_final, p_custom_eval_final, "eval2024_oracle")
    print("\nRecommended RF weight from calibration season:", best_w)
    print("future_shift:", future_shift)


if __name__ == "__main__":
    main()
