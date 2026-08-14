import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import train as core


TARGET_COL = core.TARGET_COL


def split_random(df, seed=42, eval_frac=0.2, cal_frac=0.1):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_eval = int(len(df) * eval_frac)
    n_cal = int(len(df) * cal_frac)
    eval_idx = order[:n_eval]
    cal_idx = order[n_eval : n_eval + n_cal]
    base_idx = order[n_eval + n_cal :]
    return df.iloc[base_idx].copy(), df.iloc[cal_idx].copy(), df.iloc[eval_idx].copy()


def split_temporal(df, cal_season=2023, eval_season=2024):
    base = df[df["season"] < cal_season].copy()
    cal = df[df["season"] == cal_season].copy()
    eval_df = df[df["season"] == eval_season].copy()
    return base, cal, eval_df


def run_split(name, base_df, cal_df, eval_df, loss_name, args):
    print("\n" + "=" * 90)
    print(f"split={name} loss={loss_name}")
    print(f"base={base_df.shape} cal={cal_df.shape} eval={eval_df.shape}")
    print(
        "target rates:",
        {
            "base": float(base_df[TARGET_COL].mean()),
            "cal": float(cal_df[TARGET_COL].mean()),
            "eval": float(eval_df[TARGET_COL].mean()),
        },
    )

    x_base, y_base, artifact = core.prepare_train_features(base_df)
    weights, bias = core.fit_logistic_adam(
        x_base,
        y_base,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        loss_name=loss_name,
        verbose=True,
    )
    artifact["weights"] = weights
    artifact["bias"] = bias
    artifact["prob_calibrator"] = {"type": "identity", "slope": 1.0, "intercept": 0.0}
    artifact["future_shift"] = {"anchor_season": None, "per_year_logit_shift": 0.0}

    x_cal = core.prepare_infer_features(cal_df, artifact)
    y_cal = cal_df[TARGET_COL].to_numpy(dtype=np.float32)
    p_cal_raw = core.predict_raw(x_cal, artifact)
    core.report_metrics(f"{name}_{loss_name}_cal_raw", y_cal, p_cal_raw)
    calibrator = core.fit_probability_calibrator(p_cal_raw, y_cal)
    artifact["prob_calibrator"] = calibrator
    p_cal_platt = core.apply_probability_calibration(p_cal_raw, artifact)
    core.report_metrics(f"{name}_{loss_name}_cal_platt", y_cal, p_cal_platt)
    print("prob_calibrator:", calibrator)

    x_eval = core.prepare_infer_features(eval_df, artifact)
    y_eval = eval_df[TARGET_COL].to_numpy(dtype=np.float32)
    p_eval_raw = core.predict_raw(x_eval, artifact)
    core.report_metrics(f"{name}_{loss_name}_eval_raw", y_eval, p_eval_raw)
    p_eval_platt = core.apply_probability_calibration(p_eval_raw, artifact)
    core.report_metrics(f"{name}_{loss_name}_eval_platt", y_eval, p_eval_platt)

    if name == "temporal":
        artifact["future_shift"] = core.fit_future_shift(pd.concat([base_df, cal_df], axis=0))
        p_eval_future = core.apply_future_calibration(p_eval_platt, eval_df["season"].to_numpy(), artifact)
        core.report_metrics(f"{name}_{loss_name}_eval_platt_future", y_eval, p_eval_future)
        print("future_shift:", artifact["future_shift"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Users/joyeongsang/Desktop/lg aimers/open/data")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--splits", default="temporal,random")
    parser.add_argument("--losses", default="bce,brier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = Path(args.data_dir) / "train.csv"
    df = pd.read_csv(path, nrows=args.nrows)
    print("loaded", path, df.shape)
    print(df.groupby("season")[TARGET_COL].agg(["size", "mean"]).to_string())

    requested_splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    losses = [x.strip() for x in args.losses.split(",") if x.strip()]

    for split_name in requested_splits:
        if split_name == "temporal":
            base_df, cal_df, eval_df = split_temporal(df)
        elif split_name == "random":
            base_df, cal_df, eval_df = split_random(df, seed=args.seed)
        else:
            raise ValueError(f"Unknown split={split_name}")

        for loss_name in losses:
            run_split(split_name, base_df, cal_df, eval_df, loss_name, args)


if __name__ == "__main__":
    main()
