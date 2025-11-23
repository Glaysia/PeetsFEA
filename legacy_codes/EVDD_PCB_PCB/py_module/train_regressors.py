import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def get_model_pipeline():
    """
    Return a fast, solid default regression pipeline.
    Tries HistGradientBoostingRegressor; falls back to GradientBoostingRegressor,
    then RandomForestRegressor if needed.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    model = None
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=None,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
            random_state=42,
        )
    except Exception:
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
        except Exception:
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                n_jobs=-1,
                random_state=42,
            )

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )
    return pipe


INPUT_COLS: List[str] = [
    "Ltx",
    "Lrx1",
    "Lrx2",
    "M1",
    "M2",
    "k1",
    "k2",
    "Lmt",
    "Lmr1",
    "Lmr2",
    "Llt",
    "Llr1",
    "Llr2",
    "Rtx",
    "Rrx1",
    "Rrx2",
    "copperloss_Tx",
    "copperloss_Rx1",
    "copperloss_Rx2",
    "coreloss",
    "B_core",
    "B_left",
    "B_right",
    "B_center",
    "B_top_left",
    "B_bottom_left",
    "B_top_right",
    "B_bottom_right",
    "magnetizing_copperloss_Tx",
    "magnetizing_copperloss_Rx1",
    "magnetizing_copperloss_Rx2",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train one regression model per output column using the given input columns."
        )
    )
    p.add_argument(
        "--csv",
        default="output_data.csv",
        help="Path to the input CSV (default: output_data.csv)",
    )
    p.add_argument(
        "--out",
        default="models",
        help="Directory to save models and metrics (default: models)",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout test size fraction (default: 0.2)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return p.parse_args()


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Replace inf with NaN to avoid model errors
    df = df.replace([np.inf, -np.inf], np.nan)
    # Coerce columns that look numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def split_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    missing = [c for c in INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required input columns in CSV: {missing}.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Define X using the provided input columns
    X = df[INPUT_COLS].copy()

    # Targets are all other columns
    targets = [c for c in df.columns if c not in INPUT_COLS]
    return X, targets


def train_per_target(
    X: pd.DataFrame,
    df: pd.DataFrame,
    targets: List[str],
    out_dir: str,
    test_size: float,
    seed: int,
):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    import joblib

    os.makedirs(out_dir, exist_ok=True)

    rows = []

    for i, target in enumerate(targets, start=1):
        y = pd.to_numeric(df[target], errors="coerce")

        # Drop rows where y is NaN
        mask = ~y.isna()
        Xi = X.loc[mask]
        yi = y.loc[mask]

        if len(yi) < 10:
            print(f"[skip] Target '{target}' has too few non-NaN rows: {len(yi)}")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            Xi, yi, test_size=test_size, random_state=seed
        )

        model = get_model_pipeline()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        # Save model
        model_path = os.path.join(out_dir, f"{target}.joblib")
        joblib.dump(model, model_path)

        rows.append(
            {
                "target": target,
                "n_samples": int(len(yi)),
                "r2": float(r2),
                "mae": float(mae),
                "rmse": float(rmse),
                "model_path": model_path,
            }
        )

        print(
            f"[{i}/{len(targets)}] {target:30s} -> R2={r2:.4f} MAE={mae:.6g} RMSE={rmse:.6g}"
        )

    if rows:
        metrics_df = pd.DataFrame(rows).sort_values(by=["r2"], ascending=False)
        metrics_path = os.path.join(out_dir, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics -> {metrics_path}")
    else:
        print("No models were trained (no valid targets found).")


def main():
    args = parse_args()
    df = load_dataframe(args.csv)
    X, targets = split_features_targets(df)
    print(f"Loaded {len(df)} rows, {X.shape[1]} input features, {len(targets)} targets.")
    train_per_target(
        X=X,
        df=df,
        targets=targets,
        out_dir=args.out,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
