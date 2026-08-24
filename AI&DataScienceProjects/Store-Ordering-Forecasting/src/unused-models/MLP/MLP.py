# MLP_fixed.py
# Train a Multilayer Perceptron on weekly_item_sales.csv to predict weekly_sales.

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

from tensorflow import keras
from tensorflow.keras import layers
import argparse
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------
# Make project root importable so we can use Metrics_Visuals
# ------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from Metrics_Visuals import (
    regression_metrics,
    plot_actual_vs_predicted,
    save_item_predictions,
)

# ----------------------------
# Feature engineering
# ----------------------------
def load_weekly_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize columns: strip and lowercase
    df.columns = df.columns.str.strip().str.lower()
    return df


def build_features(df: pd.DataFrame, expand_full_weeks: bool = True) -> pd.DataFrame:
    """
    Build features:
      - expects at minimum columns: store_number, product_mdm_id_norm, year, week_of_year, weekly_sales
      - optional: product_family, product_class, pizza_size_inches, weekend_ratio, avg_month
    Creates lag_1, lag_2, rolling_mean_4 per (store_number, product_mdm_id_norm)
    Optionally expands to a full 52-week grid per (store, item, year) filling missing weekly_sales with 0.
    Returns sorted dataframe.
    """
    df = df.copy()

    required = {"store_number", "product_mdm_id_norm", "year", "week_of_year", "weekly_sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure numeric columns
    for col in ["year", "week_of_year", "weekly_sales"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out = df.copy()
    keys = ["store_number", "product_mdm_id_norm", "year"]

    # Optional: expand to 52-week calendar per (store,item,year)
    if expand_full_weeks:
        base = out[keys].drop_duplicates().assign(_k=1)
        weeks = pd.DataFrame({"week_of_year": list(range(1, 53)), "_k": 1})
        full = base.merge(weeks, on="_k").drop(columns="_k")

        keep_cols = [c for c in out.columns if c not in keys + ["week_of_year"]]
        out = full.merge(
            out[keys + ["week_of_year"] + keep_cols],
            on=keys + ["week_of_year"],
            how="left",
            suffixes=("", "_orig"),
        )
        # Missing sales become 0 when we create explicit calendar
        out["weekly_sales"] = out["weekly_sales"].fillna(0)

    # Sort rows for reproducible lagging
    out = out.sort_values(keys + ["week_of_year"]).reset_index(drop=True)

    # Group and compute lags/rolling mean per (store, item)
    g = out.groupby(["store_number", "product_mdm_id_norm"], group_keys=False)
    out["lag_1"] = g["weekly_sales"].shift(1).fillna(0.0)
    out["lag_2"] = g["weekly_sales"].shift(2).fillna(0.0)
    out["rolling_mean_4"] = (
        g["weekly_sales"]
        .shift(1)
        .rolling(window=4, min_periods=1)
        .mean()
        .fillna(0.0)
    )

    # Optionally fill numeric optional features with zeros
    for opt in ["pizza_size_inches", "weekend_ratio", "avg_month"]:
        if opt in out.columns:
            out[opt] = pd.to_numeric(out[opt], errors="coerce").fillna(0.0)

    return out


# ----------------------------
# Preparing matrices & split
# ----------------------------
def prepare_matrices(df: pd.DataFrame):
    """
    Build X (features) and y (target).
    One-hot encode categorical columns across the whole dataframe to maintain consistent columns.
    Returns X (DataFrame), y (Series), feature_names list.
    """
    cat_cols = [c for c in ["product_family", "product_class"] if c in df.columns]
    num_cols = [
        c
        for c in [
            "week_of_year",
            "pizza_size_inches",
            "weekend_ratio",
            "avg_month",
            "lag_1",
            "lag_2",
            "rolling_mean_4",
        ]
        if c in df.columns
    ]

    if not num_cols and not cat_cols:
        raise ValueError("No usable feature columns found in dataframe.")

    X = df[num_cols + cat_cols].copy()

    # One-hot encode categoricals
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    y = df["weekly_sales"].astype(float).copy()
    feature_names = list(X.columns)
    return X, y, feature_names


def time_based_split(df: pd.DataFrame, train_frac: float = 0.8):
    """
    Create boolean masks for train/test based on unique (year, week_of_year) ordering.
    Returns (train_mask, test_mask) aligned with df rows (both are boolean arrays).
    """
    time_keys = df[["year", "week_of_year"]].drop_duplicates().sort_values(["year", "week_of_year"])
    cut_idx = int(np.floor(train_frac * len(time_keys)))
    train_keys = time_keys.iloc[:cut_idx]

    idx = df.set_index(["year", "week_of_year"]).index
    train_mask = idx.isin(train_keys.set_index(["year", "week_of_year"]).index)
    test_mask = ~train_mask
    return np.array(train_mask), np.array(test_mask)


# ----------------------------
# Model building
# ----------------------------
def build_mlp(input_dim, hidden=(64, 32), dropout=0.2, l2_reg=1e-4, learning_rate=1e-3):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden:
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2_reg),
            )
        )
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


# ----------------------------
# sklearn-compatible wrapper for permutation importance
# ----------------------------
def get_sklearn_wrapper(build_fn, fit_kwargs=None):
    """
    Returns an sklearn-like estimator implementing fit & predict.
    If scikeras is available, use KerasRegressor from scikeras.
    Otherwise use a simple custom wrapper that trains inside fit.
    """
    fit_kwargs = fit_kwargs or {}

    try:
        from scikeras.wrappers import KerasRegressor

        def sk_estimator_constructor():
            return KerasRegressor(
                model=build_fn,
                **fit_kwargs,
                verbose=0,
            )

        return sk_estimator_constructor(), "scikeras"
    except Exception:
        class SimpleKerasWrapper(BaseEstimator, RegressorMixin):
            def __init__(self, build_fn, epochs=10, batch_size=32, verbose=0):
                self.build_fn = build_fn
                self.epochs = epochs
                self.batch_size = batch_size
                self.verbose = verbose
                self.model_ = None

            def fit(self, X, y, **kwargs):
                self.model_ = self.build_fn()
                self.model_.fit(
                    np.asarray(X),
                    np.asarray(y),
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    verbose=self.verbose,
                    **kwargs,
                )
                return self

            def predict(self, X):
                if self.model_ is None:
                    raise ValueError("Model not fitted. Call fit() before predict().")
                return self.model_.predict(np.asarray(X)).ravel()

        params = {
            "epochs": fit_kwargs.get("epochs", 10),
            "batch_size": fit_kwargs.get("batch_size", 32),
            "verbose": 0,
        }
        return SimpleKerasWrapper(build_fn, **params), "custom"


# ----------------------------
# Plot helpers (local to MLP)
# ----------------------------
def save_plot_dir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def plot_training(history, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Training vs Validation Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions(y_true, y_pred, weeks, weeks_test, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Actual weekly sales scatter (train)
    plt.figure(figsize=(6, 6))
    plt.scatter(weeks, y_true, alpha=0.6, color="orange", label="Actual Weekly Sales")
    plt.xlabel("Week of the year")
    plt.ylabel("Weekly Sales")
    plt.title("Scatter: Actual Weekly Sales")
    plt.tight_layout()
    plt.savefig(outdir / "true_scatter.png", dpi=150)
    plt.close()

    # Predicted weekly sales scatter (test)
    plt.figure(figsize=(6, 6))
    plt.scatter(weeks_test, y_pred, alpha=0.6, color="orange", label="Prediction")
    plt.xlabel("Week of the year")
    plt.ylabel("Weekly Sales")
    plt.title("Scatter: Predicted Weekly Sales")
    plt.tight_layout()
    plt.savefig(outdir / "pred_scatter.png", dpi=150)
    plt.close()


def plot_residuals_time(y_true, y_pred, df_test, outdir: Path):
    save_plot_dir(outdir)
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure(figsize=(14, 4))
    plt.plot(residuals, lw=0.8)
    plt.axhline(0, color="red", linestyle="--", linewidth=0.8)
    plt.xlabel("Test sample index (time-ordered)")
    plt.ylabel("Residual (true - pred)")
    plt.title("Residuals over Test Time")
    plt.tight_layout()
    plt.savefig(outdir / "residuals_time.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_distribution(y_true, y_pred, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    errors = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=30, alpha=0.7, color="skyblue")
    plt.title("Error Distribution (True - Predicted)")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outdir / "error_distribution.png")
    plt.close()


def plot_time_series(df, y_true, y_pred, test_mask, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    df_test = df[test_mask].copy().reset_index(drop=True)
    df_test["true"] = y_true.reset_index(drop=True)
    df_test["pred"] = y_pred

    if "year" in df_test.columns and "week_of_year" in df_test.columns:
        df_test = df_test.sort_values(["year", "week_of_year"]).reset_index(drop=True)

    plt.figure(figsize=(16, 5))
    plt.plot(df_test["true"], label="Actual Weekly Sales", color="blue")
    plt.plot(df_test["pred"], label="Predicted Weekly Sales", color="orange")
    plt.title("Time Series: True vs Predicted Sales (Test Set)")
    plt.xlabel("Test Row Index (chronological order)")
    plt.ylabel("Weekly Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "timeseries_plot.png", dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------
# Main training/eval flow
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train MLP and produce diagnostic plots")
    parser.add_argument(
        "--data",
        type=str,
        default="weekly_item_sales.csv",
        help="Path to CSV file",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="mlp_artifacts",
        help="Output directory for models/plots",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--hidden",
        type=str,
        default="64,32",
        help="Comma-separated hidden layer sizes",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}. Provide correct --data path.")

    print("Loading data:", csv_path)
    df = load_weekly_table(csv_path)

    print("Building features...")
    df_feat = build_features(df, expand_full_weeks=True)

    df_feat = df_feat.sort_values(
        ["store_number", "product_mdm_id_norm", "year", "week_of_year"]
    ).reset_index(drop=True)

    # Prepare X, y
    X, y, feature_names = prepare_matrices(df_feat)

    # Time-based split
    train_mask, test_mask = time_based_split(df_feat, train_frac=args.train_frac)

    if not (np.any(train_mask) and np.any(test_mask)):
        raise ValueError("Train or test mask is empty. Check train_frac or your data time span.")

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    print(f"Samples → train: {len(X_train)} | test: {len(X_test)} | features: {X_train.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Parse hidden sizes
    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    def build_fn():
        return build_mlp(
            input_dim=X_train_s.shape[1],
            hidden=hidden,
            dropout=0.2,
            l2_reg=1e-4,
            learning_rate=args.lr,
        )

    model = build_fn()
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1
    )

    print("Training model...")
    history = model.fit(
        X_train_s,
        y_train,
        validation_data=(X_test_s, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    print("Predicting on test set...")
    y_pred = model.predict(X_test_s).ravel()
    weeks = X_train["week_of_year"].values
    weeks_test = X_test["week_of_year"].values

    # ---- Shared metrics + plots from Metrics_Visuals ----
    metrics = regression_metrics(y_test, y_pred, label="MLP")
    rmse = float(metrics["rmse"])
    mae = float(metrics["mae"])

    # Shared scatter plot (consistent look with other models)
    plot_actual_vs_predicted(y_test, y_pred, title="MLP — Actual vs Predicted")

    # Save item-level predictions in same format as other models
    mlp_pred_csv = outdir / "mlp_item_predictions.csv"
    save_item_predictions(X_test, y_test, y_pred, df_feat, mlp_pred_csv)

    # ---- Local plots ----
    print("Creating additional MLP-specific plots...")
    plot_training(history, outdir)
    plot_predictions(y_train, y_pred, weeks, weeks_test, outdir)
    plot_error_distribution(y_test, y_pred, outdir)
    plot_time_series(df_feat, y_test, y_pred, test_mask, outdir)
    plot_residuals_time(y_test, y_pred, df_feat[test_mask], outdir)

    # Save artifacts
    model_path = outdir / "mlp_model.h5"
    scaler_path = outdir / "scaler.pkl"
    feature_path = outdir / "feature_names.json"
    metrics_path = outdir / "metrics.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    feature_path.write_text(json.dumps(feature_names, indent=2))
    metrics_path.write_text(json.dumps({"rmse": rmse, "mae": mae}, indent=2))

    print("Saved model and artifacts to", outdir)

    # Optional: permutation importance (still using wrapper)
    print("Computing permutation importance...")
    sk_wrapper, wrapper_type = get_sklearn_wrapper(
        build_fn,
        fit_kwargs={"epochs": args.epochs, "batch_size": args.batch_size},
    )
    print(f"Fitting sklearn-style wrapper for permutation importance (wrapper type: {wrapper_type})")
    sk_wrapper.fit(X_train_s, y_train)

    print("All done. Plots and artifacts are in:", outdir)
    print(f"Final metrics — RMSE: {rmse:.4f}, MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
