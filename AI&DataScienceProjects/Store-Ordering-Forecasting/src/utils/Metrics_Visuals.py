"""
Metrics_Visuals.py
Shared metrics + visualization utilities for all forecasting models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


# =============================================================================
# REGRESSION METRICS
# =============================================================================
def regression_metrics(y_true, y_pred, label="Model"):
    """Return and print RMSE, MAE, R2."""
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)

    y_range = y_true.max() - y_true.min()
    nrmse = rmse / y_range if y_range != 0 else np.nan
    nmae = mae / y_range if y_range != 0 else np.nan

    print(f"\n===== {label} Metrics =====")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")
    print(f"NRMSE: {nrmse:.3f}")
    print(f"NMAE: {nmae:.3f}")
    print("==========================\n")

    return {"rmse": rmse, "mae": mae, "r2": r2, "nrmse": nrmse, "nmae": nmae}


# =============================================================================
# ACTUAL VS PREDICTED PLOT
# =============================================================================
def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted Sales"):
    """Scatter plot with ideal y=x reference line."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             "r--", linewidth=2)

    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# =============================================================================
# WEEKLY SALES TABLE (PIVOT)
# =============================================================================
def print_weekly_sales(df):
    """
    Create a pivot table showing weekly sales per product.
    Uses:
    - product_mdm_id_norm (preferred)
    """
    group_col = None

    if "product_mdm_id_norm" in df.columns:
        group_col = "product_mdm_id_norm"
    elif "product_family" in df.columns:
        group_col = "product_family"
    else:
        print("No suitable grouping column found for weekly sales table.")
        return

    pivot_df = df.pivot_table(
        index=group_col,
        columns="week_of_year",
        values="weekly_sales",
        aggfunc="sum",
        fill_value=0
    )

    pivot_df["Total_Sales"] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values("Total_Sales", ascending=False)

    print("\n===== Weekly Sales Pivot Table =====")
    print(pivot_df.head(10))
    print("====================================\n")

    return pivot_df


def save_item_predictions(X_test, y_test, y_pred, df, out_csv):
    """Save item-level predictions with metadata to CSV."""
    results = X_test.copy()
    results["actual_sales"] = y_test.values
    results["predicted_sales"] = y_pred
    results["product_mdm_id_norm"] = df.loc[y_test.index, "product_mdm_id_norm"].values

    for col in ["store_number", "year", "week_of_year"]:
        if col in df.columns:
            results[col] = df.loc[y_test.index, col].values

    results.to_csv(out_csv, index=False)
    print(f"Saved item-level predictions → {out_csv}")

    return results

# =============================================================================
# FEATURE IMPORTANCE (XGBoost, Tree Models, Linear Models)
# =============================================================================
def plot_feature_importance(model, feature_names, top_n=20):
    """
    Supports:
    - XGBoost / RandomForest / GradientBoosting → model.feature_importances_
    - Linear Regression / Linear SVM → model.coef_
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_)
    else:
        print("This model does not expose feature importance.")
        return

    importance = np.array(importance)
    feature_names = np.array(feature_names)

    sorted_idx = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[sorted_idx], importance[sorted_idx])
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


# =============================================================================
# WEEKLY TREND PLOT
# =============================================================================
def plot_weekly_trends(df, group_col="product_mdm_id_norm"):
    """Plot weekly sales trends for each grouping category."""
    if group_col not in df.columns:
        print(f"Column '{group_col}' not found — skipping trend plot.")
        return

    weekly_df = (
        df.groupby(["week_of_year", group_col])["weekly_sales"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=weekly_df,
        x="week_of_year",
        y="weekly_sales",
        hue=group_col,
        legend=False
    )
    plt.title("Weekly Sales Trends")
    plt.xlabel("Week of Year")
    plt.ylabel("Weekly Sales")
    plt.tight_layout()
    plt.show()


# =============================================================================
# TOP PRODUCTS WEEKLY SALES PLOT
# =============================================================================
def plot_top_products_weekly(df, product_col="Product_name", top_n=50):
    """Plot weekly sales over time for the top-N products."""
    if product_col not in df.columns:
        print(f"Column '{product_col}' not in dataframe — skipping top-product plot.")
        return

    top_products = (
        df.groupby(product_col)["weekly_sales"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    filtered_df = df[df[product_col].isin(top_products)]

    weekly_df = (
        filtered_df.groupby(["week_of_year", product_col])["weekly_sales"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(16, 8))
    sns.lineplot(data=weekly_df, x="week_of_year", y="weekly_sales", hue=product_col)
    plt.xticks(rotation=45)
    plt.title(f"Weekly Sales Over Time — Top {top_n} Products")
    plt.xlabel("Week of Year")
    plt.ylabel("Weekly Sales")
    plt.legend(
        title="Product",
        fontsize=8,
        title_fontsize=10,
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )
    plt.tight_layout()
    plt.show()


# =============================================================================
# GENERIC WEEKLY LINEPLOT FOR ANY COLUMN GROUPING
# =============================================================================
def plot_weekly_grouped(df, group_col):
    """Generic grouped weekly plot."""
    if group_col not in df.columns:
        print(f"Column '{group_col}' not in dataframe — skipping grouped weekly plot.")
        return

    weekly_df = (
        df.groupby(["week_of_year", group_col])["weekly_sales"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=weekly_df, x="week_of_year", y="weekly_sales", hue=group_col)
    plt.title(f"Weekly Sales Grouped by {group_col}")
    plt.xlabel("Week of Year")
    plt.ylabel("Weekly Sales")
    plt.tight_layout()
    plt.show()

# =============================================================================
# RESIDUALS PLOT
# =============================================================================
def plot_residuals(y_true, y_pred, title="Residuals Plot"):
    """Plot residuals (y_true - y_pred) to diagnose model errors."""
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# =============================================================================
# ACTUAL VS PREDICTED PLOT (ZOOMED)
# =============================================================================
def plot_actual_vs_predicted_zoom(y_true, y_pred, title="Actual vs Predicted (Zoomed on Small Values)", max_value=150):
    """Scatter plot with ideal y=x reference line, zoomed to small values."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             "r--", linewidth=2)

    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(alpha=0.2)
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    plt.tight_layout()
    plt.show()