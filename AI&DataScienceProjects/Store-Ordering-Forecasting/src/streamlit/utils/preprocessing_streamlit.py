import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .main import WEEKLY_ITEM_SALES_CSV


# =====================================================================
# Load + Clean + Feature Engineering  (Single Shared Pipeline)
# =====================================================================
def load_weekly_data():
    # """Load weekly_item_sales.csv and perform all preprocessing consistently."""
    
    # print("Loading weekly item sales data...")
    df = pd.read_csv(WEEKLY_ITEM_SALES_CSV)

    # ------------------------------------------------------------
    # Date parsing + seasonal features
    # ------------------------------------------------------------
    if "business_date" in df.columns:
        df["business_date"] = pd.to_datetime(df["business_date"])
        df["avg_month"] = df["business_date"].dt.avg_month

    if "is_weekend" in df.columns:
        df["weekend_ratio"] = df.groupby(
            ["product_mdm_id_norm", "week_of_year"]
        )["is_weekend"].transform("mean")

    # ------------------------------------------------------------
    # Drop bad rows
    # ------------------------------------------------------------
    df = df.dropna(subset=["weekly_sales", "pizza_size_inches"])

    # ------------------------------------------------------------
    # One-hot encoding
    # ------------------------------------------------------------
    df = pd.get_dummies(df, columns=["product_family", "product_class"], drop_first=True)

    # ------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------
    df = df.sort_values(by=["product_mdm_id_norm", "week_of_year"])
    df["lag_1"] = df.groupby("product_mdm_id_norm")["weekly_sales"].shift(1)
    df["lag_2"] = df.groupby("product_mdm_id_norm")["weekly_sales"].shift(2)
    df["rolling_mean_4"] = df.groupby("product_mdm_id_norm")["weekly_sales"].transform(
        lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
    )

    df = df.dropna(subset=["lag_1", "lag_2", "rolling_mean_4"]).reset_index(drop=True)

    return df



# =====================================================================
# Explicit Feature Whitelist
# =====================================================================

BASE_FEATURES = [
    "week_of_year",
    "year",
    "pizza_size_inches",
    "avg_month",
    "weekend_ratio",
    "lag_1",
    "lag_2",
    "rolling_mean_4",
]

def get_feature_columns(df):
    """Return the explicitly allowed feature columns."""
    one_hot_cols = [
        c for c in df.columns
        if c.startswith("product_family_") or c.startswith("product_class_")
    ]
    return BASE_FEATURES + one_hot_cols


# =====================================================================
# Shared Train/Test Split
# =====================================================================
def split_train_test(df):
    """Perform a consistent train/test split for all models."""
    X = df[get_feature_columns(df)]
    y = df["weekly_sales"].astype(float)
    
    return train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )


# =====================================================================
# Shared Scaler (for SVM / Linear Regression)
# =====================================================================
def scale_features(X_train, X_test):
    """Scale features using one StandardScaler for LR and SVM."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler

# =====================================================================
# Combined preprocessing helper for models that require scaling
# =====================================================================
def preprocess_for_model(df):
    """
    Runs train/test split and scaling for models like SVM or Linear Regression.
    Returns:
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scaler
    """
    # Split
    X_train, X_test, y_train, y_test = split_train_test(df)

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
