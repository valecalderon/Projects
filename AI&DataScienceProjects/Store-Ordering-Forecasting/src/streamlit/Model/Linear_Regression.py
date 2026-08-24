import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import json
import joblib  # to save model later

# Make project root importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

# Shared preprocessing pipeline
from preprocessing import (
    load_weekly_data,
    preprocess_for_model,
    get_feature_columns
)

# Shared metrics + visualization
from Metrics_Visuals import (
    regression_metrics,
    plot_actual_vs_predicted,
    plot_feature_importance,
    print_weekly_sales,
    plot_weekly_trends,
    plot_residuals,
    save_item_predictions
)

# Output paths
ITEM_PREDICTIONS_CSV = os.path.join(
    os.path.dirname(__file__),
    "linear_regression_item_predictions.csv"
)


# ---------------------------------------------------------
# Train the Linear Regression model
# ---------------------------------------------------------
def train_model(df):
    (
        X_train, X_test,
        y_train, y_test,
        X_train_scaled, X_test_scaled,
        scaler
    ) = preprocess_for_model(df)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test_scaled, y_test, X_test, X_train


# ---------------------------------------------------------
# Run evaluation + all shared visuals
# ---------------------------------------------------------
def run_evaluation(model, X_test_scaled, y_test, X_test, X_train, df):

    #predictions
    y_pred = model.predict(X_test_scaled)

    # print metrics
    regression_metrics(y_test, y_pred, label="Linear Regression")

    #actual vs predictions plot
    plot_actual_vs_predicted(
        y_test, y_pred,
        title="Linear Regression — Actual vs Predicted"
    )
    #Feature importance (LR uses coefficients) 
    feature_names = get_feature_columns(df)
    print("Linear Regression Features: ", feature_names)
    # this is to help with streamlit
    FEATURES = feature_names
    # Save feature list to a JSON file
    with open("linear_features.json", "w") as f:
        json.dump(FEATURES, f)

    plot_feature_importance(model, feature_names)

    # Residuals plot
    plot_residuals(y_test, y_pred)

    # Weekly sales pivot table
    print_weekly_sales(df)

    # Weekly trend plot
    plot_weekly_trends(df)

    # Save item-level predictions 
    results = save_item_predictions(
        X_test,
        y_test,
        y_pred,
        df,
        ITEM_PREDICTIONS_CSV
    )

    return results


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    df = load_weekly_data()
    model, scaler, X_test_scaled, y_test, X_test, X_train = train_model(df)
    joblib.dump(scaler, "linear_scaler.joblib")
    joblib.dump(model, "linear_model.joblib")

    results = run_evaluation(model, X_test_scaled, y_test, X_test, X_train, df)


if __name__ == "__main__":
    main()