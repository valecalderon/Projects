import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Make project root importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT_DIR)

# Shared preprocessing pipeline
from src.utils.preprocessing import (
    load_weekly_data,
    preprocess_for_model,
    get_feature_columns
)

# Shared metrics + visualization
from src.utils.Metrics_Visuals import (
    regression_metrics,
    plot_actual_vs_predicted,
    plot_actual_vs_predicted_zoom,
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
    """Trains a Linear Regression model on the provided dataframe.
    Steps:
     1. splits data
     2. scales features
     3. fits Linear Regression model
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe with features and target.
    
    Returns:
        tuple: Trained Linear Regression model, scaler, x_test_scaled, y_test, X_test, X_train)"""

    # Preprocess data for model
    (
        X_train, X_test,
        y_train, y_test,
        X_train_scaled, X_test_scaled,
        scaler
    ) = preprocess_for_model(df)

    # Initialize and train Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test_scaled, y_test, X_test, X_train


# ---------------------------------------------------------
# Run evaluation + all shared visuals
# ---------------------------------------------------------
def run_evaluation(model, X_test_scaled, y_test, X_test, X_train, df):
    """
    Evaluates the trained Linear Regression model and generates visualizations.
    Steps:
        1. Makes predictions on the test set.
        2. Computes and prints regression metrics.
        3. Plots visualizations
        4. Saves predictions to CSV.

    Args:
        model: Trained Linear Regression model.
        X_test_scaled: Scaled test feature set.
        y_test: True target values for the test set.
        X_test: Original test feature set (unscaled).
        X_train: Original training feature set (unscaled).
        df: Original dataframe used for training/testing.
    Returns:
        results (pd.DataFrame): DataFrame containing actual and predicted values.
    """
    #predictions
    y_pred = model.predict(X_test_scaled)

    # print metrics
    regression_metrics(y_test, y_pred, label="Linear Regression")

    #actual vs predictions plot
    plot_actual_vs_predicted(
        y_test, y_pred,
        title="Linear Regression — Actual vs Predicted"
    )

    plot_actual_vs_predicted_zoom(
        y_test,
        y_pred,
        title="Linear Regression — Actual vs Predicted (Zoomed on Small Values)",
        max_value=150
    )

    #Feature importance (LR uses coefficients) 
    feature_names = get_feature_columns(df)
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
    """ Main function to execute Linear Regression training and evaluation.
    Pipeline:
        1. Load weekly data
        2. Train Linear Regression model
        3. Run evaluation and generate visualizations
        4. Save item-level predictions to CSV
    """
    # Load weekly aggregated sales data
    df = load_weekly_data()

    # Train Linear Regression model
    model, scaler, X_test_scaled, y_test, X_test, X_train = train_model(df)
    # Run evaluation 
    _ = run_evaluation(model, X_test_scaled, y_test, X_test, X_train, df)


if __name__ == "__main__":
    main()