"""
===============================================================================
Title       : ingredient_forecast_xgb.py
Project     : Papa John's Store Ordering Forecasting
Author      : Alex Dimayuga and Valeria Calderon
Created     : October 12, 2025
Description : 
    This script trains and evaluates a Gradient Boosting model to predict 
    ingredient usage based on historical order data. It processes input 
    features from order headers and details, maps ingredients, and outputs 
    usage forecasts to support inventory planning and supply chain optimization.

Dependencies:
    - pandas
    - numpy
    - xgboost
    - scikit-learn
    - seaborn
    - matplotlib 

Usage:
    Run this script after preprocessing the CSV files:
        1.) cs499_order_header.csv
        2.) cs499_order_detail.csv
        3.) ingredient_mapping.csv
    into
        weekly_sales_item
        

    Example:
        python ingredient_forecast_xgb.py

Notes:
    - Hyperparameters for XGBoost can be tuned for performance.
    - Future enhancements may include multi-store generalization, seasonal 
      adjustment, and integration with real-time ordering systems.
    - TODO: This model could be improved...
        - Hypeparameter tuning
        - Additional feature engineering
        - Algorithm improvements/enhancements
        - Bug fixes
===============================================================================
"""

# Imports
#==============================================================================
import pandas as pd 
import numpy as np
import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT_DIR)
from src.utils.preprocessing import load_weekly_data, split_train_test, get_feature_columns
from src.utils.Metrics_Visuals import regression_metrics, plot_actual_vs_predicted, plot_actual_vs_predicted_zoom, save_item_predictions
from pathlib import Path
#==============================================================================

# Print versions
#==============================================================================
print()
print("Python environment versions:")
print("================================================")
print("Pandas version: ", pd.__version__)
print("Numpy version: ", np.__version__)
print("XGboost version: ", xgb.__version__)
print("scikit-learn version: ", sklearn.__version__)
print("Seaborn version: ", "N/A")
print("Matplotlib version: ", "N/A")
print("================================================")
print()
#==============================================================================

# Global Variables
# =============================================================================
# output file for storing individual item predictions
ITEM_PREDICTIONS_CSV = os.path.join(os.path.dirname(__file__), "xgb_item_predictions.csv")
# =============================================================================

# Main function
#==============================================================================
def main():
    """ Main function to run the XGBoost ingredient forecasting model. 
    Pipeline Steps:
        1. Load and preprocess weekly sales data.
        2. Generate visualizations for exploratory data analysis.
        3. Split data into training and testing sets.
        4. Encode categorical variables.
        5. Train the XGBoost regression model.
        6. Visualize feature importance.
        7. Make predictions on test set
        8. Evaluate model performance using regression metrics.
        9. Save individual item predictions to CSV and plot actual vs predicted values.

        Returns:
            None - saves outputs to files and displays plots.
        """
    print("Starting Ingredient Forecasting with XGBoost...\n")

    # Load weekly data
    df = load_weekly_data()

    # ------------------------------
    # Shared Visualizations
    # ------------------------------
    from src.utils.Metrics_Visuals import (
        print_weekly_sales,
        plot_weekly_trends,
        plot_top_products_weekly,
        plot_weekly_grouped,
        plot_feature_importance,
        plot_residuals
    )

    # Display summary stats for weekly sales
    print_weekly_sales(df)

    # Exploratory visualizations
    plot_weekly_trends(df, group_col="product_mdm_id_norm")
    plot_top_products_weekly(df, product_col="product_mdm_id_norm", top_n=20)
    plot_weekly_grouped(df, "product_family")

    # Split data
    X_train, X_test, y_train, y_test = split_train_test(df)

    # One-hot encode
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    print("Training XGBoost model...\n")
    model.fit(X_train_encoded, y_train)

    # Feature importance visualization
    feature_names = X_train_encoded.columns
    plot_feature_importance(model, feature_names, top_n=20)

    # Make predictions
    y_pred = model.predict(X_test_encoded)

    # Evaluate model
    regression_metrics(y_test, y_pred, label="XGBoost")
    # Residuals plot
    plot_residuals(y_test, y_pred, title="XGBoost Residual Plot")

    # Save individual item predictions
    save_item_predictions(
        X_test,
        y_test,
        y_pred,
        df,
        ITEM_PREDICTIONS_CSV
    )

    # Visualizations
    plot_actual_vs_predicted(y_test, y_pred, title="XGBoost — Actual vs Predicted")
    plot_actual_vs_predicted_zoom(
        y_test,
        y_pred,
        title="XGBoost — Actual vs Predicted (Zoomed on Small Values)",
        max_value=150
    )

if __name__ == "__main__":
    main()
#==============================================================================
