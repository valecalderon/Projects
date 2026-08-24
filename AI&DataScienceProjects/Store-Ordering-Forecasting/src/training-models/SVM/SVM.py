# Import necessary modules and set up paths
import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT_DIR)

from src.utils.preprocessing import load_weekly_data, preprocess_for_model
from src.utils.Metrics_Visuals import (
    regression_metrics,
    plot_actual_vs_predicted,
    save_item_predictions,
    plot_residuals,
    print_weekly_sales,
    plot_weekly_trends,
    plot_top_products_weekly,
    plot_weekly_grouped,
    plot_actual_vs_predicted_zoom
)

from pathlib import Path
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
# all item-level predictions
ITEM_PREDICTIONS_CSV = os.path.join(os.path.dirname(__file__), "svm_item_predictions.csv")

# ---------------------------
# Train/Test Split & Feature Prep
# ---------------------------

def tune_hyperparameters(X_train_scaled, y_train):
    """
    Optimize SVM hyperparameters using GridSearchCV with cross-validation.
    
    Searches over parameter grid to find optimal combination:
    - C: Regularization parameter (controls trade-off between model complexity and training error)
    - epsilon: Width of epsilon-tube (predictions within this margin have zero loss)
    - gamma: Kernel coefficient (influences decision boundary shape)
    
    Args:
        X_train_scaled (np.ndarray): Scaled training features
        y_train (pd.Series): Training target values (weekly sales quantities)
    
    Returns:
        dict: Best hyperparameters found through grid search
            Format: {'C': value, 'epsilon': value, 'gamma': value}
    """

    print("Tuning hyperparameters with GridSearchCV...")
    param_grid = {
        'C': [1, 10, 100],
        'epsilon': [0.1, 0.2, 0.5],
        'gamma': ['scale', 'auto']
    }
    # 3-fold cross-validation
    grid_search = GridSearchCV(SVR(kernel="linear"), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_params_

# ---------------------------
# Train SVM
# ---------------------------
def train_svm(X_train_scaled, y_train, best_params=None):
    """
    Train Support Vector Regression model with linear kernel.
    
    Args:
        X_train_scaled (np.ndarray): Scaled training features
        y_train (pd.Series): Training target values (weekly sales quantities)
        best_params (dict, optional): Optimized hyperparameters from tuning.
                                     If None, uses defaults (C=10, epsilon=0.2)
    
    Returns:
        SVR: Trained Support Vector Regression model
    """
        
    print("Training SVM model")
    if best_params:
        model = SVR(kernel="linear", C=best_params.get('C',10), epsilon=best_params.get('epsilon',0.2))
    else:
        # Default parameters
        model = SVR(kernel="linear", C=10, epsilon=0.2)
    model.fit(X_train_scaled, y_train)
    print("Training complete.")
    return model


# ---------------------------
# Evaluate and Save Predictions
# ---------------------------
def evaluate_and_save(model, scaler, X_train, X_test, y_test, df):
    """
    Generate predictions, evaluate model performance, and create visualizations.
    
    This function orchestrates the complete evaluation pipeline:
    - Scales test features and generates predictions
    - Calculates regression metrics (RMSE, MAE, R²)
    - Creates comprehensive visualizations (actual vs predicted, residuals, trends)
    - Saves item-level predictions for downstream ingredient calculations
    
    Args:
        model (SVR): Trained SVM model
        scaler (StandardScaler): Fitted scaler from training for consistent transformations
        X_train (pd.DataFrame): Training features (for reference)
        X_test (pd.DataFrame): Test features with product identifiers
        y_test (pd.Series): Actual test target values
        df (pd.DataFrame): Original weekly sales dataframe for context
    
    Returns:
        pd.DataFrame: Test results with actual vs predicted sales, product IDs,
                     store numbers, and temporal information (year, week)
    """

    print("Generating predictions")
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # Calculate and print regression metrics
    regression_metrics(y_test, y_pred)

    # Compile test results
    test_results = X_test.copy()
    test_results["actual_sales"] = y_test.values
    test_results["predicted_sales"] = y_pred
    test_results["product_mdm_id_norm"] = df.loc[y_test.index, "product_mdm_id_norm"].values
    test_results["store_number"] = df.loc[y_test.index, "store_number"].values
    test_results["year"] = df.loc[y_test.index, "year"].values
    test_results["week_of_year"] = df.loc[y_test.index, "week_of_year"].values

    # Save item-level predictions
    save_item_predictions(
        X_test,
        y_test,
        y_pred,
        df,
        ITEM_PREDICTIONS_CSV
    )

    # Visualizations
    plot_actual_vs_predicted(y_test, y_pred, title="SVM: Actual vs Predicted Weekly Sales")
    plot_actual_vs_predicted_zoom(y_test, y_pred, title="SVM: Actual vs Predicted (Zoomed on Small Values)", max_value=150)

    # Residuals plot
    plot_residuals(y_test, y_pred, title="SVM Residual Plot")

    # Weekly sales pivot table
    print_weekly_sales(df)

    # Weekly trend plot
    plot_weekly_trends(df)

    # Top product weekly trends
    plot_top_products_weekly(df, product_col="product_mdm_id_norm", top_n=20)

    # Generic grouped weekly plot (by product family if available)
    if "product_family" in df.columns:
        plot_weekly_grouped(df, "product_family")

    return test_results


def main():
    """
    Main execution pipeline for SVM-based ingredient forecasting.
    
    Pipeline:
        1. Load preprocessed weekly sales data
        2. Preprocess features (scaling, encoding)
        3. Tune hyperparameters using GridSearchCV
        4. Train SVM model with optimized parameters
        5. Evaluate performance and save predictions
    """
    # load data and preprocess
    df = load_weekly_data()
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = preprocess_for_model(df)
    # hyperparameter tuning
    best_params = tune_hyperparameters(X_train_scaled, y_train)
    # train model
    model = train_svm(X_train_scaled, y_train, best_params=best_params)
    # evaluate and save predictions
    evaluate_and_save(model, scaler, X_train, X_test, y_test, df)


if __name__ == "__main__":
    main()