import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from preprocessing import load_weekly_data, preprocess_for_model, get_feature_columns
from Metrics_Visuals import (
    regression_metrics,
    plot_actual_vs_predicted,
    save_item_predictions,
    plot_residuals,
    print_weekly_sales,
    plot_weekly_trends,
    plot_top_products_weekly,
    plot_weekly_grouped
)

from pathlib import Path
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib
import json

# ---------------------------
# Config
# ---------------------------
# all item-level predictions
ITEM_PREDICTIONS_CSV = os.path.join(os.path.dirname(__file__), "svm_item_predictions.csv")

# ---------------------------
# Train/Test Split & Feature Prep
# ---------------------------

def tune_hyperparameters(X_train_scaled, y_train):
    print("Tuning hyperparameters with GridSearchCV...")
    param_grid = {
        'C': [1, 10, 100],
        'epsilon': [0.1, 0.2, 0.5],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(SVR(kernel="rbf"), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_params_

# ---------------------------
# Train SVM
# ---------------------------
def train_svm(X_train_scaled, y_train, best_params=None):
    print("Training SVM model")
    if best_params:
        model = SVR(kernel="rbf", C=best_params.get('C',10), epsilon=best_params.get('epsilon',0.2), gamma=best_params.get('gamma','scale'))
    else:
        model = SVR(kernel="rbf", C=10, epsilon=0.2)
    model.fit(X_train_scaled, y_train)
    print("Training complete.")
    return model


# ---------------------------
# Evaluate and Save Predictions
# ---------------------------
def evaluate_and_save(model, scaler, X_train, X_test, y_test, df):
    print("Generating predictions")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    regression_metrics(y_test, y_pred)

    test_results = X_test.copy()
    test_results["actual_sales"] = y_test.values
    test_results["predicted_sales"] = y_pred
    test_results["product_mdm_id_norm"] = df.loc[y_test.index, "product_mdm_id_norm"].values
    test_results["store_number"] = df.loc[y_test.index, "store_number"].values
    test_results["year"] = df.loc[y_test.index, "year"].values
    test_results["week_of_year"] = df.loc[y_test.index, "week_of_year"].values

    save_item_predictions(
        X_test,
        y_test,
        y_pred,
        df,
        ITEM_PREDICTIONS_CSV
    )

    plot_actual_vs_predicted(y_test, y_pred, title="SVM: Actual vs Predicted Weekly Sales")

    feature_names = get_feature_columns(df)
    print("SVM Features: ", feature_names)
    # this is to help with streamlit
    FEATURES = feature_names
    # Save feature list to a JSON file
    with open("svm_features.json", "w") as f:
        json.dump(FEATURES, f)


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
    df = load_weekly_data()
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = preprocess_for_model(df)
    best_params = tune_hyperparameters(X_train_scaled, y_train)
    model = train_svm(X_train_scaled, y_train, best_params=best_params)

    joblib.dump(scaler, "svm_scaler.joblib")
    joblib.dump(model, "svm_model.joblib")

    evaluate_and_save(model, scaler, X_train, X_test, y_test, df)


if __name__ == "__main__":
    main()