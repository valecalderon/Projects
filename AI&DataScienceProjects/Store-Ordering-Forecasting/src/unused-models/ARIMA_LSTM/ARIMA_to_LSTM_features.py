"""
===============================================================================
Title       : ARIMA_to_LSTM_features.py
Project     : Papa John's Store Ordering Forecasting
Author      : Elizabeth Solie, Valeria Calderon, Alexander Dimayuga
Created     : November 3, 2025
Description : 
    Extract ARIMA-based features, weekly sales predictions and residuals, for 
    each product ID. These outputs are intended as input features for our LSTM
    model.

Dependencies:
    - pandas
    - numpy
    - matplotlib
    - pmdarima
    - statsmodels
    - os

Usage:
    python .\ARIMA_predictions.py

Notes:
    TODO: Hopefully my interpretation of using ARIMA for LSTM is correct...
===============================================================================
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# Helper Functions
def loadAndPreprocess(desired_file):
    # Load CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {current_dir}")

    parent_dir = os.path.dirname(current_dir)
    print(f"Parent directory: {parent_dir}")

    data_file_path = os.path.join(parent_dir, 'data', desired_file)
    print(f"Loading data from: {data_file_path}")

    df = pd.read_csv(data_file_path)

    # Converting Business_date to datetime and set as index
    df['Business_date'] = pd.to_datetime(df['Business_date'])
    df = df.set_index('Business_date')

    # Aggregate weekly sales PER product
    weekly_sales_product = df.groupby('product_mdm_id').resample('W').size().unstack(level=0).fillna(0)

    # Display 
    print(weekly_sales_product)

    return weekly_sales_product

def extractArimaFeatures(df,arima_order=(1,1,1), max_products=10):
    residuals_dict = {}
    arima_preds_dict = {}

    product_ids = df.columns

    metrics_list = []

    for idx, pid in enumerate(product_ids):
        series = df[pid].dropna()

        # Fit ARIMA
        try:
            model = ARIMA(series, order=arima_order)
            result = model.fit()

            # Predict and compute residuals
            pred = result.predict(start=series.index[0], end=series.index[-1])
            residuals = series - pred

            residuals_dict[pid] = residuals
            arima_preds_dict[pid] = pred

            # Calculate performance metrics
            mae = mean_absolute_error(series, pred)
            rmse = mean_squared_error(series, pred, squared=False)
            r2 = r2_score(series, pred)

            metrics_list.append({'product_id': pid, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

            print(f"[:)] ARIMA fitted for Product {pid}")

        except Exception as e:
            print(f"[:(] ARIMA failed for Product {pid}: {e}")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv("ARIMA_metrics.csv", index=False)

    return residuals_dict, arima_preds_dict

def main():
    # 1.) Load and Preprocess 
    df = loadAndPreprocess('CS499_10242025_order_detail.csv') # Change as needed

    # 2.) Extract ARIMA Features for LSTM
    residuals, preds = extractArimaFeatures(df)

    # 3.) Save to CSV
    residuals_df = pd.DataFrame(residuals)
    predictions_df = pd.DataFrame(preds)
    residuals_df.to_csv("ARIMA_residuals.csv", index=True, index_label="Business_date")
    predictions_df.to_csv("ARIMA_predictions.csv", index=True, index_label="Business_date")

    # 4.) Verify 
    df = pd.read_csv("ARIMA_residuals.csv")  # or "ARIMA_predictions.csv"

    # Display basic info
    print("=== Shape ===")
    print(df.shape)

    print("\n=== Column Names ===")
    print(df.columns.tolist())

    print("\n=== First 5 Rows ===")
    print(df.head())

    print("\n=== Index Info ===")
    print(df.index)

if __name__ == "__main__":
    main()
