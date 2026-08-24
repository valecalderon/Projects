"""
===============================================================================
Title       : LSTM_weekly_forecast.py
Project     : Papa John's Store Ordering Forecasting
Author      : Elizabeth Solie, Valeria Calderon, Alexander Dimayuga
Created     : November 4, 2025
Description : 
    TODO: Complete

Dependencies:
    TODO: Complete

Usage:
    TODO: Complete

Notes:
    TODO: Complete
===============================================================================
"""

# Imports

import pandas as pd
import sys
from sklearn.metrics import mean_squared_error
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from Metrics_Visuals import regression_metrics, plot_actual_vs_predicted, plot_residuals

# Helper Functions
def printFormat(message):
    print("#===============================================================================")
    print(message)
    print("#===============================================================================")
    print()

def loadFile(folder, desired_file):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {current_dir}")

    parent_dir = os.path.dirname(current_dir)
    print(f"Parent directory: {parent_dir}")

    data_file_path = os.path.join(parent_dir, folder, desired_file)
    print(f"Loading data from: {data_file_path}")

    return data_file_path
    
def preprocessDataframe(df):
    # TODO: Adjust preprocessing and columns 

    # Standardize columns (strip and lower)
    df.columns = df.columns.str.strip().str.lower()

    # Convert business_date to datetime (flexible format)
    df["business_date"] = pd.to_datetime(df["business_date"], errors="coerce")
    valid_dates = df["business_date"].notna().sum()
    

    # Ensure quantity_sold is numeric
    df["quantity_sold"] = pd.to_numeric(df["quantity_sold"], errors="coerce").fillna(0)

    # Extract time-based features
    df['day_of_week'] = df['business_date'].dt.dayofweek
    df['month'] = df['business_date'].dt.month
    df['is_weekend'] = df['day_of_week'] >= 5
    df['week_of_year'] = df['business_date'].dt.isocalendar().week

    # Aggregate daily to weekly
    weekly_sales_df = df.groupby(
    ["product_mdm_id", "product_name", "product_family", "week_of_year"]
    ).agg(
        weekly_sales=('quantity_sold', 'sum'),
        month=('month', 'first'),  # month of first day in week
        weekend_fraction=('day_of_week', lambda x: (x >= 5).mean())  # fraction of weekend days in week
    ).reset_index()
    
    # Ensure product_mdm_id is preserved when merging full week range
    all_weeks = (
        weekly_sales_df[['product_mdm_id']]
        .drop_duplicates()
        .assign(key=1)
        .merge(pd.DataFrame({'week_of_year': range(1, 53), 'key': 1}), on='key')
        .drop('key', axis=1)
    )

    # Merge back product info and fill missing weeks with 0 sales
    weekly_sales_df = pd.merge(
        all_weeks,
        weekly_sales_df,
        on=['product_mdm_id', 'week_of_year'],
        how='left'
    )

    # Add lag features per product
    weekly_sales_df = weekly_sales_df.sort_values(['product_mdm_id', 'week_of_year'])
    weekly_sales_df['lag_1'] = weekly_sales_df.groupby('product_mdm_id')['weekly_sales'].shift(1).fillna(0)
    weekly_sales_df['lag_2'] = weekly_sales_df.groupby('product_mdm_id')['weekly_sales'].shift(2).fillna(0)
    weekly_sales_df['rolling_mean_4'] = weekly_sales_df.groupby('product_mdm_id')['weekly_sales'] \
                                        .transform(lambda x: x.shift(1).rolling(4).mean())

    # Drop rows with NaN in lag features (first weeks per product)
    weekly_sales_df = weekly_sales_df.dropna().reset_index(drop=True)

    # Print summary
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("Weekly-aggregated dataframe preview:")
    print("===============================================================================")
    print(weekly_sales_df.head())
    print("===============================================================================")
    print()
    print("Dataframe Summary:")
    print("===============================================================================")
    print("Number of unique products:", weekly_sales_df['product_name'].nunique())
    print("Number of unique weeks:", weekly_sales_df['week_of_year'].nunique())
    print("Total rows for modeling:", len(weekly_sales_df))
    print("===============================================================================")
    print()

    # Add year column if missing (assume data is from 2025 if not provided)
    if 'year' not in weekly_sales_df.columns:
        weekly_sales_df['year'] = 2025

    # Reconstruct approximate business_date (Sunday of each week)
    weekly_sales_df['business_date'] = pd.to_datetime(
        '2025' + weekly_sales_df['week_of_year'].astype(str) + '-0', 
        format='%Y%W-%w', 
        errors='coerce'
    )


    return weekly_sales_df


# Helper: Train a multi-product LSTM with embedding for product_mdm_id
def train_lstm_on_merged(merged_df, look_back=4, epochs=20, batch_size=32):
    """
    Train a single LSTM model across all products, using an embedding for product_mdm_id
    and numeric features (lags + ARIMA features). Evaluates on last 2 weeks and saves predictions.
    """
    from keras.models import Model
    from keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Reshape
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np
    import pandas as pd
    import os

    # Select features and target
    feature_cols = ['lag_1','lag_2','rolling_mean_4','month','weekend_fraction','arima_prediction','arima_residual']
    target_col = 'weekly_sales'

    # Prepare dataframe: drop rows without target, sort by product/date, and fill missing features
    df = merged_df.copy()
    df = df.dropna(subset=[target_col])
    if 'business_date' in df.columns:
        df = df.sort_values(['product_mdm_id','business_date'])
    else:
        df = df.sort_values(['product_mdm_id','year','week_of_year'])
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # Encode product_mdm_id as category codes for embedding
    df['product_code'] = df['product_mdm_id'].astype('category').cat.codes
    n_products = df['product_code'].nunique()

    # Scale numeric features globally
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Build sequences for all products
    def make_sequences(df, look_back):
        X_num, X_pid, y, meta = [], [], [], []
        grouped = df.groupby('product_code')
        for pid, g in grouped:
            if len(g) < look_back + 2:
                continue
            g = g.reset_index(drop=True)
            for i in range(look_back, len(g)):
                X_num.append(g.loc[i - look_back:i - 1, feature_cols].values)
                X_pid.append(pid)
                y.append(g.loc[i, target_col])
                # Save metadata for mapping predictions
                meta.append({
                    'product_mdm_id': g.loc[i, 'product_mdm_id'],
                    'business_date': g.loc[i, 'business_date'] if 'business_date' in g.columns else None,
                    'year': g.loc[i, 'year'] if 'year' in g.columns else None,
                    'week_of_year': g.loc[i, 'week_of_year'] if 'week_of_year' in g.columns else None
                })
        return np.array(X_num), np.array(X_pid), np.array(y), meta

    X_num, X_pid, y, meta = make_sequences(df, look_back)

    # Split into train/test by date: last 2 weeks for test (across all products)
    # use the week_of_year/year to find the latest 2 weeks
    meta_df = pd.DataFrame(meta)
    if 'business_date' in meta_df.columns and meta_df['business_date'].notnull().any():
        sorted_idx = np.argsort(meta_df['business_date'].values)
        meta_df_sorted = meta_df.iloc[sorted_idx]
        unique_dates = meta_df_sorted['business_date'].dropna().sort_values().unique()
        test_dates = unique_dates[-2:]
        test_mask = meta_df['business_date'].isin(test_dates)
    else:
        # Fallback: use year and week_of_year
        meta_df['year'] = pd.to_numeric(meta_df['year'], errors='coerce')
        meta_df['week_of_year'] = pd.to_numeric(meta_df['week_of_year'], errors='coerce')
        meta_df = meta_df.sort_values(['year', 'week_of_year'])
        unique_weeks = meta_df[['year','week_of_year']].drop_duplicates().values
        test_weeks = unique_weeks[-2:]
        test_mask = [
            (y == test_weeks[0][0] and w == test_weeks[0][1]) or
            (y == test_weeks[1][0] and w == test_weeks[1][1])
            for y, w in zip(meta_df['year'], meta_df['week_of_year'])
        ]
        test_mask = np.array(test_mask)

    X_train_num, X_test_num = X_num[~test_mask], X_num[test_mask]
    X_train_pid, X_test_pid = X_pid[~test_mask], X_pid[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    meta_test = meta_df[test_mask].reset_index(drop=True)

    # Model definition
    pid_in = Input(shape=(1,), name='product_id')
    emb = Embedding(input_dim=n_products, output_dim=8, name='product_embedding')(pid_in)
    emb_r = Reshape((8,))(emb)
    num_in = Input(shape=(look_back, len(feature_cols)), name='num_features')
    lstm_out = LSTM(32)(num_in)
    concat = Concatenate()([lstm_out, emb_r])
    out = Dense(1)(concat)
    model = Model(inputs=[num_in, pid_in], outputs=out)
    model.compile(optimizer='adam', loss='mse')

    # Train model
    model.fit([X_train_num, X_train_pid], y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Predict and evaluate
    y_pred = model.predict([X_test_num, X_test_pid], verbose=0).flatten()

    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"[LSTM-Global] Evaluation Metrics:")
    print(f"  MAE  = {mae:.3f}")
    print(f"  MSE  = {mse:.3f}")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  R²   = {r2:.3f}")

    # Use shared regression metrics + plots
    regression_metrics(y_test, y_pred, label="LSTM (Global)")
    plot_actual_vs_predicted(y_test, y_pred, title="LSTM — Actual vs Predicted")
    plot_residuals(y_test, y_pred, title="LSTM — Residual Plot")

    # Save predictions: include product_code, product_mdm_id, business_date, y_true, y_pred
    pred_df = pd.DataFrame({
        'product_code': X_test_pid,
        'product_mdm_id': meta_test['product_mdm_id'].values,
        'business_date': meta_test['business_date'].values,
        'year': meta_test['year'].values,
        'week_of_year': meta_test['week_of_year'].values,
        'y_true': y_test,
        'y_pred': y_pred
    })
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lstm_predictions.csv')
    pred_df.to_csv(save_path, index=False)
    print(f"[LSTM-Global] Saved predictions to: {save_path}")
    return pred_df




# Helper function: Load and preprocess the original order detail CSV
def load_and_preprocess_original():
    unprocessed_original_df = pd.read_csv(loadFile('data', 'CS499_10242025_order_detail.csv'))
    original_df = preprocessDataframe(unprocessed_original_df)
    return original_df

# Helper function: Load, clean, and merge ARIMA predictions/residuals into a single dataframe
def load_and_prepare_arima_features():
    arima_predictions_df = pd.read_csv(loadFile('ARIMA_LSTM', 'ARIMA_predictions.csv'))
    arima_residuals_df = pd.read_csv(loadFile('ARIMA_LSTM', 'ARIMA_residuals.csv'))

    # Clean up ARIMA CSVs
    for df_name, df in [('predictions', arima_predictions_df), ('residuals', arima_residuals_df)]:
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'Business_date'}, inplace=True)
        # Strip whitespace and drop blank Business_date values
        df['Business_date'] = df['Business_date'].astype(str).str.strip()
        df = df[df['Business_date'].str.len() > 0]
        # Parse Business_date with known format (YYYY-MM-DD)
        df['Business_date'] = pd.to_datetime(df['Business_date'], format='%Y-%m-%d', errors='coerce')
        # Drop any rows where parsing failed
        df.dropna(subset=['Business_date'], inplace=True)
        print(f"[DEBUG] {df_name} Business_date unique count:", df['Business_date'].nunique())
        if df_name == 'predictions':
            arima_predictions_df = df
        else:
            arima_residuals_df = df

    # Convert both dataframes from wide to long format using pd.melt
    arima_pred_long = pd.melt(arima_predictions_df, id_vars=['Business_date'], var_name='product_mdm_id', value_name='arima_prediction')
    arima_resid_long = pd.melt(arima_residuals_df, id_vars=['Business_date'], var_name='product_mdm_id', value_name='arima_residual')

    # Convert product_mdm_id to numeric with errors='coerce', drop NaNs, then to int
    for long_df, value_col in [(arima_pred_long, 'arima_prediction'), (arima_resid_long, 'arima_residual')]:
        long_df['product_mdm_id'] = pd.to_numeric(long_df['product_mdm_id'], errors='coerce')
        long_df.dropna(subset=['product_mdm_id'], inplace=True)
        long_df['product_mdm_id'] = long_df['product_mdm_id'].astype(int)
        # Also coerce Business_date again to datetime, just in case
        long_df['Business_date'] = pd.to_datetime(long_df['Business_date'], errors='coerce')

    # Merge predictions and residuals on Business_date and product_mdm_id
    arima_features = pd.merge(
        arima_pred_long,
        arima_resid_long,
        on=['Business_date', 'product_mdm_id'],
        how='inner'
    )
    # Add week_of_year and year columns
    arima_features['week_of_year'] = arima_features['Business_date'].dt.isocalendar().week
    arima_features['year'] = arima_features['Business_date'].dt.isocalendar().year
    return arima_features

# Helper function: Merge preprocessed original dataframe with ARIMA features and print preview
def merge_with_arima(original_df, arima_features):
    # Add year to original_df if not present
    if 'year' not in original_df.columns:
        if 'Business_date' in original_df.columns:
            original_df['Business_date'] = pd.to_datetime(original_df['Business_date'], errors='coerce')
            original_df['year'] = original_df['Business_date'].dt.isocalendar().year
    merged_df = pd.merge(
        original_df,
        arima_features,
        on=['product_mdm_id', 'week_of_year', 'year'],
        how='left'
    )
    print("Merged dataframe preview:")
    print(merged_df.head())
    return merged_df


def main():
    printFormat("Starting LSTM Weekly Forecast...")
    original_df = load_and_preprocess_original()
    arima_features = load_and_prepare_arima_features()
    merged_df = merge_with_arima(original_df, arima_features)
    lstm_preds = train_lstm_on_merged(merged_df, look_back=4, epochs=20, batch_size=32)
    if not lstm_preds.empty:
        print("LSTM predictions preview:")
        print(lstm_preds.head())


if __name__ == "__main__":
    main()

