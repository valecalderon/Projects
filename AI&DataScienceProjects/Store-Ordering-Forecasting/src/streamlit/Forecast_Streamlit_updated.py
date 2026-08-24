# ==============================================================
# A Streamlit application to forecast time series data using pretrained models
# main page: gives options to select which model to use, and print out the forecasted results
# ==============================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
import xgboost as xgb
import sklearn
import joblib
import json
from pathlib import Path

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Get project root directory (3 levels up from this file)
ROOT_DIR = Path(__file__).resolve().parents[2]

# Add project root to sys.path so "src" becomes importable
sys.path.append(str(ROOT_DIR))
from src.streamlit.utils import ingredient_mapping_streamlit as ing_map
from src.streamlit.utils import preprocessing_streamlit as prep

import matplotlib.pyplot as plt 

# titles and headers
st.title("Papa John's Order Forecasting")
st.info("Forecasting Future Orders Using Pretrained Models")

# file paths 
BASE_DIR = Path(__file__).resolve().parent

# show raw data 
with st.expander("See Raw Weekly Sales Data"):
    st.write("Past Weekly Sales Data for Items")
    df_raw = pd.read_csv("utils/weekly_item_sales.csv")
    st.dataframe(df_raw)

# load and prepare features
FEATURES = []
model = None
scaler = None

# select model to use for forecasting
st.sidebar.header("Select Model for Forecasting")
model_option = st.sidebar.selectbox(
    "Choose a model:",
    ["Linear Regression", "Support Vector Machine", "XGBoost"]
)

# features paths
MODEL_DIR = BASE_DIR / "Features"

linreg_path = MODEL_DIR / "linear_model.joblib"
svm_path = MODEL_DIR / "svm_model.joblib"
xgb_path = MODEL_DIR / "xgb_model.joblib"

linreg_scaler = MODEL_DIR / "linear_scaler.joblib"
svm_scaler = MODEL_DIR / "svm_scaler.joblib"

feat_path = MODEL_DIR / "features.json"

# load the selected model, scaler, and features
if model_option == "Linear Regression":
    model = joblib.load(linreg_path)
    scaler = joblib.load(linreg_scaler)
    with open(feat_path, "r") as f:
        FEATURES = json.load(f)

elif model_option == "Support Vector Machine":
    model = joblib.load(svm_path)
    scaler = joblib.load(svm_scaler)
    with open(feat_path, "r") as f:
        FEATURES = json.load(f)

elif model_option == "XGBoost":
    model = joblib.load(xgb_path)
    scaler = None # gradient boost does not have a scaler
    with open(feat_path, "r") as f:
        FEATURES = json.load(f)


# load in csv for models and prepare features
@st.cache_data
def load_and_prepare_data():
    df = prep.load_weekly_data()
    return df

df = load_and_prepare_data()

# helper function to align features
def align_features(df, FEATURES):
    X = df.copy()
    for c in FEATURES:
        if c not in X.columns:
            X[c] = 0

    # select only the needed features
    X = X[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return X

# filters for user selection
st.sidebar.header("Filter Data")
store_num = ["ALL"] + sorted(df["store_number"].unique().tolist()) if "store_number" in df.columns else [] # if column exists
year = ["ALL"] + sorted(df["year"].unique().tolist()) if "year" in df.columns else [] # if column exists
week = ["ALL"] + sorted(df["week_of_year"].unique().tolist()) if "week_of_year" in df.columns else [] # if column exists

# user selections for filters
selected_store = st.sidebar.selectbox("Store Number", store_num) if store_num else None
selected_year = st.sidebar.selectbox("Year", year) if year else None
selected_week = st.sidebar.selectbox("Week of Year", week) if week else None

# predict when button is clicked
if st.sidebar.button("Forecast"):
    # load model and scalar based on user selection
    # load features
    mask = pd.Series(True, index=df.index)
    if selected_store != "ALL":
        mask &= df["store_number"].eq(selected_store)

    if selected_year != "ALL":
        mask &= df["year"].eq(selected_year)

    if selected_week != "ALL":
        mask &= df["week_of_year"].eq(selected_week)

    rowset = df.loc[mask].copy() # filter data
    if rowset.empty:
        st.error("No data available for the selected filters.")

    else:
        X = align_features(rowset, FEATURES)
        results = rowset.reset_index(drop=True).copy()

        # linear regression and SVM need scaling
        if model_option in ["Linear Regression", "Support Vector Machine"]:
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)

        elif model_option == "XGBoost":
            X_arr = X.to_numpy(copy=False).astype("float32")
            y_pred = model.predict(X_arr)

        # get results and display
        results["predicted_sales"] = y_pred
        if "weekly_sales" in results.columns:
            results["actual_sales"] = results["weekly_sales"]

        # keep dataframe sticky at top when scrolling (so it doesn't scroll of screen)
        st.subheader(f"Forecasted Weekly Sales using {model_option}")
        st.markdown("""
        <style>
        /* make ALL dataframes sticky at top of the viewport */
        div[data-testid="stDataFrame"] { position: sticky; top: 0, z-index: 100; }
        </style>
        """, unsafe_allow_html=True)
        st.dataframe(results, height=400, use_container_width=True)

        # ingredient mapping dataframe
        st.subheader("Forecasted Ingredients Needed")
        ing_map_csv = pd.read_csv("utils/ingredient_mapping.csv")
        ing_predict_dict = {model_option: results}
        ing_usage_df = ing_map.createIngredientPredictions(ing_map_csv, ing_predict_dict)
        st.dataframe(ing_usage_df, height=400, use_container_width=True)

        # print scatterplot
        if "actual_sales" in results.columns:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.scatter(results["actual_sales"], results["predicted_sales"], alpha=0.5)
            min_val = float(min(results["actual_sales"].min(), results["predicted_sales"].min()))
            max_val = float(max(results["actual_sales"].max(), results["predicted_sales"].max()))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            ax.set_xlabel("Actual Weekly Sales")
            ax.set_ylabel("Predicted Weekly Sales")
            ax.set_title("Predicted vs Actual Weekly Sales")
            st.pyplot(fig)