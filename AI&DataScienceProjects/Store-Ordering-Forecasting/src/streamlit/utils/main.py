"""
===============================================================================
Title       : main.py
Project     : Papa John's Store Ordering Forecasting
Author      : Tiffany
Created     : October 12, 2025
Description : 
    Core data preprocessing pipeline that transforms raw Papa John's order data
    into a modeling-ready weekly aggregated dataset. This script performs:
    
    1. Data loading and cleaning from multiple CSV sources
    2. Product ID normalization and key reconciliation
    3. Order-level merge with ingredient mapping (for inspection)
    4. Weekly aggregation by store and item (modeling target)
    5. Ingredient usage calculation from predicted sales
    
    The primary output is a weekly_item_sales.csv file where each row represents
    weekly sales of a specific item at a specific store. This serves as the
    foundation for all ML models (Linear Regression, SVM, XGBoost, Random Forest).

Data Flow:
    Raw CSVs → Clean & Normalize → Merge → Aggregate → Weekly Dataset
    
    Order Detail (sales transactions)
    Order Header (temporal/location metadata)  →  Weekly Item Sales
    Ingredient Mapping (product recipes)
    
Output Files:
    - pre_merged_raw.csv: Order-level data with ingredients (inspection/debugging)
    - weekly_item_sales.csv: Aggregated modeling dataset (target = weekly_sales)

Dependencies:
    - pandas: Data manipulation
    - pathlib: File path handling
    - re: Regular expressions for ID normalization
    
Usage:
    python data.py
    
    This creates the weekly_item_sales.csv file needed by all model training scripts.
===============================================================================
"""
from pathlib import Path
import pandas as pd
import re


# ---------------------------
# Config: input/output files
# ---------------------------
# ORDER_DETAIL_CSV = "cs499_order_detail.csv"
# ORDER_HEADER_CSV = "cs499_order_header.csv"
ORDER_DETAIL_CSV = "CS499_10242025_order_detail.csv"
ORDER_HEADER_CSV = "CS499_10242025_order_header.csv"
INGREDIENT_MAP_CSV = "ingredient_mapping.csv"

PREMERGED_RAW_CSV = "pre_merged_raw.csv"
WEEKLY_ITEM_SALES_CSV = "utils/weekly_item_sales.csv"


# ---------------------------
# Utilities
# ---------------------------
def load_csv(path: str, usecols=None) -> pd.DataFrame:
    """
    Load a CSV with optional selected columns, normalizing column names to lowercase
    """
    df = pd.read_csv(path)
    # Normalize column names to lowercase and strip spaces
    df.columns = df.columns.str.strip().str.lower()

    # If usecols is provided, filter to existing lowercase names
    if usecols is not None:
        lower_usecols = [c.lower() for c in usecols if c.lower() in df.columns]
        df = df[lower_usecols]

    print(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def normalize_mdm(x) -> str:
    """
    Normalize product_mdm_id so 107724.0 -> '107724', ' 107724 ' -> '107724'.
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    return s


def coerce_str(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Force columns to string and strip whitespace."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def parse_business_date(df: pd.DataFrame, col="business_date") -> pd.DataFrame:
    """Parse business_date -> datetime."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ---------------------------
# 1) Load & clean raw tables
# ---------------------------
def load_raw_tables():
    # Order Detail: core sales lines
    od_keep = [
        "business_date",
        "store_number",
        "store_order_number",
        "quantity_sold",
        "product_mdm_id",
        "product_name",
        "product_family",
        "product_class",
        "pizza_size_inches",
    ]
    od = load_csv(ORDER_DETAIL_CSV, usecols=[c for c in od_keep if c is not None])

    # Order Header: optional enrichment (year/week/city/state)
    oh = load_csv(ORDER_HEADER_CSV)  # load all, then trim to what we need
    oh_keep = ["store_order_number", "year", "week_of_year", "store_city", "store_state"]
    oh = oh[[c for c in oh_keep if c in oh.columns]]

    # Ingredient Mapping (long form: one row per product-ingredient)
    im_keep = [
        "product_mdm_id",
        "product_ingredient_id",
        "product_ingredient_name",
        "product_ingredient_amount",
        "product_ingredient_amount_uom",
    ]
    im = load_csv(INGREDIENT_MAP_CSV, usecols=[c for c in im_keep if c is not None])

    # Normalize keys/parse types
    od = coerce_str(od, ["store_order_number"])
    oh = coerce_str(oh, ["store_order_number"])

    od["product_mdm_id_norm"] = od["product_mdm_id"].apply(normalize_mdm)
    im["product_mdm_id_norm"] = im["product_mdm_id"].apply(normalize_mdm)

    od = parse_business_date(od, "business_date")

    # numerics
    if "pizza_size_inches" in od.columns:
        od["pizza_size_inches"] = pd.to_numeric(od["pizza_size_inches"], errors="coerce")
    if "quantity_sold" in od.columns:
        od["quantity_sold"] = pd.to_numeric(od["quantity_sold"], errors="coerce")
    if "product_ingredient_amount" in im.columns:
        im["product_ingredient_amount"] = pd.to_numeric(
            im["product_ingredient_amount"], errors="coerce"
        )

    # quick diagnostics on key overlap
    od_keys = set(od["product_mdm_id_norm"].dropna().unique())
    im_keys = set(im["product_mdm_id_norm"].dropna().unique())
    print(f"🔑 OD uniq product_mdm_id_norm: {len(od_keys)} | IM uniq: {len(im_keys)} | Overlap: {len(od_keys & im_keys)}")

    return od, oh, im


# ---------------------------
# 2) Pre-merged, order-level file (for inspection)
# ---------------------------
def save_premerged_raw(od: pd.DataFrame, oh: pd.DataFrame, im: pd.DataFrame, output_csv=PREMERGED_RAW_CSV) -> pd.DataFrame:
    """
    Create and save a pre-aggregated merged DataFrame (order-level data),
    expanded by ingredient rows (long form).
    """
    print("\n🔍 Building pre-merged raw dataset for inspection...")

    # Merge OD + OH on store_order_number (left join; OD canonical)
    merged = od.merge(oh, on="store_order_number", how="left")

    # Merge with ingredient mapping on normalized product key
    merged = merged.merge(
        im[
            [
                "product_mdm_id_norm",
                "product_ingredient_id",
                "product_ingredient_name",
                "product_ingredient_amount",
                "product_ingredient_amount_uom",
            ]
        ],
        on="product_mdm_id_norm",
        how="left",
    )

    # Compute ingredient_usage (NaN for packaging/label rows with no amount)
    merged["ingredient_usage"] = (
        pd.to_numeric(merged.get("quantity_sold", 0), errors="coerce")
        * pd.to_numeric(merged.get("product_ingredient_amount", 0), errors="coerce")
    )

    # Save output
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"✅ Pre-merged raw dataset saved to: {output_csv}")
    print(f"Shape: {merged.shape}")
    # Small preview
    print(merged.head(8).to_string(index=False))

    # Optional distinct counts for sanity checking joins
    def nunique_safe(df, col):
        return df[col].nunique(dropna=True) if col in df.columns else 0

    print(
        f"Distinct store_order_number: {nunique_safe(merged, 'store_order_number')} | "
        f"product_mdm_id_norm: {nunique_safe(merged, 'product_mdm_id_norm')} | "
        f"product_ingredient_name: {nunique_safe(merged, 'product_ingredient_name')}"
    )

    return merged


# ---------------------------
# 3) Combined sales (derive year/week if header missing)
# ---------------------------
def build_combined_sales(od: pd.DataFrame, oh: pd.DataFrame) -> pd.DataFrame:
    """Left-join Order Detail with Order Header on store_order_number; derive year/week if missing."""
    combined = od.merge(oh, on="store_order_number", how="left")

    iso = combined["business_date"].dt.isocalendar()
    if "year" not in combined.columns:
        combined["year"] = iso.year
    else:
        combined["year"] = combined["year"].fillna(iso.year)

    if "week_of_year" not in combined.columns:
        combined["week_of_year"] = iso.week
    else:
        combined["week_of_year"] = combined["week_of_year"].fillna(iso.week)

    # temporal row features (for later optional use)
    combined["month"] = combined["business_date"].dt.month
    combined["day_of_week"] = combined["business_date"].dt.dayofweek
    combined["is_weekend"] = (combined["day_of_week"] >= 5).astype(int)

    return combined


# ---------------------------
# 4) Weekly, per-store, per-item table (target = weekly_sales)
# ---------------------------
def build_weekly_item_sales(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to per-store, per-week, per-item rows.
    Target y = weekly_sales = sum(quantity_sold).
    """
    df = combined.dropna(subset=["quantity_sold"]).copy()

    grp_keys = [
        "store_number",
        "year",
        "week_of_year",
        "product_mdm_id_norm",   # normalized key for modeling/joins
        "product_mdm_id",        # keep original for reporting/debug if desired
        "product_family",
        "product_class",
        "pizza_size_inches",
    ]

    agg = (
        df.groupby(grp_keys, dropna=False)
          .agg(
              weekly_sales=("quantity_sold", "sum"),
              weekend_ratio=("is_weekend", "mean"),
              avg_month=("month", "mean"),
          )
          .reset_index()
    )

    # Enrich with store city/state if available (many-to-one)
    if {"store_city", "store_state"}.issubset(combined.columns):
        store_meta = combined[["store_number", "store_city", "store_state"]].drop_duplicates(subset=["store_number"])
        agg = agg.merge(store_meta, on="store_number", how="left")

    # Sort for any later lag/window features
    agg = agg.sort_values(["store_number", "product_mdm_id_norm", "year", "week_of_year"])

    # Reorder columns for clarity in the CSV
    cols = [
        "store_number", "year", "week_of_year",
        "product_mdm_id_norm", "product_mdm_id",
        "product_family", "product_class", "pizza_size_inches",
        "weekly_sales", "weekend_ratio", "avg_month",
        "store_city", "store_state"
    ]
    agg = agg[[c for c in cols if c in agg.columns]]
    return agg


# ---------------------------
# 5) Convert predicted item sales → ingredient usage
# ---------------------------
def ingredient_forecast_from_predictions(weekly_item_pred: pd.DataFrame, ingredient_map: pd.DataFrame) -> pd.DataFrame:
    """
    Convert predicted item sales to ingredient usage using ingredient mapping.
    weekly_item_pred must include:
      ['store_number','year','week_of_year','product_mdm_id_norm','predicted_sales']

    Returns per-store, per-week, per-ingredient totals.
    """
    im = ingredient_map[
        ["product_mdm_id_norm", "product_ingredient_name", "product_ingredient_amount"]
    ].dropna(subset=["product_ingredient_amount"]).copy()

    df = weekly_item_pred.merge(im, on="product_mdm_id_norm", how="left")
    df["predicted_ingredient_usage"] = (
        df["predicted_sales"] * df["product_ingredient_amount"]
    )

    out = (
        df.groupby(
            ["store_number", "year", "week_of_year", "product_ingredient_name"],
            dropna=False,
        )
        .agg(total_predicted_usage=("predicted_ingredient_usage", "sum"))
        .reset_index()
    )
    return out


# ---------------------------
# Main runner
# ---------------------------
def main():
    od, oh, im = load_raw_tables()

    # 1) Debug/inspection export (order-level, long with ingredients)
    _ = save_premerged_raw(od, oh, im, PREMERGED_RAW_CSV)

    # 2) Build modeling table (weekly per-store per-item)
    combined = build_combined_sales(od, oh)
    weekly_item = build_weekly_item_sales(combined)

    Path(WEEKLY_ITEM_SALES_CSV).parent.mkdir(parents=True, exist_ok=True)
    weekly_item.to_csv(WEEKLY_ITEM_SALES_CSV, index=False)
    print(f"\nSaved modeling table to: {WEEKLY_ITEM_SALES_CSV}")
    print(f"Shape: {weekly_item.shape}")


if __name__ == "__main__":
    main()
