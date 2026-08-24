"""
===============================================================================
Title       : ingredient_mapping.py
Project     : Papa John's Store Ordering Forecasting
Authors     : Alexander Dimayuga
Created     : November 10, 2025
Description : 
    Maps predicted product sales to ingredient usage and generates weekly 
    ingredient demand forecasts per model

Dependencies:
    - pandas
    - matplotlib
    - os

Usage:
   python ingredient_mapping.py

Notes:
    - TODO: 
        - Refactoring?
        - Potential improvements?
        - Output needs to be verified that it catches everything
===============================================================================
"""
# Imports
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import os
# =============================================================================

# Global Variables
# =============================================================================
ingredient_mapping = "ingredient_mapping.csv"
predictions_to_load = {"SVM": "svm_item_predictions.csv", 
                       "Linear_Regression" : "linear_regression_item_predictions.csv",
                       "gradient-boost" : "xgb_item_predictions.csv"} # TODO: Add more
# =============================================================================

# Helper Functions
# =============================================================================
def loadCSV(file_name, folder_name):
    # Load CSV
    print(f"Loading from: {file_name}")
    print("====================================================================")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {current_dir}")
    parent_dir = os.path.dirname(current_dir)
    print(f"Parent directory: {parent_dir}")
    data_file_path = os.path.join(parent_dir, folder_name , file_name)
    print(f"Loading data from: {data_file_path}")
    print("====================================================================\n")

    df = pd.read_csv(data_file_path)

    # Display 
    print(f"Dataframe from {file_name}:")
    print("====================================================================")
    print(df.head())
    print("====================================================================\n")

    return df


def createIngredientPredictionsCSV(ingredient_df, prediction_dfs, output_file="ingredient_predictions.csv"):
    
    # Multiply product ingredient amounts by predicted sales to get predicted ingredient demand per week.
    
    for model_name, pred_df in prediction_dfs.items():
        print(f"Processing ingredient predictions for model: {model_name}")
        print("====================================================================")

        # Count number of unique product ids for purposes of debugging
        print(f"Ingredient mapping has {ingredient_df['product_mdm_id'].nunique()} unique product_mdm_id values")
        col_name = 'product_mdm_id_norm' if 'product_mdm_id_norm' in pred_df.columns else 'product_mdm_id'
        print(f"Predictions ({model_name}) has {pred_df[col_name].nunique()} unique product_mdm_id_norm values")
        # Check overlap
        overlap = set(ingredient_df['product_mdm_id']).intersection(set(pred_df[col_name]))
        print(f"Overlapping product IDs between ingredient mapping and predictions: {len(overlap)}")

        # Merge ingredient mapping with predicted sales
        merged_df = ingredient_df.merge(pred_df, left_on="product_mdm_id", right_on=col_name, how="left")
        
        # Compute ingredient usage
        merged_df["ingredient_usage"] = merged_df["product_ingredient_amount"] * merged_df["predicted_sales"]
        
        # Pivot to get weeks as columns, ingredients as rows
        pivot_df = merged_df.pivot_table(
            index=["product_ingredient_id", "product_ingredient_name", "product_ingredient_amount_uom"],
            columns="week_of_year",
            values="ingredient_usage",
            aggfunc="sum"  # Sum if multiple products use same ingredient
        ).fillna(0)
        
        # Optional: reset column names to simple format
        pivot_df.columns = [f"Week_{int(col)}" for col in pivot_df.columns]
        
        # Save to CSV
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   output_file.replace(".csv", f"_{model_name}.csv"))
        pivot_df.to_csv(output_path)
        print(f"Saved predicted ingredient usage for {model_name} to: {output_path}")
        print("====================================================================")
# =============================================================================

# Main Function
# =============================================================================
def main():
    # 1.) Load ingredient_mapping.csv as a dataframe
    df = loadCSV(ingredient_mapping, 'data')

    # Count number of unique product ids for purposes of debugging
    num_unique_ingredients = df["product_ingredient_id"].nunique()
    print(f"Number of unique product_ingredient_id values: {num_unique_ingredients}")

    # 2.) Load each model's predictions
    prediction_dfs = {} 
    for model_name, file_name in predictions_to_load.items():
        prediction_df = loadCSV(file_name, model_name)
        prediction_dfs[model_name] = prediction_df
    print("All predictions loaded:")
    print("====================================================================")
    for name, df_pred in prediction_dfs.items():
         print(f"1.) {name}: {df_pred.shape[0]} rows, {df_pred.shape[1]} columns")
    print("====================================================================\n")

    # 3.) Create Ingredient Predictions CSV
    createIngredientPredictionsCSV(df, prediction_dfs)

if __name__ == "__main__":
    main()
# =============================================================================