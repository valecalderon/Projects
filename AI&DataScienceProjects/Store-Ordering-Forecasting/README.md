# Papa Johns Store Ordering Forecasting
This project is designed to predict ingredient demand at weekly intervals for a single Papa Johns' store. It uses historical sales data, ingredient mappings, preprocessing pipelines, and multiple machine learning models to generate accurate forecasts for dough, cheese, toppings, and other key inventory items. The goal is to help individuals such as store managers and supply chain analysts to optimize weekly ordering, reduce waste, and avoid ingredient shortages.

## Features
- Weekly ingredient demand forecasting
- Multiple ML models (Gradient Boosting, Linear Regression, SVM)
- Evaluation metrics and visualizations 
- Ingredient mapping utilities to convert menu items to ingredient quantities
- Streamlit interface for running forecasts interactively
- Archived experimental models (LSTM and MLP)
### Key considerations
This project was done for thesis of UKY undergraduate project in collaboration with Papa John's. 
Hence, we had privacy documents to fill and cannot publish their store data and it will not be included in the files in this repo.

## Tech Stack
- Python 3.x
- Pandas, NumPy --> data processing
- Scikit-Learn --> primary machine learning models
- XGBoost --> Gradient Boosting 
- Matplotlib/Seaborn --> data visualization
- Streamlit --> UI for running forecasts
- Joblib --> model saving/loading

## Project Structure
```
PAPA-JOHN-S-STORE-ORDERING-FORECASTING
├── data/
│   └── (raw sales data + ingredient mapping)
│
├── src/
│   ├── streamlit/
│   │   └── (Streamlit app files)
│   │
│   ├── training-models/
│   │   ├── gradient-boost/
│   │   ├── Linear_Regression/
│   │   └── SVM/
│   │
│   ├── unused-models/
│   │   ├── ARIMA_LSTM/
│   │   └── MLP/
│   │
│   └── utils/
│       ├── ingredient-mapping/
│       ├── main.py
│       ├── Metrics_Visuals.py
│       └── preprocessing.py
│
├── README.md
└── requirements.txt
```
## Installation
1.) Clone the repository
    
2.) Install the required dependencies
```
    pip install -r requirements.txt
```
