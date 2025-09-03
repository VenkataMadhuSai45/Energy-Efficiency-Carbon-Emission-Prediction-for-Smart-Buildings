# Energy Efficiency & Carbon Emission Prediction for Smart Buildings

This project predicts energy usage and carbon emissions for smart buildings using machine learning. It combines building characteristics and weather data to provide actionable insights for sustainability and energy management.

## Features
- Downloads and cleans the UCI Energy Efficiency dataset
- Enriches building data with weather information (Visual Crossing API)
- Exploratory Data Analysis (EDA) with visualizations
- Trains multiple regression models: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM, MLP, Stacking
- Hyperparameter tuning for best models
- Model evaluation and diagnostics (RMSE, R², MAE)
- SHAP and permutation importance for explainability
- Predicts carbon emissions from energy usage
- Exports results and model artifacts

## Project Structure
- `energy.ipynb` — Jupyter notebook with step-by-step code, EDA, modeling, and results



## Data Sources
- [UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx)
- [Visual Crossing Weather API](https://www.visualcrossing.com/weather-api)

## Results
- Model metrics and predictions are saved in the `artifacts` directory.
- EDA and diagnostic plots are saved in the `figures` directory.
- Final predictions and carbon emission reports are exported as CSV files.

## Usage Notes
- For weather enrichment, set your Visual Crossing API key in Colab secrets or as an environment variable.
- The notebook is designed to run in Google Colab or locally with minor adjustments.

