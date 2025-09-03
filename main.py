# ==============================================
# Energy & CO₂ Prediction for Smart Buildings
# Visual Crossing weather merge + HDD/CDD + SHAP
# Author: VMS (with ChatGPT)
# ==============================================
"""
Run this as a notebook or a Python script (e.g., in PyCharm)

What you get
------------
• Loads UCI Energy Efficiency dataset
• Expands across cities & dates; fetches weather from Visual Crossing
• Engineers HDD/CDD and other features
• EDA: distributions, correlation heatmap, scatter plots
• Model zoo: Linear/Ridge/Lasso/ElasticNet, Polynomial, Tree-based (RF/GBM/XGB/LGBM), SVR, MLP, Voting/Stacking
• Evaluation: MAE, RMSE, R² + residuals and pred-vs-actual plots
• Explainability: SHAP (tree-based best model) + permutation importance fallback
• Carbon emissions = Energy × Emission Factor (EF)
• Saves artifacts: models, metrics, figures

Quick start
-----------
1) pip install -r requirements.txt  (see below for minimal list)
2) Set your Visual Crossing key as env var:  export VC_API_KEY="YOUR_KEY" (Linux/macOS) or setx VC_API_KEY "YOUR_KEY" (Windows)
   - If you don't have a key yet, set USE_WEATHER=False to run without enrichment.
3) Run this file top-to-bottom.

Project structure (suggested)
----------------------------
project/
  ├─ data/
  ├─ artifacts/
  ├─ figures/
  ├─ notebooks/  (optional)
  ├─ src/        (optional)
  ├─ main.py     (this file)
  ├─ requirements.txt
  └─ README.md
"""

# =============== 0) Imports & Setup ===============
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# SHAP
import shap

# Paths
os.makedirs("artifacts", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Reproducibility
RNG = 42
np.random.seed(RNG)

# =============== 1) Config ===============
# Visual Crossing
VC_API_KEY = os.getenv("VC_API_KEY", "")
USE_WEATHER = True if VC_API_KEY else False

# Cities & date window: adjust based on API quota
CITIES = [
    "Bengaluru, IN",
    "Delhi, IN",
    "London, UK",
    "Singapore"
]
WEATHER_START = "2023-01-01"
WEATHER_END   = "2023-01-12"  # keep short to respect free quota

# Emission factor (kg CO2 per kWh)
EMISSION_FACTOR = 0.233

# UCI dataset URL
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
UCI_PATH = os.path.join("data", "ENB2012_data.xlsx")

# =============== 2) Helpers ===============

def download_uci_if_needed():
    if not os.path.exists(UCI_PATH):
        print("Downloading UCI Energy Efficiency dataset …")
        r = requests.get(UCI_URL, timeout=30)
        r.raise_for_status()
        with open(UCI_PATH, "wb") as f:
            f.write(r.content)
    print("UCI dataset ready:", UCI_PATH)


def get_visual_crossing_daily(city: str, start_date: str, end_date: str, key: str,
                               unit_group: str = "metric",
                               elements: str = "datetime,temp,humidity,dew,precip,pressure,cloudcover,solarradiation,windspeed" ) -> pd.DataFrame:
    """Fetch daily weather with Visual Crossing timeline API."""
    base = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{start_date}/{end_date}"
    params = {
        "unitGroup": unit_group,
        "key": key,
        "include": "days",
        "elements": elements,
        "contentType": "json",
    }
    resp = requests.get(base, params=params, timeout=60)
    resp.raise_for_status()
    js = resp.json()
    rows = []
    for d in js.get("days", []):
        row = {k: d.get(k, np.nan) for k in elements.split(',')}
        row["city"] = city
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
        df.drop(columns=["datetime"], inplace=True)
    return df


def compute_hdd_cdd(temp_c: pd.Series, base_c: float = 18.0) -> pd.DataFrame:
    """Compute Heating and Cooling Degree Days for a daily average temperature series."""
    hdd = np.maximum(0, base_c - temp_c)
    cdd = np.maximum(0, temp_c - base_c)
    return pd.DataFrame({"HDD": hdd, "CDD": cdd})


# =============== 3) Load & Clean UCI ===============
download_uci_if_needed()
uci = pd.read_excel(UCI_PATH)
uci.columns = [
    'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
    'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution',
    'Heating_Load', 'Cooling_Load'
]

# Basic checks
assert uci.isna().sum().sum() == 0, "UCI dataset unexpectedly has missing values."
uci['Orientation'] = uci['Orientation'].astype(int)
uci['Glazing_Area_Distribution'] = uci['Glazing_Area_Distribution'].astype(int)

# Targets
uci['Energy_Usage_kWh'] = uci['Heating_Load'] + uci['Cooling_Load']
uci['Carbon_Emissions_kg'] = uci['Energy_Usage_kWh'] * EMISSION_FACTOR

# =============== 4) Weather Merge (Visual Crossing) ===============
if USE_WEATHER:
    print("Fetching Visual Crossing weather …")
    weather_frames = []
    for city in CITIES:
        print(f"  -> {city} {WEATHER_START} to {WEATHER_END}")
        dfw = get_visual_crossing_daily(city, WEATHER_START, WEATHER_END, VC_API_KEY)
        weather_frames.append(dfw)
        time.sleep(1)
    weather = pd.concat(weather_frames, ignore_index=True)

    # HDD/CDD
    hddcdd = compute_hdd_cdd(weather['temp'])
    weather = pd.concat([weather, hddcdd], axis=1)

    # Fill any tiny gaps
    for c in ['humidity','dew','precip','pressure','cloudcover','solarradiation','windspeed','HDD','CDD']:
        if c in weather.columns:
            weather[c] = weather[c].astype(float)
            weather[c] = weather[c].interpolate().bfill().ffill()

    # Build Cartesian product (buildings × city × date)
    uci['_key'] = 1
    city_dates = weather[['city', 'date']].drop_duplicates().copy()
    city_dates['_key'] = 1
    expanded = uci.merge(city_dates, on='_key', how='left').drop(columns=['_key'])

    # Join weather
    data_merged = expanded.merge(weather, on=['city','date'], how='left')
else:
    print("VC_API_KEY not set. Skipping weather enrichment. (Set USE_WEATHER=True by exporting VC_API_KEY)")
    # Put placeholders so pipeline still runs
    uci['city'] = 'NoWeather'
    uci['date'] = pd.to_datetime('2023-01-01').date()
    for c in ['temp','humidity','dew','precip','pressure','cloudcover','solarradiation','windspeed']:
        uci[c] = np.nan
    uci[['HDD','CDD']] = compute_hdd_cdd(pd.Series(uci['temp']))
    data_merged = uci.copy()

print("Merged shape:", data_merged.shape)

# =============== 5) EDA ===============
# Save distributions for key numeric columns
num_cols_eda = [
    'Relative_Compactness','Surface_Area','Wall_Area','Roof_Area','Overall_Height','Glazing_Area',
    'Heating_Load','Cooling_Load','Energy_Usage_kWh','temp','humidity','solarradiation','HDD','CDD'
]

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(14, 12))
axes = axes.flatten()
for i, col in enumerate([c for c in num_cols_eda if c in data_merged.columns][:16]):
    axes[i].hist(data_merged[col].dropna(), bins=30)
    axes[i].set_title(col)
plt.tight_layout(); plt.savefig("figures/01_distributions.png", dpi=140); plt.close()

# Correlation heatmap (numeric only)
num_corr_cols = data_merged.select_dtypes(include=[np.number]).columns
corr = data_merged[num_corr_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap (numeric)')
plt.tight_layout(); plt.savefig("figures/02_corr_heatmap.png", dpi=140); plt.close()

# Scatter with target
for col in ['Surface_Area','Glazing_Area','temp','solarradiation','HDD','CDD']:
    if col in data_merged.columns:
        plt.figure(figsize=(6,4))
        plt.scatter(data_merged[col], data_merged['Energy_Usage_kWh'], s=10, alpha=0.6)
        plt.xlabel(col); plt.ylabel('Energy_Usage_kWh'); plt.title(f'Energy vs {col}')
        plt.tight_layout(); plt.savefig(f"figures/03_scatter_energy_{col}.png", dpi=140); plt.close()

# =============== 6) Modeling Data Prep ===============
FEATURES = [
    'Relative_Compactness','Surface_Area','Wall_Area','Roof_Area','Overall_Height',
    'Orientation','Glazing_Area','Glazing_Area_Distribution',
    'temp','humidity','dew','precip','pressure','cloudcover','solarradiation','windspeed','HDD','CDD','city'
]
FEATURES = [c for c in FEATURES if c in data_merged.columns]
TARGET = 'Energy_Usage_kWh'
ALT_TARGET = 'Carbon_Emissions_kg'

model_df = data_merged[FEATURES + [TARGET, ALT_TARGET]].copy()
model_df = model_df.dropna(subset=[TARGET])

X = model_df[FEATURES]
y = model_df[TARGET]

# Define numeric & categorical feature sets
num_feats = [c for c in FEATURES if c not in ['city','Orientation','Glazing_Area_Distribution']]
cat_feats = [c for c in FEATURES if c in ['city','Orientation','Glazing_Area_Distribution']]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_feats),
        ('cat', categorical_transformer, cat_feats)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RNG)

# =============== 7) Model Zoo ===============
models = {
    'Linear': Pipeline([('pre', preprocess), ('est', LinearRegression())]),
    'Ridge': Pipeline([('pre', preprocess), ('est', Ridge(alpha=1.0, random_state=RNG))]),
    'Lasso': Pipeline([('pre', preprocess), ('est', Lasso(alpha=0.001, max_iter=10000, random_state=RNG))]),
    'ElasticNet': Pipeline([('pre', preprocess), ('est', ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RNG, max_iter=10000))]),
    'Polynomial(2)': Pipeline([('pre', preprocess), ('poly', PolynomialFeatures(degree=2, include_bias=False)), ('est', Ridge(alpha=1.0))]),
    'RandomForest': Pipeline([('pre', preprocess), ('est', RandomForestRegressor(n_estimators=400, random_state=RNG, n_jobs=-1))]),
    'GBM': Pipeline([('pre', preprocess), ('est', GradientBoostingRegressor(random_state=RNG))]),
    'XGB': Pipeline([('pre', preprocess), ('est', XGBRegressor(n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, max_depth=6, reg_lambda=1.0, n_jobs=-1, random_state=RNG))]),
    'LightGBM': Pipeline([('pre', preprocess), ('est', LGBMRegressor(n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=RNG, n_jobs=-1))]),
    'SVR(RBF)': Pipeline([('pre', preprocess), ('est', SVR(kernel='rbf', C=10, gamma='scale'))]),
    'MLP': Pipeline([('pre', preprocess), ('est', MLPRegressor(hidden_layer_sizes=(128,64), activation='relu', max_iter=500, random_state=RNG))])
}

# Optionally, a stacked ensemble (uses strong base learners)
stack = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=400, random_state=RNG, n_jobs=-1)),
        ('xgb', XGBRegressor(n_estimators=400, learning_rate=0.05, n_jobs=-1, random_state=RNG)),
        ('lgb', LGBMRegressor(n_estimators=400, learning_rate=0.05, n_jobs=-1, random_state=RNG))
    ],
    final_estimator=Ridge(alpha=1.0)
)
models['Stacking'] = Pipeline([('pre', preprocess), ('est', stack)])

# =============== 8) Train & Evaluate ===============
metrics = []
trained = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)
    r2 = r2_score(y_test, pred)
    metrics.append((name, mae, rmse, r2))
    trained[name] = pipe
    print(f"{name:14s} | MAE {mae:7.3f} | RMSE {rmse:7.3f} | R2 {r2:6.3f}")

metrics_df = pd.DataFrame(metrics, columns=["Model","MAE","RMSE","R2"]).sort_values("R2", ascending=False)
metrics_df.to_csv("artifacts/metrics_energy.csv", index=False)
print("\nTop models by R²:\n", metrics_df.head())

best_name = metrics_df.iloc[0]['Model']
best_model = trained[best_name]

import pickle
with open("artifacts/best_energy_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# =============== 9) Diagnostics Plots ===============
y_pred = best_model.predict(X_test)
resid = y_test - y_pred

# Pred vs Actual
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, alpha=0.6)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims)
plt.xlabel('Actual Energy_Usage_kWh')
plt.ylabel('Predicted Energy_Usage_kWh')
plt.title(f'Predicted vs Actual — {best_name}')
plt.tight_layout(); plt.savefig("figures/04_pred_vs_actual.png", dpi=150); plt.close()

# Residuals vs Fitted
plt.figure(figsize=(6,5))
plt.scatter(y_pred, resid, alpha=0.6)
plt.axhline(0, linestyle='--')
plt.xlabel('Fitted (Predicted)')
plt.ylabel('Residuals')
plt.title(f'Residuals vs Fitted — {best_name}')
plt.tight_layout(); plt.savefig("figures/05_residuals.png", dpi=150); plt.close()

# Residual distribution
plt.figure(figsize=(6,5))
plt.hist(resid, bins=40)
plt.xlabel('Residual')
plt.ylabel('Count')
plt.title(f'Residual Distribution — {best_name}')
plt.tight_layout(); plt.savefig("figures/06_residual_hist.png", dpi=150); plt.close()

# =============== 10) Explainability (SHAP + Permutation) ===============
# Build a small sample for SHAP speed
X_sample = X_test.sample(min(500, len(X_test)), random_state=RNG)

# Try SHAP for tree-based models
try:
    est = best_model.named_steps['est']
    # Need the transformed matrix and feature names out of the preprocessor
    pre = best_model.named_steps['pre']
    Xtr = pre.transform(X_sample)

    # Get output feature names from ColumnTransformer
    out_names = []
    for name, trans, cols in pre.transformers_:
        if name == 'num':
            out_names += list(cols)
        elif name == 'cat':
            ohe = trans.named_steps['ohe']
            cat_in_cols = trans.named_steps['imputer'].feature_names_in_
            cats = ohe.categories_
            for c_in, c_vals in zip(cat_in_cols, cats):
                out_names += [f"{c_in}={v}" for v in c_vals]

    if hasattr(est, 'get_booster') or hasattr(est, 'feature_importances_'):
        # TreeExplainer works well for tree models
        explainer = shap.Explainer(est)
        shap_vals = explainer(Xtr)
        shap.plots.beeswarm(shap_vals, show=False)
        plt.title(f'SHAP Beeswarm — {best_name}')
        plt.tight_layout(); plt.savefig("figures/07_shap_beeswarm.png", dpi=150); plt.close()
        shap.plots.bar(shap_vals, show=False)
        plt.title(f'SHAP Bar — {best_name}')
        plt.tight_layout(); plt.savefig("figures/08_shap_bar.png", dpi=150); plt.close()
    else:
        raise RuntimeError("Best model is not tree-based; using permutation importance.")
except Exception as e:
    print("[INFO] SHAP tree explainability skipped or failed:", e)
    # Fallback: permutation importance on the whole pipeline
    from sklearn.inspection import permutation_importance
    r = permutation_importance(best_model, X_sample, y.loc[X_sample.index], n_repeats=10, random_state=RNG, n_jobs=-1)
    importances = r.importances_mean
    order = np.argsort(importances)[::-1][:30]
    labels = np.array(out_names)[order]
    plt.figure(figsize=(8,10))
    plt.barh(labels[::-1], importances[order][::-1])
    plt.title(f'Permutation Importance — {best_name}')
    plt.tight_layout(); plt.savefig("figures/09_perm_importance.png", dpi=150); plt.close()

# =============== 11) Carbon Emission Outputs ===============
# Since Carbon Emissions = Energy × EF, we can produce CO₂ predictions directly
co2_pred = y_pred * EMISSION_FACTOR

# Save a small report
report = pd.DataFrame({
    'y_true_energy': y_test,
    'y_pred_energy': y_pred,
    'pred_co2_kg'  : co2_pred
})
report.to_csv("artifacts/test_predictions_energy_co2.csv", index=False)

print("\nAll done. Figures saved in ./figures, models & metrics in ./artifacts")

# =============== 12) Minimal requirements.txt suggestion (print) ===============
print("\nrequirements.txt (suggested):\n"
      "pandas\n"
      "numpy\n"
      "scikit-learn\n"
      "matplotlib\n"
      "seaborn\n"
      "xgboost\n"
      "lightgbm\n"
      "shap\n"
      "requests\n"
      "openpyxl\n")

# =============== 13) README.md template (print to console) ===============
README = f"""
# Energy & CO₂ Prediction for Smart Buildings

**Goal:** Predict daily building energy consumption (kWh) and carbon emissions (kg CO₂) using building design + weather (Visual Crossing), with HDD/CDD features and full ML pipeline.

## Highlights
- Visual Crossing weather merge (daily) for multiple cities & dates
- HDD/CDD engineered from avg temperature
- Model zoo (Linear → XGB/LightGBM → Stacking) with RMSE/MAE/R²
- Diagnostics: residuals, pred-vs-actual
- Explainability: SHAP (tree-based) + permutation importance fallback
- CO₂ = Energy × EF (EF = {EMISSION_FACTOR} kg/kWh)

## Quick Start
```bash
pip install -r requirements.txt
export VC_API_KEY=YOUR_KEY   # Windows: setx VC_API_KEY YOUR_KEY
python main.py
```

## Config
- Cities & dates: edit `CITIES`, `WEATHER_START`, `WEATHER_END` in the script
- To run without weather: unset `VC_API_KEY` (the script creates placeholders)

## Files
- `figures/` — EDA, correlations, residuals, SHAP/perm importance
- `artifacts/` — `best_energy_model.pkl`, `metrics_energy.csv`, predictions CSV

## Notes
- The UCI dataset lacks real dates/locations; we simulate climate exposure by expanding across chosen cities and a date window — common in energy design studies.
- Replace EF with a grid- or time-specific factor if available to get dynamic CO₂.

## License
MIT (or your preference)
"""
print("\nREADME.md template:\n\n" + README)
