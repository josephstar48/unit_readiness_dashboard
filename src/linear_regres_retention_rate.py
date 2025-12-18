
"""
linear_regres_retention_rate.py

Regression pipeline for retention_rate (continuous 0–1).

Models:
- LinearRegression (inferential)
- RandomForestRegressor (predictive)

Includes:
- Preprocessing
- Train/test split
- Cross-validation
- RMSE, MAE, R^2
- Residual plots
- Feature importance
- SHAP (RF)
- Saved models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.append('..')
from logistic_regress_models import load_data
from linear_regress_soldier_readiness import build_xgb_pipeline

#---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

DATA_PATH = os.path.join("data", "cleaned_soldier_readiness.csv.xlsx")

TARGET = "retention_rate"

NUM_FEATURES = [
    "age",
    "body_mass_index",
    "overweight",
    "acft_history_average",
    "current_acft_score",
    "acft_failure",
    "airborne_ops",
    "jump_count",
    "musculoskeletal_injury",
    "airborne_related_injury",
    "acft_related_injury",
    "deployments",
    "dwell_time",
    "months_deployed_last_rotation",
    "training_density",
    "stress_score",
    "sleep_score",
    "fatigue",
    "OPTEMPO",
    "family_stress_score",
    "financial_stress_score",
    "unit_climate_score",
    "leadership_trust_index",
    "low_morale",
    "poor_behavioral_health",

    # retention-specific
    "leave_days_taken_past_12mo",
    "days_since_last_leave",
    "years_at_unit",
]

CAT_FEATURES = [
    "type_of_jump",
    "day_or_night_jump",
    "parachute_type",
    "aircraft_type",
    "load",
    "stress_trend",
    "sleep_trend",
    "optempo_trend",
    "behavioral_health",
    "morale",
    "acft_score_trend",

    # retention-specific
    "marital_status",
    "pcs_in_last_3_years",
    "rank",
    "mos",
]

# ---------------------------------------------------------
# 2. Functions
# ---------------------------------------------------------

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_FEATURES),
        ]
    )

def regression_metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{label}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")

def plot_residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.show()

# ---------------------------------------------------------
# 2B. Command-Level: Top 10 Linear Drivers (5 +, 5 -)
# ---------------------------------------------------------

def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer):
    """
    Extract feature names after preprocessing.
    Matches order of transformed matrix:
      num__<col>
      cat__<onehot name>
    """
    feature_names = []

    # numeric
    feature_names.extend([f"num__{c}" for c in NUM_FEATURES])

    # categorical one-hot
    ohe = preprocessor.named_transformers_["cat"]
    cat_out = ohe.get_feature_names_out(CAT_FEATURES)
    feature_names.extend([f"cat__{c}" for c in cat_out])

    return feature_names


def top_linear_drivers(pipe: Pipeline, max_drivers: int = 10) -> pd.DataFrame:
    """
    Returns up to max_drivers coefficients:
      - top half most positive
      - top half most negative
    """
    preprocessor: ColumnTransformer = pipe.named_steps["prep"]
    model: LinearRegression = pipe.named_steps["model"]

    feat_names = get_feature_names_from_preprocessor(preprocessor)
    coefs = model.coef_

    df = pd.DataFrame({"feature": feat_names, "coef": coefs})
    df["feature_clean"] = (
        df["feature"]
        .str.replace("^num__", "", regex=True)
        .str.replace("^cat__", "", regex=True)
        .str.replace("=", "_", regex=False)
    )

    pos = df.sort_values("coef", ascending=False)
    neg = df.sort_values("coef", ascending=True)

    half = max_drivers // 2
    top = pd.concat([pos.head(half), neg.head(max_drivers - half)], axis=0).copy()
    top = top.sort_values("coef", ascending=False).reset_index(drop=True)

    return top[["feature_clean", "coef"]]


def explain_linear_drivers(drivers_df: pd.DataFrame, target_name: str) -> None:
    print(f"\nTop {len(drivers_df)} drivers for {target_name} (Linear Regression):")
    for _, row in drivers_df.iterrows():
        feat = row["feature_clean"]
        coef = row["coef"]
        if coef > 0:
            print(f"- {feat}: associated with HIGHER {target_name} (coef={coef:.3f})")
        else:
            print(f"- {feat}: associated with LOWER {target_name} (coef={coef:.3f})")


def save_linear_driver_outputs(drivers_df: pd.DataFrame, target_name: str) -> None:
    os.makedirs("reports", exist_ok=True)

    csv_path = os.path.join("reports", f"drivers_linreg_{target_name}.csv")
    drivers_df.to_csv(csv_path, index=False)
    print(f"Saved top drivers CSV to {csv_path}")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=drivers_df, x="coef", y="feature_clean")
    plt.axvline(0.0, color="red", linestyle="--")
    plt.title(f"Top Drivers (Linear Coefficients) — {target_name}")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()

    png_path = os.path.join("reports", f"drivers_linreg_{target_name}.png")
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"Saved top drivers chart to {png_path}")



# ---------------------------------------------------------
# 3. Main training & evaluation
# ---------------------------------------------------------

def train_retention_regression():
    df = load_data()

    df = df.dropna(subset=NUM_FEATURES + CAT_FEATURES + [TARGET])

    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear Regression
    lin_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", LinearRegression())
    ])

    lin_pipe.fit(X_train, y_train)

    cv_rmse = -cross_val_score(
        lin_pipe, X_train, y_train,
        cv=5, scoring="neg_root_mean_squared_error"
    ).mean()

    print(f"\nLinear Regression CV RMSE: {cv_rmse:.3f}")

    y_pred_lin = lin_pipe.predict(X_test)
    regression_metrics(y_test, y_pred_lin, "Linear Regression Test Metrics")
    plot_residuals(y_test, y_pred_lin, "Linear Regression Residuals")
      
    # Command-level: Top 10 drivers + save to reports/
    top10_lin = top_linear_drivers(lin_pipe, max_drivers=10)
    print(top10_lin)
    explain_linear_drivers(top10_lin, target_name=TARGET)
    save_linear_driver_outputs(top10_lin, target_name=TARGET)


    os.makedirs("models", exist_ok=True)
    joblib.dump(lin_pipe, "models/linreg_retention_rate.pkl")

    # Random Forest
    rf_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ])

    rf_pipe.fit(X_train, y_train)
    y_pred_rf = rf_pipe.predict(X_test)

    regression_metrics(y_test, y_pred_rf, "Random Forest Test Metrics")
    plot_residuals(y_test, y_pred_rf, "Random Forest Residuals")

    rf_path = os.path.join("models", "rfreg_retention_rate.pkl")
    joblib.dump(rf_pipe, rf_path)
    print(f"\nRandom Forest model saved to {rf_path}")

    # XGBoost Regressor
    build_xgb_pipeline()

    xgb_pipe = build_xgb_pipeline()
    xgb_pipe.fit(X_train, y_train)

    cv_rmse_xgb = cross_val_score(
    xgb_pipe, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
    cv_r2_xgb = cross_val_score(
    xgb_pipe, X_train, y_train, cv=5, scoring="r2")

    print(f"XGB CV RMSE: {(-cv_rmse_xgb.mean()):.3f}")
    print(f"XGB CV R^2:  {cv_r2_xgb.mean():.3f}")

    y_pred_xgb = xgb_pipe.predict(X_test)
    regression_metrics(y_test, y_pred_xgb, "XGBoost Regressor Test Metrics")
    plot_residuals(y_test, y_pred_xgb, "XGBoost Residuals")

    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_pipe, "models/xgbreg_retention_rate.pkl")
    print("Saved XGBoost model to models/xgbreg_retention_rate.pkl")


if __name__ == "__main__":
    train_retention_regression()