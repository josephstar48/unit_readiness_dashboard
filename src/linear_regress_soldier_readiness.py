"""
linear_regress_models.py

Regression pipeline for soldier_readiness (continuous target):

- LinearRegression (interpretable)
- RandomForestRegressor (non-linear comparison)

Includes:
- Preprocessing (StandardScaler + OneHotEncoder)
- Train/test split
- Cross-validation (RMSE, R^2)
- Test metrics (RMSE, MAE, R^2)
- Residual plots
- Feature importance (coef_ and feature_importances_)
- SHAP explanation (for RandomForest)
- Model saving to models/ folder
"""

import os
from typing import List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


import joblib

import sys
sys.path.append('..')
from logistic_regress_models import load_data



# SHAP for explainability
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: shap not installed. SHAP explanations will be skipped.")


#---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

DATA_PATH = os.path.join("data", "cleaned_soldier_readiness.csv.xlsx")

# Using the same independent feature sets as in the logistic script
NUM_FEATURES: List[str] = [
    "stress_score",
    "sleep_score",
    "fatigue",
    "OPTEMPO",
    "family_stress_score",
    "financial_stress_score",
    "unit_climate_score",
    "leadership_trust_index",
    "acft_failure",
    "low_morale",
    "poor_behavioral_health",
]

CAT_FEATURES: List[str] = [
    "stress_trend",
    "sleep_trend",
    "optempo_trend",
    "behavioral_health",
    "morale",
    "acft_score_trend",
]

TARGET = "soldier_readiness"


# ---------------------------------------------------------
# 2. Functions
# ---------------------------------------------------------

# def load_data(path: str = DATA_PATH) -> pd.DataFrame:
#     print(f"Loading data from {path}...")
#     df = pd.read_excel(path)
#     print(f"Data shape: {df.shape}")
#     return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build ColumnTransformer for:
    - numeric: StandardScaler
    - categorical: OneHotEncoder
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_FEATURES),
        ]
    )
    return preprocessor


def build_linear_pipeline() -> Pipeline:
    """
    Pipeline: preprocess -> LinearRegression
    """
    preprocessor = build_preprocessor()
    lin_reg = LinearRegression()

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", lin_reg),
        ]
    )
    return pipe


def build_rf_pipeline(random_state: int = 42) -> Pipeline:
    """
    Pipeline: preprocess -> RandomForestRegressor
    """
    preprocessor = build_preprocessor()
    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", rf_model),
        ]
    )
    return pipe


def build_xgb_pipeline(random_state: int = 42) -> Pipeline:
    """
    Pipeline: preprocess -> XGBRegressor
    Works well with mixed numeric + one-hot encoded categorical features.
    """
    preprocessor = build_preprocessor()

    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", xgb),
        ]
    )
    return pipe


def print_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str):
    """
    Print RMSE, MAE, and R^2 for regression model.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n=== {name} Test Metrics ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R^2:  {r2:.3f}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title_prefix: str):
    """
    Plot residuals vs predicted and residual distribution.
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 4))

    # Residuals vs predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"{title_prefix}: Residuals vs Predicted")

    # Residual distribution
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.title(f"{title_prefix}: Residual Distribution")
    plt.xlabel("Residual")

    plt.tight_layout()
    plt.show()


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract full feature names after preprocessing.
    num__feature_name, cat__<encoded_feature_name>
    """
    feature_names = []

    # Numeric
    # num_transformer = preprocessor.named_transformers_["num"]
    num_features_out = [f"num__{name}" for name in NUM_FEATURES]
    feature_names.extend(num_features_out)

    # Categorical
    cat_transformer = preprocessor.named_transformers_["cat"]
    # OneHotEncoder has get_feature_names_out
    cat_feature_names = cat_transformer.get_feature_names_out(CAT_FEATURES)
    cat_features_out = [f"cat__{name}" for name in cat_feature_names]
    feature_names.extend(cat_features_out)

    return feature_names

# ---------------------------------------------------------
# 2B. Command-Level: Top 10 Linear Drivers (5 +, 5 -)
# ---------------------------------------------------------

def top_linear_drivers(pipe: Pipeline, max_drivers: int = 10) -> pd.DataFrame:
    """
    Returns up to max_drivers coefficients:
      - top half most positive (increase target)
      - top half most negative (decrease target)

    NOTE: since numeric features are scaled with StandardScaler, numeric coefficients
    are roughly "effect per 1 standard deviation increase" (good for comparing drivers).
    """
    preprocessor: ColumnTransformer = pipe.named_steps["prep"]
    model: LinearRegression = pipe.named_steps["model"]

    feature_names = get_feature_names_from_preprocessor(preprocessor)
    coefs = model.coef_

    df = pd.DataFrame({"feature": feature_names, "coef": coefs})
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

    # sort with strongest positive first
    top = top.sort_values("coef", ascending=False).reset_index(drop=True)
    return top[["feature_clean", "coef"]]


def explain_linear_drivers(drivers_df: pd.DataFrame, target_name: str) -> None:
    """
    Print command-friendly explanation:
    - Positive coef => increases target
    - Negative coef => decreases target
    """
    print(f"\nTop {len(drivers_df)} drivers for {target_name} (Linear Regression):")
    for _, row in drivers_df.iterrows():
        feat = row["feature_clean"]
        coef = row["coef"]
        if coef > 0:
            print(f"- {feat}: associated with HIGHER {target_name} (coef={coef:.3f})")
        else:
            print(f"- {feat}: associated with LOWER {target_name} (coef={coef:.3f})")


def save_linear_driver_outputs(drivers_df: pd.DataFrame, target_name: str) -> None:
    """
    Save CSV + bar chart PNG to reports/
    """
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



def show_linear_feature_importance(pipe: Pipeline, title: str = "Linear Regression Coefficients"):
    """
    Display sorted coefficients from Linear Regression.
    """
    preprocessor: ColumnTransformer = pipe.named_steps["prep"]
    model: LinearRegression = pipe.named_steps["model"]

    feature_names = get_feature_names_from_preprocessor(preprocessor)
    coefs = model.coef_

    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    print("\nTop 20 Linear Coefficients (by absolute value):")
    print(coef_df.head(20)[["feature", "coef"]])

    plt.figure(figsize=(8, 6))
    top_n = coef_df.head(15)
    sns.barplot(data=top_n, x="coef", y="feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def show_rf_feature_importance(pipe: Pipeline, title: str = "Random Forest Feature Importance"):
    """
    Display sorted feature importances from RandomForestRegressor.
    """
    preprocessor: ColumnTransformer = pipe.named_steps["prep"]
    model: RandomForestRegressor = pipe.named_steps["model"]

    feature_names = get_feature_names_from_preprocessor(preprocessor)
    importances = model.feature_importances_

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)

    print("\nTop 20 Random Forest Feature Importances:")
    print(imp_df.head(20))

    plt.figure(figsize=(8, 6))
    top_n = imp_df.head(15)
    sns.barplot(data=top_n, x="importance", y="feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def show_shap_explanations(pipe: Pipeline, X_train: pd.DataFrame, sample_size: int = 500):
    """
    Compute and display SHAP explanations for RandomForestRegressor.
    Only runs if shap is installed.
    """
    if not HAS_SHAP:
        print("\nSHAP not available (shap library not installed). Skipping SHAP explanations.")
        return

    print("\nComputing SHAP explanations (RandomForestRegressor)...")

    preprocessor: ColumnTransformer = pipe.named_steps["prep"]
    rf_model: RandomForestRegressor = pipe.named_steps["model"]

    # Transform X_train to model input space
    X_train_transformed = preprocessor.transform(X_train)

    # Sample for speed
    if X_train_transformed.shape[0] > sample_size:
        idx = np.random.choice(X_train_transformed.shape[0], sample_size, replace=False)
        X_sample = X_train_transformed[idx]
    else:
        X_sample = X_train_transformed

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)

    feature_names = get_feature_names_from_preprocessor(preprocessor)

    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=True)


# ---------------------------------------------------------
# 3. Main training & evaluation
# ---------------------------------------------------------

def train_and_evaluate_regression():
    # Load data
    df = load_data()

    # Drop rows with missing features or target
    feature_cols = NUM_FEATURES + CAT_FEATURES
    df_clean = df.dropna(subset=feature_cols + [TARGET]).copy()

    print(f"Data shape after dropping rows with missing values: {df_clean.shape}")

    X = df_clean[feature_cols]
    y = df_clean[TARGET].astype(float)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # -----------------------------------------------------
    # A) Linear Regression
    # -----------------------------------------------------
    lin_pipe = build_linear_pipeline()
    lin_pipe.fit(X_train, y_train)

    # Cross-validation (negative RMSE and R^2)
    print("\n=== Linear Regression Cross-Validation ===")
    # 5-fold CV
    cv_rmse = cross_val_score(
        lin_pipe,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error",
    )
    cv_r2 = cross_val_score(
        lin_pipe,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
    )
    print(f"CV RMSE (mean ± std): {(-cv_rmse.mean()):.3f} ± {cv_rmse.std():.3f}")
    print(f"CV R^2 (mean ± std): {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

    # Test predictions & metrics
    y_pred_lin = lin_pipe.predict(X_test)
    print_regression_metrics(y_test.values, y_pred_lin, name="Linear Regression")
    plot_residuals(y_test.values, y_pred_lin, title_prefix="Linear Regression")
    show_linear_feature_importance(lin_pipe, title="Linear Regression Feature Effects")

    # -----------------------------------------------------
    # Command-level: Top 10 linear drivers + save to reports/
    # -----------------------------------------------------
    top10_lin = top_linear_drivers(lin_pipe, max_drivers=10)
    print(top10_lin)
    explain_linear_drivers(top10_lin, target_name=TARGET)
    save_linear_driver_outputs(top10_lin, target_name=TARGET)

    # Save linear regression model
    os.makedirs("models", exist_ok=True)
    lin_path = os.path.join("models", "linreg_soldier_readiness.pkl")
    joblib.dump(lin_pipe, lin_path)
    print(f"Saved Linear Regression model to {lin_path}")

    # -----------------------------------------------------
    # B) RandomForestRegressor
    # -----------------------------------------------------
    rf_pipe = build_rf_pipeline()
    rf_pipe.fit(X_train, y_train)

    print("\n=== Random Forest Cross-Validation ===")
    cv_rmse_rf = cross_val_score(
        rf_pipe,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error",
    )
    cv_r2_rf = cross_val_score(
        rf_pipe,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
    )

    print(f"CV RMSE (mean ± std): {(-cv_rmse_rf.mean()):.3f} ± {cv_rmse_rf.std():.3f}")
    print(f"CV R^2 (mean ± std): {cv_r2_rf.mean():.3f} ± {cv_r2_rf.std():.3f}")

    # Test predictions & metrics
    y_pred_rf = rf_pipe.predict(X_test)
    print_regression_metrics(y_test.values, y_pred_rf, name="Random Forest Regressor")
    plot_residuals(y_test.values, y_pred_rf, title_prefix="Random Forest")
    show_rf_feature_importance(rf_pipe, title="Random Forest Feature Importance")

    # Save RandomForest model
    rf_path = os.path.join("models", "rfreg_soldier_readiness.pkl")
    joblib.dump(rf_pipe, rf_path)
    print(f"Saved Random Forest model to {rf_path}")

    # SHAP explanations for RandomForest
    show_shap_explanations(rf_pipe, X_train)

    # -----------------------------------------------------
    # C) XGBoost Regressor
    # -----------------------------------------------------
    xgb_pipe = build_xgb_pipeline()
    xgb_pipe.fit(X_train, y_train)

    print("\n=== XGBoost Cross-Validation ===")

    cv_rmse_xgb = cross_val_score(
        xgb_pipe,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error",
    )

    cv_r2_xgb = cross_val_score(
        xgb_pipe,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
    )

    print(f"CV RMSE (mean ± std): {(-cv_rmse_xgb.mean()):.3f} ± {cv_rmse_xgb.std():.3f}")
    print(f"CV R^2 (mean ± std): {cv_r2_xgb.mean():.3f} ± {cv_r2_xgb.std():.3f}")

    # Test predictions & metrics
    y_pred_xgb = xgb_pipe.predict(X_test)
    print_regression_metrics(y_test.values, y_pred_xgb, name="XGBoost Regressor")
    plot_residuals(y_test.values, y_pred_xgb, title_prefix="XGBoost")

    # Save model
    os.makedirs("models", exist_ok=True)
    xgb_path = os.path.join("models", "xgbreg_soldier_readiness.pkl")  
    joblib.dump(xgb_pipe, xgb_path)
    print(f"Saved XGBoost model to {xgb_path}")


if __name__ == "__main__":
    train_and_evaluate_regression()
