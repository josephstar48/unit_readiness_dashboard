"""
logistic_models.py

Logistic regression pipeline for multiple readiness / risk targets:
- high_burnout_risk
- high_risk_of_injury
- ucmj
- poor_performance
- suicide_risk
- non_deployable
- profile_status (binarized: on profile vs not)
- soldier_readiness (binarized: low readiness vs others)
"""

import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import matplotlib.pyplot as plt
import seaborn as sns



# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

DATA_PATH = os.path.join("data", "cleaned_soldier_readiness.csv.xlsx")

NUM_FEATURES = [
    "leave_days_taken_past_12mo",
    "days_since_last_leave",
    "years_at_unit",
    "dwell_time",
    "quarterly_counseling_frequency",
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
]

CAT_FEATURES = [
    "marital_status",
    "pcs_in_last_3_years",
    "rank",
    "mos",
    "corps_level",
    "division_level",
    "brigade_level",
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
]

# Target features and how to binarize
TARGET_SPECS: Dict[str, Dict] = {
    "high_burnout_risk": {"type": "binary"},
    "high_risk_of_injury": {"type": "binary"},
    "ucmj": {"type": "binary"},
    "poor_performance": {"type": "binary"},
    "suicide_risk": {"type": "binary"},
    "non_deployable": {"type": "binary"},

    "profile_status": {
        "type": "custom",
        "func": "binarize_profile_status",
    },

     # NEW: risk flags derived from continuous columns
    "low_readiness_risk": {
        "type": "quantile_from_column", 
        "column": "soldier_readiness",
        "q": 0.25
        },

    "low_retention_risk": {
        "type": "threshold_from_column",
        "func": "binarize_retention_rate",
        "column": "retention_rate",
        "threshold": 0.85,
    },
}

# ---------------------------------------------------------
# 2. Functions
# ---------------------------------------------------------

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_excel(path)
    print(f"Data shape: {df.shape}")
    return df


def binarize_profile_status(series: pd.Series) -> pd.Series:
    """
    Convert profile_status to 0/1:
    - 1 = on any profile (temporary or permanent)
    - 0 = no profile / 'none'
    """
    return series.fillna("none").apply(
        lambda x: 1 if str(x).lower() in ["temporary", "permanent"] else 0
    )

def binarize_retention_rate(series: pd.Series, threshold: float = 0.85) -> pd.Series:
    """
    Convert retention_rate (0-1) into a binary label:
    - 1 = retention risk (low retention_rate)
    - 0 = not at risk
    """
    return (series.astype(float) < threshold).astype(int)


def binarize_soldier_readiness(series: pd.Series, q: float = 0.25) -> Tuple[pd.Series, float]:
    """
    Convert continuous soldier_readiness to 0/1:
    - 1 = low readiness (bottom q quantile)
    - 0 = otherwise
    Returns the binary series and the threshold used.
    """
    thresh = series.quantile(q)
    print(f"Soldier readiness quantile threshold (q={q}): {thresh:.3f}")
    binary = (series <= thresh).astype(int)
    return binary, thresh


def make_target(df: pd.DataFrame, target_name: str) -> Tuple[pd.Series, Optional[float]]:
    """
    Build a binary target based on TARGET_SPECS rules.
    Returns: y, optional_threshold
    """
    spec = TARGET_SPECS[target_name]

    if spec["type"] == "binary":
        y = df[target_name].copy()
        # Ensure 0/1 ints
        y = y.astype(int)
        return y, None
    
    # -------------------------
    # Custom binarizers (functions)
    # -------------------------

    elif spec["type"] == "custom":
        func_name = spec["func"]

        if spec["func"] == "binarize_profile_status":
            y = binarize_profile_status(df[target_name])
            return y, None
        
        if func_name == "binarize_retention_rate":
            threshold = spec.get("threshold", 0.85)
            y = binarize_retention_rate(df[target_name], threshold=threshold)
            return y, threshold

        else:
            raise ValueError(f"Unknown custom func in target spec for {target_name}")

    elif spec["type"] == "quantile":
        q = spec.get("q", 0.25)
        y, thresh = binarize_soldier_readiness(df[target_name], q=q)
        return y, thresh
    
    
    # -------------------------
    # Example: low_readiness_risk derived from soldier_readiness column
    # -------------------------

    elif spec["type"] == "quantile_from_column":
        col = spec["column"]
        q = spec.get("q", 0.25)
        thresh = df[col].quantile(q)
        y = (df[col] <= thresh).astype(int)
        return y, thresh
    
    # -------------------------
    # Example: low_retention_risk derived from retention_rate column
    # -------------------------
    elif spec["type"] == "threshold_from_column":
        col = spec["column"]
        threshold = spec.get("threshold", 0.85)
        y = (df[col].astype(float) < threshold).astype(int)
        return y, threshold



    else:
        raise ValueError(f"Unknown target type for {target_name}: {spec['type']}")


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - scales numeric features
    - one-hot encodes categorical features
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_FEATURES),
        ]
    )
    return preprocessor


def build_logistic_pipeline() -> Pipeline:
    """
    Build full pipeline: preprocess + LogisticRegression
    Using class_weight='balanced' to handle class imbalance.
    """
    preprocessor = build_preprocessor()
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", log_reg),
        ]
    )
    return pipe

def plot_conf_matrix(cm: np.ndarray, classes, title: str = "Confusion Matrix"):
    """Simple confusion matrix heatmap."""
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def evaluate_model(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    target_name: str,
    threshold: float,
):
    """Print metrics for one target."""
    print(f"\n=== Evaluation for target: {target_name} (threshold={threshold:.2f}) ===")
    print(classification_report(y_test, y_pred, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_conf_matrix(cm, classes=[0, 1], title=f"Confusion Matrix - {target_name}")

    # ROC-AUC (only if there is more than one class in y_test)
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC: {auc:.3f}")
    else:
        print("ROC-AUC not defined (only one class present in y_test).")


def choose_best_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    candidate_thresholds=None,
) -> float:
    """
    Very simple threshold tuning:
    try multiple thresholds and pick the one maximizing F1-score.
    """
    from sklearn.metrics import f1_score

    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.1, 0.9, 17)  # 0.1, 0.15, ..., 0.9

    best_thresh = 0.5
    best_f1 = -1

    for t in candidate_thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"Best threshold by F1: {best_thresh:.2f} (F1={best_f1:.3f})")
    return best_thresh

# ---------------------------------------------------------
# 2B. Driver Extraction (Top 10) for Command-Level Explainability
# ---------------------------------------------------------

def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Get feature names after preprocessing (numeric + one-hot).
    Output names look like:
      num__stress_score
      cat__morale_high
      cat__rank_E-6
    """
    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if transformer == "drop":
            continue

        # If transformer is a Pipeline, get last step
        if hasattr(transformer, "named_steps"):
            last_step = list(transformer.named_steps.values())[-1]
        else:
            last_step = transformer

        # OneHotEncoder supports get_feature_names_out
        if hasattr(last_step, "get_feature_names_out"):
            out = last_step.get_feature_names_out(cols)
            feature_names.extend([f"{name}__{f}" for f in out])
        else:
            # numeric transformer (StandardScaler): keep original col names
            feature_names.extend([f"{name}__{c}" for c in cols])

    return feature_names


def top_logistic_drivers(pipe: Pipeline, max_drivers: int = 10) -> pd.DataFrame:
    """
    Returns up to max_drivers drivers:
      - Top half increasing risk (highest odds ratio)
      - Top half decreasing risk (lowest odds ratio)

    Adds:
      - coef
      - odds_ratio = exp(coef)
      - pct_change_odds ≈ (odds_ratio - 1) * 100
    """
    preprocessor = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    if not isinstance(model, LogisticRegression):
        raise TypeError("Pipeline model is not LogisticRegression.")

    feat_names = get_feature_names(preprocessor)
    coefs = model.coef_[0]

    df = pd.DataFrame({
        "feature": feat_names,
        "coef": coefs,
        "odds_ratio": np.exp(coefs),
    })

    # Pick top increasers and top decreasers
    inc = df.sort_values("odds_ratio", ascending=False)
    dec = df.sort_values("odds_ratio", ascending=True)

    half = max_drivers // 2
    top = pd.concat([inc.head(half), dec.head(max_drivers - half)], axis=0).copy()

    top["pct_change_odds"] = (top["odds_ratio"] - 1.0) * 100.0

    # Clean names for readability
    top["feature_clean"] = (
        top["feature"]
        .str.replace("^num__", "", regex=True)
        .str.replace("^cat__", "", regex=True)
        .str.replace("=", "_", regex=False)
    )

    # Final sort: strongest risk increasers first
    top = top.sort_values("odds_ratio", ascending=False).reset_index(drop=True)
    return top[["feature_clean", "coef", "odds_ratio", "pct_change_odds"]]


def explain_logistic_drivers(drivers_df: pd.DataFrame, target_name: str) -> None:
    """
    Print command-level interpretation (plain English).
    """
    print(f"\nTop {len(drivers_df)} drivers for {target_name} (Logistic Regression):")
    for _, row in drivers_df.iterrows():
        feat = row["feature_clean"]
        orr = row["odds_ratio"]
        pct = row["pct_change_odds"]

        if orr > 1:
            print(f"- {feat}: increases odds of {target_name} by ~{pct:.0f}% (OR={orr:.2f})")
        else:
            print(f"- {feat}: decreases odds of {target_name} by ~{abs(pct):.0f}% (OR={orr:.2f})")


def save_driver_outputs(drivers_df: pd.DataFrame, target_name: str) -> None:
    """
    Save drivers to reports/ as CSV + a simple bar chart PNG.
    """
    os.makedirs("reports", exist_ok=True)

    # Save CSV
    csv_path = os.path.join("reports", f"drivers_logreg_{target_name}.csv")
    drivers_df.to_csv(csv_path, index=False)
    print(f"Saved top drivers CSV to {csv_path}")

    # Save chart (odds ratio)
    plt.figure(figsize=(8, 5))
    plot_df = drivers_df.copy()

    # For plotting: odds_ratio > 1 means higher risk, < 1 means lower risk
    sns.barplot(data=plot_df, x="odds_ratio", y="feature_clean")
    plt.axvline(1.0, color="red", linestyle="--")
    plt.title(f"Top Drivers (Odds Ratios) — {target_name}")
    plt.xlabel("Odds Ratio (OR)")
    plt.ylabel("Feature")
    plt.tight_layout()

    png_path = os.path.join("reports", f"drivers_logreg_{target_name}.png")
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"Saved top drivers chart to {png_path}")


# ---------------------------------------------------------
# 3. Main training loop for all targets
# ---------------------------------------------------------

def train_and_evaluate_all_targets():
    df = load_data()

    # Drop rows with missing key features (simple approach)
    feature_cols = NUM_FEATURES + CAT_FEATURES
    df_clean = df.dropna(subset=feature_cols).copy()

    print(f"Data shape after dropping rows with missing features: {df_clean.shape}")

    for target_name in TARGET_SPECS.keys():
        print("\n" + "=" * 80)
        print(f"Training logistic regression for target: {target_name}")
        print("=" * 80)

        # Build binary target
        y, _ = make_target(df_clean, target_name)

        # Drop rows where y is NaN (if any)
        mask = ~y.isna()
        X = df_clean.loc[mask, feature_cols]
        y = y.loc[mask].astype(int)

        # Train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        print("Class distribution in train set:")
        print(y_train.value_counts(normalize=True).rename("proportion"))

        # Build pipeline and fit
        pipe = build_logistic_pipeline()
        pipe.fit(X_train, y_train)

        # Predicted probabilities for positive class
        y_prob = pipe.predict_proba(X_test)[:, 1]

        # Pick a good threshold based on F1
        best_threshold = choose_best_threshold(y_test, y_prob)

        # Use chosen threshold
        y_pred = (y_prob >= best_threshold).astype(int)

        # Evaluate
        evaluate_model(y_test, y_pred, y_prob, target_name, best_threshold)

        # ---------------------------------------------------------
        # Top 10 drivers (command-level explanation)
        # ---------------------------------------------------------
        drivers_df = top_logistic_drivers(pipe, max_drivers=10)
        print(drivers_df)
        explain_logistic_drivers(drivers_df, target_name=target_name)
        save_driver_outputs(drivers_df, target_name=target_name)


        # ---------------------------------------------------------
        # Save trained model for deployment or Streamlit use
        # ---------------------------------------------------------
        os.makedirs("models", exist_ok=True)
        model_path = f"models/logreg_{target_name}.pkl"
        joblib.dump(pipe, model_path)
        print(f"Saved model to {model_path}")   





if __name__ == "__main__":
    train_and_evaluate_all_targets()
