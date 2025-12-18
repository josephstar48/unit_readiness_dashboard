import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------
# Config
# -----------------------------------------

MODELS_DIR = "models"
OUTPUT_DIR = "reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_N = 10

# -----------------------------------------
# Helper: extract feature names
# -----------------------------------------

def get_feature_names(pipe):
    prep = pipe.named_steps["prep"]

    num_features = prep.transformers_[0][2]
    cat_encoder = prep.transformers_[1][1]
    cat_features = cat_encoder.get_feature_names_out(prep.transformers_[1][2])

    return (
        [f"num__{f}" for f in num_features] +
        [f"cat__{f}" for f in cat_features]
    )

# -----------------------------------------
# Logistic Regression → Odds Drivers
# -----------------------------------------

def plot_logistic_top_drivers(model_path, target_name):
    pipe = joblib.load(model_path)
    model = pipe.named_steps["model"]

    feature_names = get_feature_names(pipe)
    coefs = model.coef_[0]

    df = pd.DataFrame({
        "feature": feature_names,
        "impact": coefs
    })

    df["abs_impact"] = df["impact"].abs()
    top = df.sort_values("abs_impact", ascending=False).head(TOP_N)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=top,
        x="impact",
        y="feature",
        palette="coolwarm"
    )
    plt.title(f"Top {TOP_N} Drivers of {target_name.replace('_',' ').title()}")
    plt.xlabel("Impact on Risk (↑ increases risk)")
    plt.ylabel("")
    plt.tight_layout()

    out = f"{OUTPUT_DIR}/logreg_top10_{target_name}.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved: {out}")

# -----------------------------------------
# Linear Regression → Readiness Drivers
# -----------------------------------------

def plot_linear_top_drivers(model_path, target_name):
    pipe = joblib.load(model_path)
    model = pipe.named_steps["model"]

    feature_names = get_feature_names(pipe)
    coefs = model.coef_

    df = pd.DataFrame({
        "feature": feature_names,
        "impact": coefs
    })

    df["abs_impact"] = df["impact"].abs()
    top = df.sort_values("abs_impact", ascending=False).head(TOP_N)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=top,
        x="impact",
        y="feature",
        palette="viridis"
    )
    plt.title(f"Top {TOP_N} Drivers of {target_name.replace('_',' ').title()}")
    plt.xlabel("Impact on Outcome")
    plt.ylabel("")
    plt.tight_layout()

    out = f"{OUTPUT_DIR}/linreg_top10_{target_name}.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved: {out}")

# -----------------------------------------
# Run for your ASR²D models
# -----------------------------------------

if __name__ == "__main__":

    # Logistic risk models
    LOGISTIC_TARGETS = [
        "high_burnout_risk",
        "high_risk_of_injury",
        "ucmj",
        "poor_performance",
        "suicie_risk",
        "non_deployable",
        "low_readiness_risk",
        "low_retention_risk",
    ]

    for t in LOGISTIC_TARGETS:
        plot_logistic_top_drivers(
            f"{MODELS_DIR}/logreg_{t}.pkl",
            t
        )

    # Linear models
    plot_linear_top_drivers(
        f"{MODELS_DIR}/linreg_soldier_readiness.pkl",
        "soldier_readiness"
    )

    plot_linear_top_drivers(
        f"{MODELS_DIR}/linreg_retention_rate.pkl",
        "retention_rate"
    )
