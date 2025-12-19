import os
import glob
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from typing import Dict, Tuple, List, Optional

from scipy import sparse

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False



# -------------------------------
# App Config
# -------------------------------
st.set_page_config(
    page_title="ASR²D — Airborne Soldier Readiness & Risk Prediction Dashboard",
    layout="wide",
)


# --- High-visibility radio styles (sidebar-safe) ---
st.markdown("""
<style>
/* Scope to sidebar only */
section[data-testid="stSidebar"] .stRadio > label {
  font-weight: 700;
  font-size: 1rem;
}

/* Radio container spacing */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
  display: block;
  margin: 6px 0;
}

/* Individual radio options */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input + div {
  /* Make clickable area larger */
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid var(--radio-border, #9aa0a6);
}

/* Light mode colors */
@media (prefers-color-scheme: light) {
  section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input + div {
    background: #ffffff;
    color: #1f1f1f;
    border-color: #bfc5d2;
  }
  section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input:checked + div {
    background: #e8f0fe;
    border-color: #3b82f6; /* bright blue */
    color: #0b1d3a;
    font-weight: 700;
  }
}

/* Dark mode colors */
@media (prefers-color-scheme: dark) {
  section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input + div {
    background: #121317;
    color: #e6e6e6;
    border-color: #2b2f3a;
  }
  section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input:checked + div {
    background: #1f2937; /* slate */
    border-color: #60a5fa; /* lighter blue */
    color: #ffffff;
    font-weight: 700;
  }
}

/* Hover & focus for better affordance */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input + div:hover {
  filter: brightness(1.02);
  border-color: #60a5fa;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input:focus + div {
  outline: 2px solid #60a5fa;
}

</style>
""", unsafe_allow_html=True)



# Hypothesis test
try:
    from scipy.stats import ttest_ind
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False



APP_TITLE = "🪂Airborne Soldier Readiness & Risk Prediction Dashboard (ASR²D)"

DEFAULT_DATA_CANDIDATES = [
    os.path.join("data", "cleaned_soldier_readiness.csv.xlsx"),

]

MODELS_DIR = "models"

# Model name conventions used 
LOGREG_PREFIX = "logreg_"
LINREG_READINESS = "linreg_soldier_readiness.pkl"
LINREG_RETENTION = "linreg_retention_rate.pkl"
RFREG_READINESS = "rfreg_soldier_readiness.pkl"
RFREG_RETENTION = "rfreg_retention_rate.pkl"
XGBREG_READINESS = "xgbreg_soldier_readiness.pkl"
XGBREG_RETENTION = "xgbreg_retention_rate.pkl"


# -------------------------------
# Utilities
# -------------------------------
@st.cache_data
def load_dataset(uploaded_file=None) -> pd.DataFrame:
    """Load dataset from uploader or from default paths."""
    if uploaded_file is not None:
        # Determine type
        name = uploaded_file.name.lower()
        if name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        return df

    # try defaults
    for p in DEFAULT_DATA_CANDIDATES:
        if os.path.exists(p):
            if p.lower().endswith(".xlsx"):
                return pd.read_excel(p)
            else:
                return pd.read_csv(p)

    raise FileNotFoundError(
        "No dataset found. Upload it in the sidebar or put it in data/ (xlsx or csv)."
    )


@st.cache_resource
def load_models(models_dir: str = MODELS_DIR) -> Dict[str, object]:
    """Load all models we can find in /models."""
    models = {}

    if not os.path.exists(models_dir):
        return models

    # Logistic models
    for path in glob.glob(os.path.join(models_dir, f"{LOGREG_PREFIX}*.pkl")):
        name = os.path.basename(path).replace(".pkl", "")
        models[name] = joblib.load(path)

    # Regression models (optional)
    for fname in [
        LINREG_READINESS, LINREG_RETENTION,
        RFREG_READINESS, RFREG_RETENTION,
        XGBREG_READINESS, XGBREG_RETENTION,
    ]:
        p = os.path.join(models_dir, fname)
        if os.path.exists(p):
            models[fname.replace(".pkl", "")] = joblib.load(p)

    return models


def safe_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def commander_caption(text: str):
    st.caption(f"🪖 **Commanders' note:** {text}")


def plot_hist(series: pd.Series, title: str, bins: int = 30):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    s = series.dropna()
    ax.hist(s, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(series.name if series.name else "")
    ax.set_ylabel("Count")
    st.pyplot(fig, clear_figure=True)


# def plot_bar_counts(series: pd.Series, title: str):
#     fig, ax = plt.subplots(figsize=(6, 3.2))
#     vc = series.fillna("Unknown").astype(str).value_counts().head(20)
#     ax.bar(vc.index, vc.values)
#     ax.set_title(title)
#     ax.set_xlabel(series.name if series.name else "")
#     ax.set_ylabel("Count")
#     ax.tick_params(axis="x", rotation=45)
#     st.pyplot(fig, clear_figure=True)


def get_feature_names_from_pipe(pipe) -> np.ndarray:
    """
    Best effort feature name extraction from sklearn ColumnTransformer.
    Works on modern sklearn via get_feature_names_out().
    """
    prep = pipe.named_steps.get("prep", None)
    if prep is None:
        return np.array([])

    try:
        return prep.get_feature_names_out()
    except Exception:
        # fallback: empty
        return np.array([])


def build_row_from_base(df: pd.DataFrame, base_idx: int) -> pd.DataFrame:
    """Return a single-row DF for prediction based on selected Soldier row."""
    row = df.iloc[[base_idx]].copy()
    return row


def apply_overrides(row: pd.DataFrame, overrides: Dict[str, object]) -> pd.DataFrame:
    """Override selected columns in a one-row DF."""
    for k, v in overrides.items():
        if k in row.columns:
            row.loc[row.index[0], k] = v
    return row


def predict_logistic(pipe, X_one_row: pd.DataFrame) -> float:
    """Return probability of class 1."""
    prob = pipe.predict_proba(X_one_row)[:, 1][0]
    return float(prob)


def predict_regression(pipe, X_one_row: pd.DataFrame) -> float:
    pred = pipe.predict(X_one_row)[0]
    return float(pred)


def explain_top10_contributions(pipe, X_one_row: pd.DataFrame, model_kind: str) -> pd.DataFrame:
    """
    Command-friendly “Why?” panel:
    - Logistic: coef * transformed_feature_value (approx local driver)
    - Linear:   coef * transformed_feature_value (approx local driver)
    """
    prep = pipe.named_steps.get("prep")
    model = pipe.named_steps.get("model")

    if prep is None or model is None:
        return pd.DataFrame()

    # Transform row
    try:
        x_t = prep.transform(X_one_row)
    except Exception:
        # Sometimes requires same columns as training; if missing, explanation won't work.
        return pd.DataFrame()

    # Get feature names
    feat_names = get_feature_names_from_pipe(pipe)
    if feat_names.size == 0:
        return pd.DataFrame()

    # Get coefficients
    if model_kind == "logistic":
        coefs = model.coef_[0]
    else:
        coefs = model.coef_

    # align shapes
    x_vec = np.array(x_t).ravel()
    if len(x_vec) != len(coefs) or len(coefs) != len(feat_names):
        return pd.DataFrame()

    contrib = coefs * x_vec
    df = pd.DataFrame({
        "feature": feat_names,
        "contribution": contrib,
        "direction": np.where(contrib >= 0, "↑ increases", "↓ decreases")
    })
    df["abs_contribution"] = df["contribution"].abs()
    top = df.sort_values("abs_contribution", ascending=False).head(10).drop(columns=["abs_contribution"])

    return top


def top10_bar(df_top: pd.DataFrame, title: str):
    if df_top.empty:
        st.info("Explain panel unavailable (missing columns or feature names).")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    # sort for nicer bar
    d = df_top.sort_values("contribution")
    ax.barh(d["feature"].astype(str), d["contribution"])
    ax.set_title(title)
    ax.set_xlabel("driver strength (coef × input)")
    ax.set_ylabel("")
    st.pyplot(fig, clear_figure=True)

@st.cache_resource
def get_tree_explainer(_model):
    # Cache explainer so it doesn't rebuild every rerun
    return shap.TreeExplainer(_model)


def observed_rate(df: pd.DataFrame, target_col: str) -> Optional[float]:
    if target_col not in df.columns:
        return None
    s = df[target_col].dropna()
    if s.empty:
        return None
    # if already 0/1
    try:
        return float(s.astype(int).mean())
    except Exception:
        return None


def choose_base_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Sidebar filters leaders can use to look at subgroup rates.
    Returns filtered df and filter values.
    """
    filters = {}

    cols = {
        "corps_level": "Corps",
        "division_level": "Division",
        "brigade_level": "Brigade",
        "mos": "MOS",
        "rank": "Rank",
    }

    filtered = df.copy()

    for col, label in cols.items():
        if col in df.columns:
            opts = ["All"] + sorted(df[col].dropna().astype(str).unique().tolist())
            sel = st.sidebar.selectbox(label, opts, index=0)
            filters[col] = sel
            if sel != "All":
                filtered = filtered[filtered[col].astype(str) == sel]

    return filtered, filters

def get_preprocessed_feature_names(pipe) -> np.ndarray:
    prep = pipe.named_steps.get("prep", None)
    if prep is None:
        return np.array([])
    try:
        return prep.get_feature_names_out()
    except Exception:
        return np.array([])

def top_logistic_coefficients(pipe, top_n: int = 10) -> pd.DataFrame:
    model = pipe.named_steps.get("model", None)
    if model is None or not hasattr(model, "coef_"):
        return pd.DataFrame()

    feats = get_preprocessed_feature_names(pipe)
    coefs = model.coef_[0]

    if feats.size == 0 or len(feats) != len(coefs):
        return pd.DataFrame()

    dfc = pd.DataFrame({"feature": feats, "coef": coefs})
    dfc["abs_coef"] = dfc["coef"].abs()
    dfc = dfc.sort_values("abs_coef", ascending=False).head(top_n).drop(columns="abs_coef")
    dfc["effect"] = np.where(dfc["coef"] > 0, "↑ increases non-deployable risk", "↓ decreases risk")
    return dfc


# -------------------------------
# Unit aggregation helpers (NEW)
# -------------------------------

UNIT_HIERARCHY = ["division_level", "brigade_level"]

def unit_selectors(df: pd.DataFrame, prefix: str = "unit"):
    """
    Cascading selectors for Division -> Brigade.
    Only shows selectors that exist in df.columns.
    Returns selected values dict.
    """
    sel = {}

    # DIVISION
    if "division_level" in df.columns:
        div_opts = ["All"] + sorted(df["division_level"].dropna().astype(str).unique().tolist())
        sel["division_level"] = st.selectbox("Division", div_opts, key=f"{prefix}_division")
        df1 = df if sel["division_level"] == "All" else df[df["division_level"].astype(str) == sel["division_level"]]
    else:
        df1 = df

    # BRIGADE
    if "brigade_level" in df.columns:
        bde_opts = ["All"] + sorted(df1["brigade_level"].dropna().astype(str).unique().tolist())
        sel["brigade_level"] = st.selectbox("Brigade", bde_opts, key=f"{prefix}_brigade")
        df2 = df1 if sel["brigade_level"] == "All" else df1[df1["brigade_level"].astype(str) == sel["brigade_level"]]
    else:
        df2 = df1

    return sel


def filter_unit(df: pd.DataFrame, sel: Dict[str, str]) -> pd.DataFrame:
    """Filter df using a selection dict from unit_selectors()."""
    dff = df.copy()
    for col, val in sel.items():
        if col in dff.columns and val and val != "All":
            dff = dff[dff[col].astype(str) == str(val)]
    return dff


def infer_feature_cols_from_model(pipe) -> List[str]:
    """
    Pull raw input feature names from a trained pipeline if possible.
    This avoids hardcoding FEATURE_COLS in the app.
    """
    prep = pipe.named_steps.get("prep")
    if prep is None:
        return []
    try:
        # ColumnTransformer stores original column names in transformers_
        cols = []
        for name, trans, col_list in prep.transformers_:
            if isinstance(col_list, list):
                cols.extend(col_list)
        # keep unique order
        seen = set()
        out = []
        for c in cols:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out
    except Exception:
        return []


def predict_proba_batch(pipe, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(X)[:, 1]


def predict_reg_batch(pipe, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict(X)


def unit_summary_from_models(
    df_unit: pd.DataFrame,
    feature_cols: List[str],
    clf_models: Dict[str, object],
    reg_models: Dict[str, object],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    X = df_unit[feature_cols].copy()

    rows = []

    # classification
    for name, pipe in clf_models.items():
        probs = predict_proba_batch(pipe, X)
        t = thresholds.get(name, 0.5)
        rows.append({
            "metric": name,
            "type": "risk_rate",
            "threshold": t,
            "avg_probability": float(np.mean(probs)),
            "predicted_rate": float((probs >= t).mean()),
        })

    # regression
    for name, pipe in reg_models.items():
        preds = predict_reg_batch(pipe, X)
        rows.append({
            "metric": name,
            "type": "regression",
            "avg_pred": float(np.mean(preds)),
            "p10": float(np.percentile(preds, 10)),
            "p50": float(np.percentile(preds, 50)),
            "p90": float(np.percentile(preds, 90)),
        })

    return pd.DataFrame(rows)


def unit_risk_matrix(
    df_unit: pd.DataFrame,
    unit_col: str,
    feature_cols: List[str],
    clf_models: Dict[str, object],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    out = []
    for unit_val, grp in df_unit.groupby(unit_col):
        X = grp[feature_cols]
        row = {unit_col: str(unit_val), "n": int(len(grp))}
        for name, pipe in clf_models.items():
            probs = predict_proba_batch(pipe, X)
            t = thresholds.get(name, 0.5)
            row[name] = float((probs >= t).mean())
        out.append(row)

    mat = pd.DataFrame(out).set_index(unit_col).sort_values("n", ascending=False)
    return mat


def simulate_sleep_improvement(df_unit: pd.DataFrame, delta_sleep_score: float) -> pd.DataFrame:
    sim = df_unit.copy()
    if "sleep_score" in sim.columns:
        sim["sleep_score"] = np.clip(sim["sleep_score"].astype(float) + float(delta_sleep_score), 0.0, 10.0)
    return sim


# def compare_unit_risk_before_after(
#     df_before: pd.DataFrame,
#     df_after: pd.DataFrame,
#     feature_cols: List[str],
#     clf_models: Dict[str, object],
#     thresholds: Dict[str, float],
# ) -> pd.DataFrame:
#     Xb = df_before[feature_cols]
#     Xa = df_after[feature_cols]

#     rows = []
#     for name, pipe in clf_models.items():
#         pb = predict_proba_batch(pipe, Xb)
#         pa = predict_proba_batch(pipe, Xa)
#         t = thresholds.get(name, 0.5)

#         rb = float((pb >= t).mean())
#         ra = float((pa >= t).mean())

#         rows.append({
#             "risk": name,
#             "baseline_rate": rb,
#             "after_rate": ra,
#             "delta": ra - rb,
#             "baseline_avg_prob": float(pb.mean()),
#             "after_avg_prob": float(pa.mean()),
#         })

#     return pd.DataFrame(rows).sort_values("delta")

# What is controls for Tabs 5 and 6

def what_if_controls_T5(base_row, context: str, container=None):
    container = container or st.sidebar
    overrides = {}

    container.markdown("### Prediction/Simulation")
    container.markdown("## What-If Controls")

    if "stress_score" in base_row.columns:
        overrides["stress_score"] = container.slider(
            "Stress score",
            0.0, 100.0,
            float(base_row["stress_score"].iloc[0]),
            1.0,
            key=f"{context}_stress_score_slider",
        )

    if "sleep_score" in base_row.columns:
        overrides["sleep_score"] = container.slider(
            "Sleep score",
            0.0, 10.0,
            float(base_row["sleep_score"].iloc[0]),
            0.1,
            key=f"{context}_sleep_score_slider",
        )

    if "fatigue" in base_row.columns:
        overrides["fatigue"] = container.slider(
            "Fatigue",
            0.0, 1.0,
            float(base_row["fatigue"].iloc[0]),
            0.01,
            key=f"{context}_fatigue_slider",
        )

    if "OPTEMPO" in base_row.columns:
        overrides["OPTEMPO"] = container.slider(
            "OPTEMPO",
            0.0, 1.0,
            float(base_row["OPTEMPO"].iloc[0]),
            0.01,
            key=f"{context}_optempo_slider",
        )

    # 0/1 flags -> use checkboxes (cleaner than sliders)
    if "morale_low" in base_row.columns:
        overrides["morale_low"] = 1 if container.checkbox(
            "Low Morale",
            value=bool(base_row["morale_low"].iloc[0]),
            key=f"{context}_morale_low_cb",
        ) else 0

    if "acft_failure" in base_row.columns:
        overrides["acft_failure"] = 1 if container.checkbox(
            "ACFT Failure",
            value=bool(base_row["acft_failure"].iloc[0]),
            key=f"{context}_acft_failure_cb",
        ) else 0

    if "behavioral_health_poor" in base_row.columns:
        overrides["behavioral_health_poor"] = 1 if container.checkbox(
            "Behavioral Health Challenges",
            value=bool(base_row["behavioral_health_poor"].iloc[0]),
            key=f"{context}_bh_poor_cb",
        ) else 0

    return overrides

def apply_overrides_unitwide_T6(
    df_unit: pd.DataFrame,
    base_row: pd.DataFrame,
    overrides: Dict[str, object],
) -> pd.DataFrame:
    """
    Apply what-if overrides unit-wide.

    For continuous fields, apply a SHIFT:
      shift = override_value - base_row_value

    For 0/1 flags (morale_low, acft_failure, behavioral_health_poor):
      If override is 1 -> set entire unit to 1
      If override is 0 -> set entire unit to 0
    """
    df_sim = df_unit.copy()

    # numeric shifts
    for col, clip_min, clip_max in [
        ("stress_score", 0.0, 100.0),
        ("sleep_score", 0.0, 10.0),
        ("fatigue", 0.0, 1.0),
        ("OPTEMPO", 0.0, 1.0),
    ]:
        if col in df_sim.columns and col in overrides and col in base_row.columns:
            base_val = float(base_row[col].iloc[0])
            new_val  = float(overrides[col])
            shift = new_val - base_val
            df_sim[col] = (df_sim[col].astype(float) + shift).clip(clip_min, clip_max)

    # binary flags (unit-wide set)
    for col in ["morale_low", "acft_failure", "behavioral_health_poor"]:
        if col in df_sim.columns and col in overrides:
            df_sim[col] = int(overrides[col])

    return df_sim



# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("ASR²D Controls")

uploaded = st.sidebar.file_uploader(
    "Upload dataset (xlsx or csv)",
    type=["xlsx", "csv"]
)

models = load_models(MODELS_DIR)

st.sidebar.markdown("---")
st.sidebar.subheader("Models detected")
if not models:
    st.sidebar.warning("No models found in /models. Put your *.pkl files in models/ and refresh.")
else:
    st.sidebar.success(f"Loaded {len(models)} model file(s).")
    st.sidebar.write(sorted(models.keys()))

st.sidebar.markdown("---")

# Load data
try:
    df = load_dataset(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

def add_derived_flags(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if "morale_low" not in d.columns and "morale" in d.columns:
        d["morale_low"] = d["morale"].astype(str).str.lower().isin(["low", "poor"]).astype(int)

    if "behavioral_health_poor" not in d.columns and "behavioral_health" in d.columns:
        d["behavioral_health_poor"] = d["behavioral_health"].astype(str).str.lower().isin(["poor", "at risk", "high risk"]).astype(int)

    return d

df = add_derived_flags(df)
# df_view = add_derived_flags(df_view)


st.sidebar.subheader("Soldier Selection Filters")

# Filters for unit-level analysis
df_view, active_filters = choose_base_filters(df)

# Pick a soldier row
st.sidebar.subheader("Select a Soldier")
idx_default = 0
max_rows = len(df_view)
if max_rows == 0:
    st.error("No rows match your filters. Reset filters.")
    st.stop()

pick_by = st.sidebar.radio("Select by:", ["DoD ID #", "Random"], horizontal=False)
if pick_by == "Random":
    base_idx = int(np.random.randint(0, max_rows))
else:
    base_idx = st.sidebar.number_input("Enter Row Index/Number", min_value=0, max_value=max_rows-1, value=idx_default, step=1)

# base row index relative to filtered df -> get global index
base_global_index = df_view.index[base_idx]
base_row = build_row_from_base(df, df.index.get_loc(base_global_index))

# ---------------------------------
# Global sidebar what-if controls (shared across tabs)
# ---------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("## Applies to Tab_4 & Tab_6")

with st.sidebar.container():
    st.sidebar.markdown("## 🔧 What-If Controls")
    shared_overrides = what_if_controls_T5(
        base_row,
        context="shared",
        container=st.sidebar
    )



st.title(APP_TITLE)
st.caption("Synthetic Data / Statistical Testing / Machine Learning Risk & Readiness Predictions")

tabs = st.tabs([
    "1) Overview_TAB 1",
    "2) Soldier Burnout Risk_TAB 2",
    "3) Soldier Readiness & Retention_TAB 3",
    "4) Soldier Performance Risks_TAB 4",
    "5) Hypothesis Test_TAB 5",
    "6) Division & Brigade Summary_TAB 6",
    "7) Future Features/Modules_TAB 7",
])

# -------------------------------
# Model dictionaries (NEW)
# -------------------------------

# Classification models found
CLF_TARGETS = ["high_burnout_risk", "high_risk_of_injury", "ucmj", "poor_performance", "suicide_risk", "non_deployable", "profile_status", "low_readiness_risk", "low_retention_risk"]

CLF_MODELS = {}
for t in CLF_TARGETS:
    k = f"{LOGREG_PREFIX}{t}"
    if k in models:
        CLF_MODELS[t] = models[k]

# Regression models found
REG_MODELS = {}
for k in ["xgbreg_soldier_readiness", "rfreg_soldier_readiness", "linreg_soldier_readiness"]:
    if k in models:
        REG_MODELS["soldier_readiness"] = models[k]
        break

for k in ["xgbreg_retention_rate", "rfreg_retention_rate", "linreg_retention_rate"]:
    if k in models:
        REG_MODELS["retention_rate"] = models[k]
        break

# Thresholds (tune later; these are safe defaults)
THRESHOLDS = {name: 0.5 for name in CLF_MODELS.keys()}

# Infer FEATURE_COLS from any one model 
FEATURE_COLS = []
if "high_burnout_risk" in CLF_MODELS:
    FEATURE_COLS = infer_feature_cols_from_model(CLF_MODELS["high_burnout_risk"])
elif REG_MODELS:
    any_reg = list(REG_MODELS.values())[0]
    FEATURE_COLS = infer_feature_cols_from_model(any_reg)

# Fallback if inference fails
if not FEATURE_COLS:
    st.warning("Could not infer FEATURE_COLS from models. Add FEATURE_COLS manually (raw columns used in training).")



# -------------------------------
# Shared What-If Controls for TABS 3 & 4
# -------------------------------
def what_if_controls(base_row, context: str):
    overrides = {}
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    if "stress_score" in base_row.columns:
        overrides["stress_score"] = c1.slider(
            "Stress score",
            0.0, 100.0,
            float(base_row["stress_score"].iloc[0]),
            1.0,
            key=f"{context}_stress_score_slider"
        )

    if "sleep_score" in base_row.columns:
        overrides["sleep_score"] = c2.slider(
            "Sleep score",
            0.0, 10.0,
            float(base_row["sleep_score"].iloc[0]),
            0.1,
            key=f"{context}_sleep_score_slider"
        )

    if "fatigue" in base_row.columns:
        overrides["fatigue"] = c3.slider(
            "Fatigue",
            0.0, 1.0,
            float(base_row["fatigue"].iloc[0]),
            0.01,
            key=f"{context}_fatigue_slider"
        )

    if "OPTEMPO" in base_row.columns:
        overrides["OPTEMPO"] = c4.slider(
            "OPTEMPO",
            0.0, 1.0,
            float(base_row["OPTEMPO"].iloc[0]),
            0.01,
            key=f"{context}_optempo_slider"
        )
    
    if "morale" in base_row.columns:
        overrides["morale"] = int(c5.checkbox(
            "Low Morale",
            value=bool(base_row["morale"].iloc[0]),
            key=f"{context}_morale_cb"
        ))

    if "acft_failure" in base_row.columns:
        overrides["acft_failure"] = int(c6.checkbox(
            "ACFT failure",
            value=bool(base_row["acft_failure"].iloc[0]),
            key=f"{context}_acft_failure_cb"
        ))

    if "behavioral_health" in base_row.columns:
        overrides["behavioral_health"] = int(c7.checkbox(
            "Behavioral Health Challenges",
            value=bool(base_row["behavioral_health"].iloc[0]),
            key=f"{context}_behavioral_health_cb"
     ))

    return overrides


def risk_gauge(prob: float, label: str):
    """Simple probability display."""
    st.metric(label, f"{prob*100:.1f}%")
    if prob >= 0.7:
        st.error("High risk (≥ 70%).")
    elif prob >= 0.4:
        st.warning("Moderate risk (40–69%).")
    else:
        st.success("Lower risk (< 40%).")




# -------------------------------
# Tab 1: Overview
# -------------------------------
with tabs[0]:
    st.subheader("Dataset Summary")
    st.write(f"Synthetic Dataset Rows Total: **{len(df):,}**")
    # st.write("Filters:", {k: v for k, v in active_filters.items() if v != "All"} or "None")

    # High-level metrics
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    # Try common columns
    if safe_col(df_view, "soldier_readiness"):
        c1.metric("% Soldier Readiness", f"{df_view['soldier_readiness'].mean()*100:.1f}%")
    else:
        c1.metric("% Soldier Readiness", "N/A")

    if safe_col(df_view, "retention_rate"):
        c2.metric("% Retention", f"{df_view['retention_rate'].mean()*100:.1f}%")
    else:
        c2.metric("% Retention", "N/A")

    for col, label, slot in [
        ("high_burnout_risk", "% Burnout Risk", c3),
        ("high_risk_of_injury", "% Injury Risk", c4),
    ]:
        r = observed_rate(df_view, col)
        slot.metric(label, f"{(100*r):.1f}%" if r is not None else "N/A")

    commander_caption("This overview displays notional readiness and risk levels for XVIII Airborne Corps.")

    st.markdown("---")
    st.subheader("Exploration Data Analysis / Data Distribution")
    eda_cols = st.columns(3)

    if safe_col(df_view, "stress_score"):
        with eda_cols[0]:
            plot_hist(df_view["stress_score"], "Stress Score Distribution")
            commander_caption("Higher stress is consistently associated with elevated burnout and performance risk.")
    if safe_col(df_view, "sleep_score"):
        with eda_cols[1]:
            plot_hist(df_view["sleep_score"], "Sleep Score Distribution")
            commander_caption("Lower sleep quality can amplify risks even when physical readiness is strong.")
    if safe_col(df_view, "OPTEMPO"):
        with eda_cols[2]:
            plot_hist(df_view["OPTEMPO"], "OPTEMPO Distribution")
            commander_caption("Higher OPTEMPO indicates sustained workload intensity and reduced recovery time.")


    st.markdown("---")
    st.subheader("Observed Risks")

    observed_targets = [c for c in ["high_burnout_risk", "high_risk_of_injury", "ucmj", "poor_performance", "non_deployable"] if c in df.columns]
    if observed_targets:
        target = st.selectbox("Choose an observed risk flag", observed_targets)

        groupby_opts = [c for c in ["brigade_level", "mos", "rank", "division_level"] if c in df.columns]
        grp = st.selectbox("Group by", groupby_opts) if groupby_opts else None

        if grp:
            tmp = df_view[[grp, target]].dropna()
            tmp[target] = tmp[target].astype(int)

            rates = (
                tmp.groupby(grp)[target]
                .mean()
                .sort_values(ascending=False)
                .head(20)
            )

            plot_df = rates.reset_index().rename(columns={target: "rate"})
            plot_df["percent"] = plot_df["rate"] * 100

            fig = px.bar(
                plot_df,
                x=grp,
                y="rate",
                hover_data={"rate": False, "percent": ":.1f"},
                labels={"rate": "Rate"},
                title=f"Observed {target} % By {grp}"
            )

        # show bars as percent on hover + better y-axis formatting
            fig.update_traces(
                hovertemplate=f"{grp}: %{{x}}<br>{target} rate: %{{customdata[0]:.1f}}%<extra></extra>",
                customdata=plot_df[["percent"]].to_numpy()
            )
            fig.update_yaxes(tickformat=".0%")

            st.plotly_chart(fig, width='stretch')

            commander_caption("Observed percentages help leaders identify where risk is concentrated, enabling informed decision-making before developing and executing policies and preparing for field exercises and deployments—ultimately enhancing Soldier and unit readiness, reducing risk, and supporting mission success.")
    else:
        st.info("No observed risk columns found (e.g., high_burnout_risk).")

    st.markdown("---")
    st.subheader("Selected Soldier 'current row' (static data)")
    st.dataframe(base_row, width='content', height=150)


# -------------------------------
# Tab 2: Soldier Burnout Risk Model
# -------------------------------
with tabs[1]:
    st.subheader("Soldier Burnout Risk Prediction Model")

    model_key = f"{LOGREG_PREFIX}high_burnout_risk"
    if model_key not in models:
        st.warning(f"Missing model: {model_key}. Train it and save to models/logreg_high_burnout_risk.pkl")
        st.stop()

    pipe = models[model_key]

    overrides = what_if_controls(base_row, context="burnout")
    X_one = apply_overrides(base_row.copy(), overrides)

    prob = predict_logistic(pipe, X_one)

    c1, c2 = st.columns([1, 1])

    with c1:
        risk_gauge(prob, "Predicted Burnout Risk (probability)")

        commander_caption(
            "Use this as an early-warning signal: if stress rises and sleep drops while OPTEMPO remains high, burnout risk tends to increase."
        )

    with c2:
        st.markdown("### Top 10 Prediction Drivers")
        top = explain_top10_contributions(pipe, X_one, model_kind="logistic")
        top10_bar(top, "Top 10 Drivers for Burnout Risk")
        commander_caption("These drivers show what is pushing this Soldier’s risk up or down right now (based on the current what-if inputs).")

    st.markdown("---")
    st.markdown("### Soldier snapshot (changes dynamically with 'what-if' inputs from sliders above)")
    st.dataframe(X_one, width='content', height=100)


# -------------------------------
# Tab 3: Soldier Readiness and Retention
# -------------------------------

with tabs[2]:
    st.subheader("Soldier Readiness Score Prediction")

    # Prefer XGB if present, else RF, else Linear
    readiness_pipe = None
    chosen = None
    for k in ["xgbreg_soldier_readiness", "rfreg_soldier_readiness", "linreg_soldier_readiness"]:
        if k in models:
            readiness_pipe = models[k]
            chosen = k
            break

    if readiness_pipe is None:
        st.warning("No readiness regression model found.")
        st.stop()

    st.info("")

    overrides = what_if_controls(base_row, context="readiness")
    X_one = apply_overrides(base_row.copy(), overrides)
    pred = predict_regression(readiness_pipe, X_one)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Predicted Soldier Readiness", f"{pred*100:.1f}%")  # display as %
        commander_caption("Reducing fatigue and stabilizing stress/sleep often improves Soldier readiness outcomes.")

    with c2:
        st.markdown("### ")

    # Explain only if linear model (coef available)
    if chosen == "linreg_soldier_readiness":
        st.markdown("### Explain prediction (Top 10 local drivers)")
        top = explain_top10_contributions(readiness_pipe, X_one, model_kind="linear")
        top10_bar(top, "Top 10 Local Drivers of Soldier Readiness (Linear)")
        commander_caption("These drivers show what is pushing readiness up or down for this Soldier (linear model).")
    else:
        st.info("")

    st.markdown("---")

    # -------------------------
    # Retention prediction panel
    # -------------------------
    retention_pipe = None
    chosen_r = None
    for k in ["xgbreg_retention_rate", "rfreg_retention_rate", "linreg_retention_rate"]:
        if k in models:
            retention_pipe = models[k]
            chosen_r = k
            break

    if retention_pipe is not None and "retention_rate" in df.columns:
        st.subheader("Soldier Retention Score Prediction")
        st.info("")

        pred_r = predict_regression(retention_pipe, X_one)
        pred_r = max(0.0, min(1.0, pred_r))  # clamp
        c3, c4 = st.columns([1, 1])

        with c3:
            st.metric("Predicted Soldier Retention", f"{pred_r*100:.1f}%")  # display as %
            commander_caption("Retention can be influenced by personal stressors and unit climate—use this to prioritize leader engagement.")

        with c4:
            st.markdown("### ")

    if chosen_r == "linreg_retention_rate":
        st.markdown("### Explain prediction (Top 10 local drivers)")
        top = explain_top10_contributions(retention_pipe, X_one, model_kind="linear")
        top10_bar(top, "Top 10 Local Drivers of Retention (Linear)")
        commander_caption("These drivers show what is pushing retention up or down for this Soldier (linear model).")
    else:
        st.info("")


# -------------------------------
# Tab 4: Soldier Injury & UCMJ Risk
# -------------------------------

with tabs[3]:

    st.subheader("Soldier Predicted Performance Risks")

    # Apply overrides to ONE soldier
    X_one = apply_overrides(
        base_row.copy(),
        shared_overrides
    )

    results = []

    risk_targets = []
    for t in ["high_risk_of_injury", "ucmj", "poor_performance", "non_deployable", "suicide_risk"]:
        key = f"{LOGREG_PREFIX}{t}"
        if key in models:
            risk_targets.append(t)
        
    if not risk_targets:
        st.warning("No injury/UCMJ logistic models found in /models (logreg_*.pkl).")
        st.stop()


    st.markdown("### Baseline Observed Rates in Filtered Soldier")

    cols = st.columns(min(5, len(risk_targets)))
    for i, t in enumerate(risk_targets[:5]):
        r = observed_rate(df_view, t)
        cols[i].metric(f"{t}", f"{(100*r):.1f}%" if r is not None else "N/A")

    commander_caption("Baseline rates show what is happening now in the unit slice—model predictions estimate risk for the selected Soldier under what-if inputs.")

    st.markdown("### Model-Predicted Probabilities for Selected Soldier")

    for t in risk_targets:
        pipe = models[f"{LOGREG_PREFIX}{t}"]
        prob = predict_logistic(pipe, X_one)

        c1, c2 = st.columns([1, 1])
        with c1:
            risk_gauge(prob, f"{t} — predicted probability")
        with c2:
            top = explain_top10_contributions(pipe, X_one, model_kind="logistic")
            top10_bar(top, f"Top 10 Drivers for {t}")


        st.markdown("---")



# -------------------------------
# Tab 5: Hypothesis Test
# -------------------------------
with tabs[4]:
    st.subheader("Hypothesis Test — OPTEMPO trend vs Stress (Welch t-test)")

    if not HAS_SCIPY:
        st.warning("SciPy not installed, so Welch t-test can’t run here. Install scipy to enable this tab.")
        st.stop()

    if "optempo_trend" not in df.columns or "stress_score" not in df.columns:
        st.info("Missing required columns: optempo_trend and/or stress_score.")
        st.stop()

    st.markdown("""
**Question:** The average stress level for Soldiers with a **high** OPTEMPO is less than or equal to the average stress level for Soldiers with a sustainable OPTEMPO.

- **H₀:** μ_improving = μ_declining  
- **H₁:** μ_declining > μ_improving (one-sided)  
- **α:** 0.05
""")

    alpha = st.slider("Significance level α", 0.01, 0.10, 0.05, 0.01)

    gA = df_view[df_view["optempo_trend"].astype(str).str.lower() == "improving"]["stress_score"].dropna()
    gB = df_view[df_view["optempo_trend"].astype(str).str.lower() == "declining"]["stress_score"].dropna()

    c1, c2, c3 = st.columns(3)
    c1.metric("n (improving)", f"{len(gA):,}")
    c2.metric("n (declining)", f"{len(gB):,}")
    c3.metric("Mean stress (declining - improving)", f"{(gB.mean() - gA.mean()):.2f}")

    # Welch t-test (two-sided), then convert to one-sided
    t_stat, p_two = ttest_ind(gB, gA, equal_var=False, nan_policy="omit")  # B vs A
    # one-sided p for H1: mean(B) > mean(A)
    p_one = p_two / 2 if t_stat > 0 else 1 - (p_two / 2)

    st.write(f"**Welch t-test:** t = `{t_stat:.3f}`, one-sided p = `{p_one:.6f}`")
    decision = "Reject H₀" if p_one < alpha else "Fail to reject H₀"
    st.success(f"Decision at α={alpha:.2f}: **{decision}**")

    # Plot distributions (boxplot-ish using hist overlays)
    # fig, ax = plt.subplots(figsize=(7, 3.5))
    # ax.hist(gA, bins=30, alpha=0.6, label="improving")
    # ax.hist(gB, bins=30, alpha=0.6, label="declining")
    # ax.set_title("Stress Score Distribution: improving vs declining OPTEMPO trend")
    # ax.set_xlabel("stress_score")
    # ax.set_ylabel("Count")
    # ax.legend()
    # st.pyplot(fig, clear_figure=True)

    commander_caption("If declining OPTEMPO trend is linked to higher stress, leaders can treat OPTEMPO stability and recovery time as a measurable risk lever.")






# -------------------------------
# Tab 6: Unit Summary (NEW)
# -------------------------------
with tabs[5]:
    st.subheader("Division & Brigade Summary — Predicted Risks")

    if not FEATURE_COLS:
        st.error("FEATURE_COLS missing. Cannot run unit aggregation.")
        st.stop()

    if not CLF_MODELS and not REG_MODELS:
        st.warning("No models loaded to generate unit summary.")
        st.stop()

    st.markdown("### Select unit level (Division → Brigade)")
    sel = unit_selectors(df, prefix="unit_summary")
    df_unit = filter_unit(df, sel)

    st.caption(f"Selected unit slice size: **{len(df_unit):,} Soldiers**")

    # Drop rows missing needed features
    df_unit_clean = df_unit.dropna(subset=[c for c in FEATURE_COLS if c in df_unit.columns]).copy()
    st.caption(f"Usable rows (after missing feature drop): **{len(df_unit_clean):,}**")


    st.markdown("---")
    st.markdown("### Division or Brigade Level Summary Table (select from dropdown below)")

    heat_levels = [c for c in ["company_level", "battalion_level", "brigade_level", "division_level"] if c in df_unit_clean.columns]
    if not heat_levels:
        st.info("No unit hierarchy columns found for heatmap (company/battalion/brigade/division).")
    else:
        unit_col = st.selectbox("Division/Brigade", heat_levels, key="heatmap_level")
        mat = unit_risk_matrix(df_unit_clean, unit_col, FEATURE_COLS, CLF_MODELS, THRESHOLDS)

        # show table
        st.dataframe(mat.style.format("{:.0%}", subset=[c for c in mat.columns if c != "n"]), height=350)

# -------------------------------
# 🧭Non-Deployable — Global Model Drivers (Coefficients)
# -------------------------------
    st.markdown("---")
    if f"{LOGREG_PREFIX}non_deployable" in models:

        with st.expander("🧭Non-Deployable — Top 10 Risk Drivers for XVIII Airborne Corps", expanded=False):

            nd_pipe = models[f"{LOGREG_PREFIX}non_deployable"]

            coef_df = top_logistic_coefficients(nd_pipe, top_n=10)

            if coef_df.empty:
                st.info("Could not extract coefficients or feature names for this model.")
            else:
                st.dataframe(
                   coef_df,
                   width="content",
                height=350,
            )

            commander_caption(
                "These are the strongest overall drivers in the model. "
                "Positive coefficients increase non-deployable risk; "
                "negative coefficients decrease non-deployable across the force."
            )
    
    st.markdown("---")
    st.markdown("## Prediction/Simulation — What if sleep improves or decreases across the unit?")
    commander_caption(
        "This simulation applies a unit-wide change to sleep scores, then recomputes model-predicted risk rates. "
        "It helps leaders estimate the potential impact of sleep-focused interventions."
    )

# --- Choose which models to include in the simulation ---
    SIM_RISK_TARGETS = [
        "high_burnout_risk",
        "high_risk_of_injury",
        "ucmj",
        "poor_performance",
        "non_deployable",
        "suicide_risk",
        "profile_status",
        "low_readiness_risk",
        "low_retention_risk",
    ]

    available_targets = [t for t in SIM_RISK_TARGETS if f"{LOGREG_PREFIX}{t}" in models]

    if not available_targets:
        st.warning("No logistic risk models found for simulation (logreg_*.pkl).")
        st.stop()  # ✅ critical: prevent stale state / stale outputs

# Slider is on the same scale as sleep_score (0–10), but allow +/- adjustments
    # sleep_boost = st.slider(
    #     "Unit-wide sleep change (points on 0–10 scale)",
    #     min_value=0.0,
    #     max_value=10.0,
    #     value=1.0,
    #     step=0.1,
    #     key="unit_sleep_boost_slider",
    # )

# ✅ Make sampling optional
    use_sampling = st.checkbox(
        "Limit simulation sample size (faster)",
        value=True,
        key="unit_sim_use_sampling"
    )

# -------------------------
# Apply shared sidebar what-if controls unit-wide
# -------------------------
    df_sim_base = df_unit_clean.copy()

    if len(df_sim_base) == 0:
        st.error("No usable rows after dropping missing feature columns. Check your dataset / FEATURE_COLS.")
        st.stop()

    # OPTIONAL sampling (keep your existing logic if you want)
    if use_sampling:
        # IMPORTANT: slider bounds must be valid (min < max)
        max_upper = min(10000, len(df_sim_base))
        if max_upper < 150:
            st.warning("Unit slice is smaller than 150; running full simulation without sampling.")
        else:
            max_n = st.slider(
                "Max Soldiers to simulate (for speed)",
                min_value=150,
                max_value=max_upper,
                value=min(1000, max_upper),
                step=50,
                key="unit_sim_max_n",
            )
            if len(df_sim_base) > max_n:
                df_sim_base = df_sim_base.sample(n=max_n, random_state=42)

    # Simulated unit data
    df_sim = apply_overrides_unitwide_T6(df_sim_base, base_row, shared_overrides)


    def unit_predicted_rate(pipe, X: pd.DataFrame, chunk_size: int = 5000) -> float:
        """Mean predicted probability computed in chunks (safe for big units)."""
        n = len(X)
        if n == 0:
            return float("nan")

        total = 0.0
        seen = 0
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            probs = pipe.predict_proba(X.iloc[start:end])[:, 1]
            total += float(np.sum(probs))
            seen += (end - start)
        return total / seen


    # -------------------------
    # Friendly "After" label (SAFE — no undefined variables)
    # -------------------------
    after_parts = []

    # If you still want to show the sleep_boost slider in Tab 6, keep it in this label
    # after_parts.append(f"Sleep {sleep_boost:+.1f}")

    # Also show what changed via shared overrides (relative to selected soldier baseline)
    def _delta(col: str) -> float:
        if col in shared_overrides and col in base_row.columns:
            return float(shared_overrides[col]) - float(base_row[col].iloc[0])
        return 0.0

    d_opt = _delta("OPTEMPO")
    d_fat = _delta("fatigue")
    d_str = _delta("stress_score")
    d_slp = _delta("sleep_score")

    # Only include non-zero deltas to keep it commander-friendly
    if abs(d_str) > 1e-9:
        after_parts.append(f"Stress {d_str:+.1f}")
    if abs(d_slp) > 1e-9:
        after_parts.append(f"SleepΔ {d_slp:+.1f}")  # optional since you already show Sleep {sleep_boost}
    if abs(d_opt) > 1e-9:
        after_parts.append(f"OPTEMPO {d_opt:+.2f}")
    if abs(d_fat) > 1e-9:
        after_parts.append(f"Fatigue {d_fat:+.2f}")

    # Binary flags show as ON/OFF if overridden
    for flag, label in [
        ("morale_low", "LowMorale"),
        ("acft_failure", "ACFTFail"),
        ("behavioral_health_poor", "BHpoor"),
    ]:
        if flag in shared_overrides:
            after_parts.append(f"{label}={'ON' if int(shared_overrides[flag])==1 else 'OFF'}")

    after_label = "After (" + ", ".join(after_parts) + ") (Predicted %)"


    # -------------------------
    # Build Before vs After table
    # -------------------------
    results = []
    for t in available_targets:
        pipe = models[f"{LOGREG_PREFIX}{t}"]

        base_rate = unit_predicted_rate(pipe, df_sim_base)
        sim_rate  = unit_predicted_rate(pipe, df_sim)

        results.append({
            "Risk Type": t,
            "Baseline (Predicted %)": base_rate * 100,
            after_label: sim_rate * 100,
            "Change (pp)": (sim_rate - base_rate) * 100,
        })

    sim_df = pd.DataFrame(results).sort_values("Change (pp)")

    # 1 decimal everywhere
    for c in ["Baseline (Predicted %)", after_label, "Change (pp)"]:
        sim_df[c] = sim_df[c].round(1)

    st.markdown("### Before vs After — Unit Model Prediction Impact")
    st.caption(
        f"Unit-wide overrides applied from sidebar. "
        f"n used: **{len(df_sim_base):,}** (sampling={'ON' if use_sampling else 'OFF'})"
    )

    st.dataframe(sim_df, width="content", height=320)

    commander_caption(
        "Baseline and After values are model-predicted unit risk rates (mean probability). "
        "Change (pp) is the estimated shift if conditions change across the unit."
    )






# # Always simulate from df_unit_clean so models receive complete feature columns
#     df_sim_base = df_unit_clean.copy()

#     if len(df_sim_base) == 0:
#         st.error("No usable rows after dropping missing feature columns. Check your dataset / FEATURE_COLS.")
#         st.stop()

#     if len(df_sim_base) > 5000 and not use_sampling:
#         st.warning("Large unit slice detected. Full simulation may be slow. Consider enabling sample size limiting.")

#     if use_sampling:
#         max_n = st.slider(
#             "Max Soldiers to simulate (for speed)",
#             min_value=150,
#             max_value=min(10000, len(df_sim_base)),
#             value=min(1000, len(df_sim_base)),
#             step=500,
#             key="unit_sim_max_n",
#         )
#         if len(df_sim_base) > max_n:
#             df_sim_base = df_sim_base.sample(n=max_n, random_state=42)

# # Base slice for simulation must already be clean:
#     df_sim_base = df_unit_clean.copy()

# # Apply shared sidebar what-if controls unit-wide
#     df_sim = apply_overrides_unitwide_T6(df_sim_base, base_row, shared_overrides)










# # -------------------------
# # NEW simulation controls (additions)
# # -------------------------

# optempo_shift = st.slider(
#     "Unit-wide OPTEMPO change",
#     min_value=-0.00,
#     max_value=1.00,
#     # value=0.10,
#     # step=0.01,
#     key="unit_optempo_shift_slider",
# )
# commander_caption("Reducing sustained OPTEMPO improves recovery time and can lower burnout and injury risk.")

# fatigue_shift = st.slider(
#     "Unit-wide fatigue change",
#     min_value=-0.00,
#     max_value=1.00,
#     # value=0.0,
#     # step=0.05,
#     key="unit_fatigue_shift_slider",
# )
# commander_caption("Lower fatigue improves readiness and reduces injury risk even when OPTEMPO remains high.")

# # Morale simulation (Yes/No checkbox)
# apply_morale = st.checkbox(
#     "Apply morale improvement simulation? (Yes/No)",
#     value=False,
#     key="unit_apply_morale_sim",
# )
# morale_improve_pct = 0
# if apply_morale:
#     morale_improve_pct = st.slider(
#         "Reduce low morale (%)",
#         0, 50, 0, step=5,
#         key="unit_morale_improve_pct",
#     )
#     commander_caption("Leadership engagement that improves morale can materially reduce performance and burnout risk.")

# # Behavioral health simulation (Yes/No checkbox)
# apply_bh = st.checkbox(
#     "Apply behavioral health improvement simulation? (Yes/No)",
#     value=False,
#     key="unit_apply_bh_sim",
# )
# bh_improve_pct = 0
# if apply_bh:
#     bh_improve_pct = st.slider(
#         "Reduce poor behavioral health (%)",
#         0, 50, 0, step=5,
#         key="unit_bh_improve_pct",
#     )
#     commander_caption("Earlier access to behavioral health care can reduce non-deployable and suicide risk.")

# # Apply OPTEMPO / fatigue shifts if columns exist
# if "OPTEMPO" in df_sim.columns:
#     df_sim["OPTEMPO"] = (df_sim["OPTEMPO"].astype(float) + optempo_shift).clip(0.0, 1.0)

# if "fatigue" in df_sim.columns:
#     df_sim["fatigue"] = (df_sim["fatigue"].astype(float) + fatigue_shift).clip(0.0, 1.0)

# # Apply morale improvement by flipping some 1 -> 0
# if apply_morale and "low_morale" in df_sim.columns and morale_improve_pct > 0:
#     morale_ones = df_sim.index[df_sim["low_morale"].astype(int) == 1]
#     if len(morale_ones) > 0:
#         flip_n = int(round(len(morale_ones) * (morale_improve_pct / 100.0)))
#         if flip_n > 0:
#             idx = pd.Series(morale_ones).sample(
#                 n=min(flip_n, len(morale_ones)),
#                 random_state=42
#             ).values
#             df_sim.loc[idx, "low_morale"] = 0

# # Apply BH improvement by flipping some 1 -> 0
# if apply_bh and "poor_behavioral_health" in df_sim.columns and bh_improve_pct > 0:
#     bh_ones = df_sim.index[df_sim["poor_behavioral_health"].astype(int) == 1]
#     if len(bh_ones) > 0:
#         flip_n = int(round(len(bh_ones) * (bh_improve_pct / 100.0)))
#         if flip_n > 0:
#             idx = pd.Series(bh_ones).sample(
#                 n=min(flip_n, len(bh_ones)),
#                 random_state=42
#             ).values
#             df_sim.loc[idx, "poor_behavioral_health"] = 0


# -------------------------
# Chunked prediction helper (use this instead of the old unit_predicted_rate)
# -------------------------
# def unit_predicted_rate(pipe, X: pd.DataFrame, chunk_size: int = 5000) -> float:
#     """
#     Mean predicted probability computed in chunks (safe for big units).
#     """
#     n = len(X)
#     if n == 0:
#         return float("nan")

#     total = 0.0
#     seen = 0
#     for start in range(0, n, chunk_size):
#         end = min(start + chunk_size, n)
#         probs = pipe.predict_proba(X.iloc[start:end])[:, 1]
#         total += float(np.sum(probs))
#         seen += (end - start)

#     return total / seen


# # -------------------------
# # Friendly "After" label so commanders can see what changed
# # -------------------------
# after_parts = [f"Sleep {sleep_boost:+.1f}"]
# if abs(OPTEMPO) > 1e-9:
#     after_parts.append(f"OPTEMPO {optempo_shift:+.2f}")
# if abs(fatigue_shift) > 1e-9:
#     after_parts.append(f"Fatigue {fatigue_shift:+.2f}")
# if apply_morale and morale_improve_pct > 0:
#     after_parts.append(f"Morale -{morale_improve_pct}%")
# if apply_bh and bh_improve_pct > 0:
#     after_parts.append(f"BH -{bh_improve_pct}%")

# after_label = "After (" + ", ".join(after_parts) + ") (Predicted %)"


# # -------------------------
# # Build Before vs After table (wired to ALL sliders/checkboxes)
# # -------------------------
# results = []

# for t in available_targets:
#     pipe = models[f"{LOGREG_PREFIX}{t}"]

#     base_rate = unit_predicted_rate(pipe, df_sim_base)
#     sim_rate  = unit_predicted_rate(pipe, df_sim)

#     results.append({
#         "Risk Type": t,
#         "Baseline (Predicted %)": base_rate * 100,
#         after_label: sim_rate * 100,
#         "Change (pp)": (sim_rate - base_rate) * 100,
#     })

# sim_df = pd.DataFrame(results).sort_values("Change (pp)")

# # one decimal everywhere
# for c in ["Baseline (Predicted %)", after_label, "Change (pp)"]:
#     sim_df[c] = sim_df[c].round(1)

# st.markdown("### Before vs After — Unit Model Prediction Impact")
# st.caption(
#     f"Unit-wide changes applied. "
#     f"n used: **{len(df_sim_base):,}** (sampling={'ON' if use_sampling else 'OFF'})"
# )

# st.dataframe(
#     sim_df,
#     width="content",
#     height=320,
#     key=f"sleep_sim_table_{sleep_boost:.1f}_{optempo_shift:.2f}_{fatigue_shift:.2f}_{apply_morale}_{morale_improve_pct}_{apply_bh}_{bh_improve_pct}_{len(df_sim_base)}_{use_sampling}"
# )

# commander_caption(
#     "Baseline and After values are model-predicted unit risk rates (mean probability). "
#     "Change (pp) shows the estimated shift if conditions change across the unit."
# )









# def unit_predicted_rate_chunked(pipe, X: pd.DataFrame, chunk_size: int = 5000) -> float: 
#     """ Compute mean predicted probability in chunks (prevents slowdowns / memory spikes on big units). """ 
#     n = len(X) 
#     if n == 0: 
#         return float("nan") 
#     total = 0.0 
#     seen = 0 
#     for start in range(0, n, chunk_size): 
#         end = min(start + chunk_size, n) 
#         probs = pipe.predict_proba(X.iloc[start:end])[:, 1] 
#         total += float(np.sum(probs)) 
#         seen += (end - start) 
        
#     return total / seen





# -------------------------------
# Tab 7: Future Modules
# -------------------------------
with tabs[6]:
    st.subheader("Future Modules (placeholders)")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 📝 Counseling Analysis")
        st.write("Analyze counseling statements to detect morale/burnout signals")
        commander_caption("Provides early warning indicators from written counseling notes, helping leaders act before performance declines.")

    with c2:
        st.markdown("### 🧠 Graph Networks")
        st.write("Graph Networks show how people, events, or factors are connected and how a change in one affects the others.")
        st.write("Models how training events, leadership climate, and readiness are connected over time.")
        st.write("If training tempo increases, a graph network can show how that impacts morale, fatigue, and readiness over time.")
        commander_caption("Helps leaders see how changes in one training or stress factor affect readiness across the unit.")

    with c3:
        st.markdown("### 📱 Mobile Optimization")
        st.write("Streamlit UI optimized for leader phone use (quick filters + one-click views).")
        commander_caption("Leaders can check risk and readiness at the point of decision—without a laptop.")







# FUTURE STUFF FOR TAB 6 and TAB 7


#     st.markdown("---")
#     st.markdown("## Trend Simulation — What if sleep improves or decreases across the unit?")
#     commander_caption(
#     "This simulation applies a unit-wide improvement to sleep scores, then recomputes model-predicted risk rates. "
#     "It helps leaders estimate potential impact of sleep-focused interventions."
#     )

# # --- Choose which models to include in the simulation ---
#     SIM_RISK_TARGETS = [
#         "high_burnout_risk",
#         "high_risk_of_injury",
#         "ucmj",
#         "poor_performance",
#         "non_deployable",
#         "suicide_risk",
#         "profile_status",
#         "low_readiness_risk",
#         "low_retention_risk",
#     ]

#     available_targets = [t for t in SIM_RISK_TARGETS if f"{LOGREG_PREFIX}{t}" in models]

#     if not available_targets:
#         st.warning("No logistic risk models found for simulation (logreg_*.pkl).")
#     else:
#     # Slider is in same scale as sleep_score (your dataset uses 0–10)
#         sleep_boost = st.slider(
#             "Unit-wide sleep improvement (points on 0–10 scale)",
#             min_value=0.0,
#             max_value=10.0,
#             value=1.0,
#             step=0.1,
#             key="unit_sleep_boost_slider"
#         ) 

#     # Optional performance control: sample large units
#     max_n = st.slider(
#         "Max Soldiers to simulate (for speed)",
#         min_value=500,
#         max_value=10000,
#         value=3000,
#         step=500,
#         key="unit_sim_max_n"
#     )

#     df_sim_base = df_unit.copy()
#     if len(df_sim_base) > max_n:
#         df_sim_base = df_sim_base.sample(max_n, random_state=42)

    # Build simulated dataframe (sleep increased)
    df_sim = df_sim_base.copy()
    # if "sleep_score" in df_sim.columns:
    #     df_sim["sleep_score"] = (df_sim["sleep_score"].astype(float) + sleep_boost).clip(0.0, 10.0)

    # Helper to compute predicted unit risk rate from a logistic model
    def unit_predicted_rate(pipe, X: pd.DataFrame) -> float:
        probs = pipe.predict_proba(X)[:, 1]
        return float(np.mean(probs))

    results = []
    for t in available_targets:
        pipe = models[f"{LOGREG_PREFIX}{t}"]

        # Baseline predicted risk rate (mean probability)
        base_rate = unit_predicted_rate(pipe, df_sim_base)

        # Simulated predicted risk rate after sleep boost
        sim_rate = unit_predicted_rate(pipe, df_sim)

        delta_pp = (sim_rate - base_rate) * 100  # percentage points
        # results.append({
        #     "Risk Type": t,
        #     "Baseline (Predicted %)": base_rate * 100,
        #     f"After +{sleep_boost:.1f} sleep (Predicted %)": sim_rate * 100,
        #     "Change (percentage points)": delta_pp
        # })

    # sim_df = pd.DataFrame(results).sort_values("Change (percentage points)")

#     sim_df["Baseline (Predicted %)"] = sim_df["Baseline (Predicted %)"].round(1)
#     sim_df[f"After +{sleep_boost:.1f} sleep (Predicted %)"] = sim_df[
#        f"After +{sleep_boost:.1f} sleep (Predicted %)"
#     ].round(1)
#     sim_df["Change (percentage points)"] = sim_df["Change (percentage points)"].round(1)


#     # ✅ Simple commander-friendly table
#     st.markdown("Before and After Unit Model Prediction Impact Table")
#     st.dataframe(sim_df, width="content", height=300)

#     commander_caption(
#         "Baseline and After values are model-predicted unit risk rates (mean probability). "
#         "Change (pp) shows the shift in predicted risk if average sleep improves across the unit."
#     )



    # # ✅ Optional: simple bar chart of change (pp)
    # fig, ax = plt.subplots(figsize=(8, 3.6))
    # ax.barh(sim_df["Risk Type"].astype(str), sim_df["Change (pp)"].values)
    # ax.set_title("Estimated Unit Impact from Sleep Improvement")
    # ax.set_xlabel("Change in predicted risk rate (percentage points)")
    # st.pyplot(fig, clear_figure=True)

    # st.markdown("### Trend Simulation — “What if sleep improves across the unit?”")
    # delta = st.slider("Sleep_score increase (0–10 scale)", 0.0, 0.10, key="unit_sleep_delta")
    # df_after = simulate_sleep_improvement(df_unit_clean, delta_sleep_score=delta)

    # comp = compare_unit_risk_before_after(df_unit_clean, df_after, FEATURE_COLS, CLF_MODELS, THRESHOLDS)

    # st.dataframe(
    #     comp.style.format({
    #         "baseline_rate": "{:.0%}",
    #         "after_rate": "{:.0%}",
    #         "delta": "{:+.1%}",
    #         "baseline_avg_prob": "{:.3f}",
    #         "after_avg_prob": "{:.3f}",
    #     }),
    #     height=300
    # )

    # commander_caption("This simulation estimates how unit-wide sleep improvement could shift predicted risk rates (useful for policy and training calendar decisions).")

# -------------------------------
# Tab 7: Company Commander (NEW)
# -------------------------------
# with tabs[6]:
#     st.subheader("Company Commander Dashboard")

#     if "company_level" not in df.columns:
#         st.warning("company_level not found in dataset. Add company_level to enable this view.")
#         st.stop()

#     if not FEATURE_COLS:
#         st.error("FEATURE_COLS missing. Cannot run company dashboard.")
#         st.stop()

#     company_opts = sorted(df["company_level"].dropna().astype(str).unique().tolist())
#     company = st.selectbox("Select Company", company_opts, key="co_cmd_company")

#     df_co = df[df["company_level"].astype(str) == str(company)].copy()
#     st.caption(f"Company size: **{len(df_co):,} Soldiers**")

#     df_co = df_co.dropna(subset=[c for c in FEATURE_COLS if c in df_co.columns]).copy()
#     st.caption(f"Usable rows (after missing feature drop): **{len(df_co):,}**")

#     if len(df_co) == 0:
#         st.error("No rows left after dropping missing features for this company.")
#         st.stop()

#     # Unit summary
#     summary = unit_summary_from_models(df_co, FEATURE_COLS, CLF_MODELS, REG_MODELS, THRESHOLDS)

#     # Quick metrics
#     c1, c2, c3, c4 = st.columns(4)
#     if "soldier_readiness" in REG_MODELS and "avg_pred" in summary.columns:
#         sr = summary[summary["metric"] == "soldier_readiness"]["avg_pred"].values
#         if len(sr): c1.metric("Avg Predicted Readiness", f"{sr[0]:.3f}")
#     if "retention_rate" in REG_MODELS and "avg_pred" in summary.columns:
#         rr = summary[summary["metric"] == "retention_rate"]["avg_pred"].values
#         if len(rr): c2.metric("Avg Predicted Retention", f"{rr[0]:.3f}")

#     # risk rate KPIs (if available)
#     for i, tgt in enumerate(["high_burnout_risk", "high_risk_of_injury", "non_deployable", "suicide_risk"]):
#         if tgt in CLF_MODELS:
#             row = summary[summary["metric"] == tgt]
#             if not row.empty:
#                 rate = float(row["predicted_rate"].iloc[0])
#                 [c3, c4, c3, c4][i % 2].metric(f"% {tgt}", f"{rate*100:.1f}%")

#     st.markdown("---")
#     st.markdown("### Company predicted risk rates (bar view)")
#     risk_rows = summary[summary["type"] == "risk_rate"].copy()
#     if not risk_rows.empty:
#         risk_rows = risk_rows.sort_values("predicted_rate", ascending=False)
#         fig, ax = plt.subplots(figsize=(9, 3.8))
#         ax.bar(risk_rows["metric"].astype(str), risk_rows["predicted_rate"].astype(float))
#         ax.set_title("Predicted Risk Rates (Company)")
#         ax.set_ylabel("Rate")
#         ax.tick_params(axis="x", rotation=45)
#         st.pyplot(fig, clear_figure=True)
#         commander_caption("These bars show the share of Soldiers predicted high risk by category (helps prioritize leader actions).")
#     else:
#         st.info("No classification models loaded to display risk-rate bars.")

#     st.markdown("---")
#     st.markdown("### Roster view (top-risk Soldiers) — no PII")
#     # Pick a risk model for sorting
#     sort_risk = st.selectbox("Sort roster by risk", list(CLF_MODELS.keys()), key="co_cmd_sort_risk") if CLF_MODELS else None

#     if sort_risk:
#         X = df_co[FEATURE_COLS]
#         probs = predict_proba_batch(CLF_MODELS[sort_risk], X)
#         roster = df_co.copy()
#         roster[f"pred_{sort_risk}_prob"] = probs

#         show_cols = [c for c in ["division_level","brigade_level","battalion_level","company_level","mos","rank","age","stress_score","sleep_score","fatigue","OPTEMPO"] if c in roster.columns]
#         show_cols = show_cols + [f"pred_{sort_risk}_prob"]
#         roster = roster.sort_values(f"pred_{sort_risk}_prob", ascending=False).head(25)

#         st.dataframe(roster[show_cols], height=400)
#         commander_caption("Roster view highlights the highest predicted risks for targeted leader engagement (mentorship, rest cycles, resources).")

