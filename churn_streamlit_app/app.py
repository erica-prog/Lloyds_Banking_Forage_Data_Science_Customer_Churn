# app.py (or app/utils.py)

# (Optional) Install dependencies: streamlit, pandas, scikit-learn, joblib
# Run in terminal: pip install -U streamlit pandas scikit-learn joblib

import json
from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

# ---------- CONFIG ----------
ROOT        = Path(__file__).resolve().parents[1]  # adjust if app.py lives elsewhere
DATA_PATH   = ROOT / "cleaned_data" / "churn_predictive_data_cleaned.csv"  # change if needed
MODEL_DIR   = ROOT / "models"
FEATS_JSON  = MODEL_DIR / "feature_names.json"     # written during training
DEFAULT_MDL = "rf"                                 # or whatever you set as best

# Columns you dropped in the notebook
DROP_COLS = ["CustomerID", "last_interaction", "resolved_interactions", "LastLoginDate"]

# Simple ordinal mappings you used
ENCODINGS = {
    "Gender":        {"M": 0, "F": 1},
    "MaritalStatus": {"Single": 0, "Married": 1, "Widowed": 2, "Divorced": 3, "Unknown": 4},
    "ServiceUsage":  {"Mobile App": 0, "Website": 1, "Online Banking": 2},
    "IncomeLevel":   {"Low": 0, "Medium": 1, "High": 2},
}

# ---------- DATA LOADING ----------
@st.cache_data(show_spinner=False)
def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """
    Load your cleaned churn dataset (CSV).
    """
    df = pd.read_csv(path)
    return df


# ---------- PREPROCESSING / FEATURE PREP ----------
@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same lightweight preprocessing you used pre-model:
    - drop columns you removed in the notebook
    - apply simple label maps where you used .map(...)
    (Heavy lifting like scaling/SMOTE happens inside the saved sklearn Pipeline.)
    """
    df_p = df.copy()
    df_p.drop(columns=[c for c in DROP_COLS if c in df_p.columns], inplace=True, errors="ignore")

    for col, mapping in ENCODINGS.items():
        if col in df_p.columns:
            df_p[col] = df_p[col].map(mapping)

    # If you created any engineered columns in notebook BEFORE modeling, do them here as well.
    # e.g.
    # df_p["resolution_rate"].fillna(0, inplace=True)
    # (add more steps as needed)

    return df_p


@st.cache_data(show_spinner=False)
def prepare_features(df_processed: pd.DataFrame):
    """
    Align columns to what the model expects.

    Returns:
      X_aligned: dataframe ordered & subset to match training columns
      feature_list: list of feature names in order
    """
    # Load the training feature order (saved during training)
    with open(FEATS_JSON) as f:
        feature_list = json.load(f)

    # Add any missing columns as 0, drop extras
    X = df_processed.copy()
    for col in feature_list:
        if col not in X.columns:
            X[col] = 0
    X_aligned = X[feature_list]

    return X_aligned, feature_list


# ---------- MODEL LOADING / PREDICT ----------
@st.cache_resource(show_spinner=False)
def load_model(name: str = DEFAULT_MDL):
    """
    Load a serialized sklearn Pipeline (.pkl).
    """
    path = MODEL_DIR / f"{name}_pipe.pkl"
    return joblib.load(path)


def predict(df_raw: pd.DataFrame, model_name: str = DEFAULT_MDL):
    """
    Full flow: preprocess -> align -> predict probabilities.
    """
    df_proc = preprocess_data(df_raw)
    X, _ = prepare_features(df_proc)
    pipe = load_model(model_name)
    proba = pipe.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)  # or expose a threshold slider in UI
    return preds, proba
