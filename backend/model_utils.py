import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "mental_health_model.pkl"

model = joblib.load(MODEL_PATH)

# --- keep your feature order exactly as during training
FEATURES = [
    "Age", "no_employees_mid", "leave", "work_interfere",
    "self_employed", "family_history", "remote_work", "tech_company",
    "benefits", "care_options", "wellness_program", "seek_help",
    "anonymity", "mental_health_interview", "phys_health_interview",
    "mental_vs_physical", "mental_health_consequence",
    "phys_health_consequence", "obs_consequence",
    "coworkers", "supervisor",
    "Gender_female", "Gender_male", "Gender_other"
]

def preprocess(payload: dict) -> pd.DataFrame:
    """
    Convert JSON payload from the frontend into the exact
    single-row dataframe the model expects.
    """
    row = {f: 0 for f in FEATURES}

    # ------------ numeric fields ------------
    row["Age"]               = payload["age"]
    row["no_employees_mid"]  = payload["no_employees_mid"]
    row["leave"]             = payload["leave"]             # 0/1
    row["work_interfere"]    = payload["work_interfere"]    # 0–3

    # ------------ binary 0/1 flags -----------
    binaries = [
        "self_employed", "family_history", "remote_work",
        "tech_company", "benefits", "care_options",
        "wellness_program", "seek_help", "anonymity",
        "mental_health_interview", "phys_health_interview",
        "mental_vs_physical", "mental_health_consequence",
        "phys_health_consequence", "obs_consequence",
        "coworkers", "supervisor"
    ]
    for b in binaries:
        row[b] = payload[b]

    # ------------ one-hot gender -------------
    gender = payload["gender"]          # "male" | "female" | "other"
    row[f"Gender_{gender}"] = 1

    return pd.DataFrame([row])
