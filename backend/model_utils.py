import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "mental_health_model.pkl"
model = joblib.load(MODEL_PATH)

def preprocess(payload: dict) -> pd.DataFrame:
    row = {}

    # Numerical + binary fields
    row["Age"]               = payload["age"]
    row["no_employees_mid"]  = payload["no_employees_mid"]
    row["leave"]             = payload["leave"]
    row["work_interfere"]    = payload["work_interfere"]

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

    # One-hot encode gender
    gender = payload["gender"]  # should be "male", "female", or "other"
    row[f"Gender_{gender}"] = 1

    # Fill missing features with 0 and ensure correct order
    df = pd.DataFrame([row])
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[model.feature_names_in_]

    return df
