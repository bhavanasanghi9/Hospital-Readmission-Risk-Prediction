# import joblib
# from pathlib import Path
# from src.pipeline import run_pipeline
# from src.preprocessing import transform_test_to_match_train

# ARTIFACTS_DIR = Path("artifacts")

# MODEL_PATH = ARTIFACTS_DIR / "xgb_reduced.joblib"
# FEATURES_PATH = ARTIFACTS_DIR / "xgb_reduced_features.json"


# def load_model():
#     return joblib.load(MODEL_PATH)


# def load_feature_list():
#     import json
#     with open(FEATURES_PATH, "r") as f:
#         return json.load(f)
    
# def load_artifacts():
#     model = load_model()
#     feature_list = load_feature_list()
#     return model, feature_list

# def prepare_input(df, feature_list):
    
#     # Rename underscore columns back to hyphens
#     # (Pydantic doesn't allow hyphens in field names)
#     df = df.rename(columns={
#         "glyburide_metformin": "glyburide-metformin",
#         "glipizide_metformin": "glipizide-metformin",
#         "glimepiride_pioglitazone": "glimepiride-pioglitazone",
#         "metformin_rosiglitazone": "metformin-rosiglitazone",
#         "metformin_pioglitazone": "metformin-pioglitazone",
#     })

#     X, _ = run_pipeline(df, include_target=False)
#     X_processed = transform_test_to_match_train(
#         X,
#         reference_columns=feature_list,
#     )
#     return X_processed
#     X, _ = run_pipeline(df, include_target=False)
#     X_processed = transform_test_to_match_train(
#         X,
#         reference_columns=feature_list,
#     )
#     return X_processed

# def predict_risk(df):
#     model, feature_list = load_artifacts()
#     thresholds = load_thresholds()

#     X_prepared = prepare_input(df, feature_list)
    

#     if X_prepared.empty:
#         raise ValueError("No valid records available for scoring after preprocessing.")
#     probs = model.predict_proba(X_prepared)[:, 1]

#     results = []
#     for score in probs:
#         results.append({
#             "risk_score": float(score),
#             "risk_tier": assign_risk_tier(score, thresholds)
#         })

#     return results

# def load_thresholds():
#     import json

#     thresholds_path = ARTIFACTS_DIR / "risk_thresholds.json"
#     with open(thresholds_path, "r") as f:
#         return json.load(f)

# def assign_risk_tier(score, thresholds):
#     if score >= thresholds["high"]:
#         return "High"
#     elif score >= thresholds["medium"]:
#         return "Medium"
#     return "Low"










import json
from pathlib import Path

import joblib
import pandas as pd

from src.pipeline import run_pipeline
from src.preprocessing import transform_test_to_match_train

# Always resolve relative to this file's location
# src/inference.py → up to src/ → up to project root → artifacts/
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "xgb_reduced.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "xgb_reduced_features.json"
THRESHOLDS_PATH = ARTIFACTS_DIR / "risk_thresholds.json"


def load_model():
    return joblib.load(MODEL_PATH)


def load_feature_list() -> list:
    with open(FEATURES_PATH, "r") as f:
        return json.load(f)


def load_artifacts():
    model = load_model()
    feature_list = load_feature_list()
    return model, feature_list


def load_thresholds() -> dict:
    with open(THRESHOLDS_PATH, "r") as f:
        return json.load(f)


def prepare_input(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    # Rename underscored columns back to hyphens
    # (Pydantic does not allow hyphens in field names)
    df = df.rename(columns={
        "glyburide_metformin": "glyburide-metformin",
        "glipizide_metformin": "glipizide-metformin",
        "glimepiride_pioglitazone": "glimepiride-pioglitazone",
        "metformin_rosiglitazone": "metformin-rosiglitazone",
        "metformin_pioglitazone": "metformin-pioglitazone",
    })

    X, _ = run_pipeline(df, include_target=False)
    X_processed = transform_test_to_match_train(
        X,
        reference_columns=feature_list,
    )
    return X_processed


def assign_risk_tier(score: float, thresholds: dict) -> str:
    if score >= thresholds["high"]:
        return "High"
    elif score >= thresholds["medium"]:
        return "Medium"
    return "Low"