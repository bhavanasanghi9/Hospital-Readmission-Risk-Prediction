from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import PatientRecord, PredictionResponse
from src.inference import (
    assign_risk_tier,
    load_artifacts,
    load_thresholds,
    prepare_input,
)

artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, feature_list = load_artifacts()
        thresholds = load_thresholds()
        artifacts["model"] = model
        artifacts["feature_list"] = feature_list
        artifacts["thresholds"] = thresholds
        print("✅ Model artifacts loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading artifacts: {type(e).__name__}: {e}")

    yield  # ← moved outside try/except

    artifacts.clear()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     try:
#         model, feature_list = load_artifacts()
#         thresholds = load_thresholds()

#         artifacts["model"] = model
#         artifacts["feature_list"] = feature_list
#         artifacts["thresholds"] = thresholds

#         print("Model artifacts loaded successfully")
#         yield
#     finally:
#         artifacts.clear()


app = FastAPI(
    title="Hospital Readmission Risk API",
    description="Predicts 30-day readmission risk for diabetic patients at point of discharge.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictionResponse)
def predict(record: PatientRecord):
    try:
        df = pd.DataFrame([record.model_dump()])
        X_prepared = prepare_input(df, artifacts["feature_list"])

        if X_prepared.empty:
            raise HTTPException(
                status_code=400,
                detail="No valid records available for scoring after preprocessing.",
            )

        score = float(artifacts["model"].predict_proba(X_prepared)[:, 1][0])
        risk_tier = assign_risk_tier(score, artifacts["thresholds"])

        return PredictionResponse(
            encounter_id=record.encounter_id,
            risk_score=score,
            risk_tier=risk_tier,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )


@app.get("/health")
def health_check():
    if not artifacts:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded",
        )

    return {
        "status": "ok",
        "model_loaded": True,
        "version": "1.0.0",
    }