from __future__ import annotations

import pandas as pd

from src.data_cleaning import (
    replace_question_with_nan,
    exclude_non_readmission_eligible_encounters,
    drop_early_columns,
    create_binary_target,
    clean_invalid_gender_rows,
)

from src.feature_engineering import (
    encode_age_ordinal,
    handle_lab_results,
    add_diagnosis_category_features,
    add_diabetes_primary_flag,
    engineer_medication_features,
    engineer_utilization_features,
    engineer_encounter_intensity_features,
    engineer_admission_type,
    engineer_admission_source,
    engineer_discharge_disposition,
    engineer_discharge_features,
    engineer_medical_specialty,
    engineer_race,
    MEDICATION_FEATURES,
)

from src.preprocessing import prepare_features


def run_data_cleaning(
    df: pd.DataFrame,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Apply all raw-data cleaning steps.
    """
    df = df.copy()

    df = replace_question_with_nan(df)
    df = exclude_non_readmission_eligible_encounters(df)
    df = drop_early_columns(df)
    df = clean_invalid_gender_rows(df)

    if include_target:
        df = create_binary_target(df)

    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all deterministic feature engineering steps.
    """
    df = df.copy()

    if "age" in df.columns:
        df["age_ordinal"] = encode_age_ordinal(df["age"])

    df = handle_lab_results(df)
    df = add_diagnosis_category_features(df)
    df = add_diabetes_primary_flag(df)
    df = engineer_medication_features(df, MEDICATION_FEATURES)
    df = engineer_utilization_features(df)
    df = engineer_encounter_intensity_features(df)
    df = engineer_admission_type(df)
    df = engineer_admission_source(df)
    df = engineer_discharge_disposition(df)
    df = engineer_discharge_features(df)
    df = engineer_medical_specialty(df)
    df = engineer_race(df)

    return df


def run_pipeline(
    df: pd.DataFrame,
    include_target: bool = True,
):
    """
    Full end-to-end preparation pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.
    include_target : bool
        True for labeled training/validation/test data.
        False for future scoring/inference data.

    Returns
    -------
    X : pd.DataFrame
        Final approved feature matrix.
    y : pd.Series or None
        Target if include_target=True, else None.
    """
    df = df.copy()

    df = run_data_cleaning(df, include_target=include_target)
    df = run_feature_engineering(df)
    X, y = prepare_features(df, include_target=include_target)

    return X, y
