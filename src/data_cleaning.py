import numpy as np
import pandas as pd


EXCLUDE_DISPOSITION_IDS = [11, 13, 14, 19, 20, 21]
EARLY_DROP_COLS = ["weight", "payer_code"]


def replace_question_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace '?' placeholders with proper missing values.
    """
    return df.replace("?", np.nan)


def exclude_non_readmission_eligible_encounters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove encounters where patients expired or were discharged to hospice,
    since these encounters are not meaningfully at risk of readmission.
    """
    return df.loc[~df["discharge_disposition_id"].isin(EXCLUDE_DISPOSITION_IDS)].copy()


def drop_early_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with excessive missingness or limited modeling value.
    """
    cols_present = [col for col in EARLY_DROP_COLS if col in df.columns]
    return df.drop(columns=cols_present).copy()

def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary 30-day readmission target.
    1 = readmitted within 30 days
    0 = otherwise
    """
    df = df.copy()
    df["target_30day_readmit"] = (df["readmitted"] == "<30").astype(int)
    return df

def clean_invalid_gender_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where gender is coded as Unknown/Invalid.
    """
    df = df.copy()

    if "gender" in df.columns:
        df = df[df["gender"] != "Unknown/Invalid"]

    return df
