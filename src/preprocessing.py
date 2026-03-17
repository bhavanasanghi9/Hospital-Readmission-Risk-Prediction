import pandas as pd

# ── Constants ──────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "A1Cresult",
    "admission_source",
    "admission_type",
    "age_ordinal",
    "any_prior_emergency",
    "any_prior_inpatient",
    "any_prior_outpatient",
    "change",
    "diabetesMed",
    "diabetes_primary_diag",
    "diag_1_category",
    "diag_2_category",
    "diag_3_category",
    "discharge_disposition",
    "gender",
    "institutional_discharge",
    "insulin_change",
    "insulin_flag",
    "lab_intensity",
    "med_intensity",
    "medical_specialty_group",
    "n_active_diabetes_meds",
    "n_med_decreases",
    "n_med_increases",
    "num_lab_procedures",
    "num_medications",
    "num_procedures",
    "number_diagnoses",
    "number_emergency",
    "number_inpatient",
    "number_outpatient",
    "prior_utilization_score",
    "race",
    "time_in_hospital",
    "total_prior_visits",
]

TARGET_COLUMN = "target_30day_readmit"


# ── Functions ──────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    include_target: bool = False,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Select the approved feature columns from a fully engineered
    dataframe and optionally return the target variable.

    This is the single entry point for preparing data for
    modeling — enforcing the exact approved feature set and
    raising an explicit error if any required column is missing.

    Parameters
    ----------
    df : pd.DataFrame
        Fully engineered dataframe — all feature engineering
        functions must be applied before calling this.
    include_target : bool
        If True, also returns the target series.
        Set True for train/validation, False for production scoring.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with approved columns only.
    y : pd.Series or None
        Target series if include_target=True, else None.

    Raises
    ------
    ValueError
        If any required feature column is missing from df.

    Examples
    --------
    # Training
    X_train, y_train = prepare_features(train_df, include_target=True)

    # Production scoring — no target available
    X_new, _ = prepare_features(new_df, include_target=False)
    """
    missing_cols = [
        c for c in FEATURE_COLUMNS if c not in df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Missing required feature columns: {missing_cols}"
        )

    X = df[FEATURE_COLUMNS].copy()

    if include_target:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' not found in df."
            )
        y = df[TARGET_COLUMN].copy()
    else:
        y = None

    return X, y


from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


BINARY_MAPS = {
    "gender": {"Female": 0, "Male": 1},
    "change": {"No": 0, "Ch": 1},
    "diabetesMed": {"No": 0, "Yes": 1},
}


def get_model_feature_groups() -> Dict[str, List[str]]:
    """
    Return the finalized feature groups for modeling.
    """
    return {
        "binary_features": [
            "gender",
            "change",
            "diabetesMed",
        ],
        "categorical_features": [
            "race",
            "A1Cresult",
            "admission_type",
            "admission_source",
            "discharge_disposition",
            "medical_specialty_group",
            "diag_1_category",
            "diag_2_category",
            "diag_3_category",
        ],
    }


def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert binary categorical variables to numeric 0/1.
    """
    df = df.copy()

    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df


def get_numeric_features(
    df: pd.DataFrame,
    binary_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    """
    Infer numeric features as all columns not in binary/categorical groups.
    """
    excluded = set(binary_features + categorical_features)
    return [col for col in df.columns if col not in excluded]


def one_hot_encode_features(
    df: pd.DataFrame,
    categorical_features: List[str],
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    One-hot encode categorical variables.
    """
    df = df.copy()

    existing_cats = [c for c in categorical_features if c in df.columns]

    df = pd.get_dummies(
        df,
        columns=existing_cats,
        drop_first=drop_first,
    )

    return df


def fit_numeric_scaler(
    df: pd.DataFrame,
    numeric_features: List[str],
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on numeric features and transform training data.
    """
    df = df.copy()
    scaler = StandardScaler()

    existing_numeric = [c for c in numeric_features if c in df.columns]

    if existing_numeric:
        df[existing_numeric] = scaler.fit_transform(df[existing_numeric])

    return df, scaler


def transform_numeric_with_scaler(
    df: pd.DataFrame,
    numeric_features: List[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Apply a pre-fitted StandardScaler to numeric features.
    """
    df = df.copy()

    existing_numeric = [c for c in numeric_features if c in df.columns]

    if existing_numeric:
        df[existing_numeric] = scaler.transform(df[existing_numeric])

    return df


def find_near_constant_features(
    df: pd.DataFrame,
    threshold: float = 0.995,
) -> List[str]:
    """
    Identify near-constant features where the most frequent value exceeds
    the specified frequency threshold.
    """
    near_constant = []

    for col in df.columns:
        vc = df[col].value_counts(normalize=True, dropna=False)
        if len(vc) > 0 and vc.iloc[0] > threshold:
            near_constant.append(col)

    return near_constant


def drop_columns_if_present(
    df: pd.DataFrame,
    cols: List[str],
) -> pd.DataFrame:
    """
    Drop columns only if they exist.
    """
    df = df.copy()
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing)


def align_columns_to_train(
    df: pd.DataFrame,
    reference_columns: List[str],
) -> pd.DataFrame:
    """
    Align a dataframe to the training columns after one-hot encoding.
    Missing columns are added with zeros; extra columns are removed.
    """
    df = df.copy()

    for col in reference_columns:
        if col not in df.columns:
            df[col] = 0

    extra_cols = [c for c in df.columns if c not in reference_columns]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    df = df[reference_columns]

    return df


NEAR_CONSTANT_COLUMNS = [
    "discharge_disposition_Unknown",
    "admission_source_Other",
    "admission_type_Other",
    "discharge_disposition_Long_Term_Care",
]

DUPLICATE_COLUMNS = [
    "diag_1_category_Diabetes",
]


def transform_test_to_match_train(
    X_test: pd.DataFrame,
    reference_columns: list[str],
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Apply the same encoding and column cleanup steps used on training data,
    then align the result to the encoded training columns.
    """
    X_test = X_test.copy()

    groups = get_model_feature_groups()
    categorical_features = groups["categorical_features"]

    # 1. Binary encode
    X_test = encode_binary_features(X_test)

    # 2. One-hot encode
    X_test = one_hot_encode_features(
        X_test,
        categorical_features=categorical_features,
        drop_first=drop_first,
    )

    # 3. Drop train-identified near-constant columns
    X_test = drop_columns_if_present(X_test, NEAR_CONSTANT_COLUMNS)

    # 4. Drop train-identified duplicate columns
    X_test = drop_columns_if_present(X_test, DUPLICATE_COLUMNS)

    # 5. Align to encoded training columns
    X_test = align_columns_to_train(X_test, reference_columns)

    return X_test
