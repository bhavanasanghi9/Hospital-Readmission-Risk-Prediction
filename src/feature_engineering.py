from typing import Dict, List


NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

ORDINAL_FEATURES = [
    "age",
]

LAB_RESULT_FEATURES = [
    "max_glu_serum",
    "A1Cresult",
]

DIAGNOSIS_FEATURES = [
    "diag_1",
    "diag_2",
    "diag_3",
]

ID_ENCODED_FEATURES = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

CONTEXTUAL_FEATURES = [
    "medical_specialty",
]

MEDICATION_FEATURES = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

BINARY_FEATURES = [
    "change",
    "diabetesMed",
]

NOMINAL_FEATURES = [
    "race",
    "gender",
]


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Return the project-specific feature grouping dictionary.
    """
    return {
        "numeric_features": NUMERIC_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,
        "lab_result_features": LAB_RESULT_FEATURES,
        "diagnosis_features": DIAGNOSIS_FEATURES,
        "id_encoded_features": ID_ENCODED_FEATURES,
        "contextual_features": CONTEXTUAL_FEATURES,
        "medication_features": MEDICATION_FEATURES,
        "binary_features": BINARY_FEATURES,
        "nominal_features": NOMINAL_FEATURES,
    }

def encode_age_ordinal(age_series):
    """
    Convert age buckets to ordinal index 0–9.
    """
    age_map = {
        "[0-10)": 0,
        "[10-20)": 1,
        "[20-30)": 2,
        "[30-40)": 3,
        "[40-50)": 4,
        "[50-60)": 5,
        "[60-70)": 6,
        "[70-80)": 7,
        "[80-90)": 8,
        "[90-100)": 9
    }

    return age_series.map(age_map)

def handle_lab_results(df):
    df = df.copy()

    # drop extremely sparse glucose test
    if "max_glu_serum" in df.columns:
        df = df.drop(columns=["max_glu_serum"])

    # keep A1C and encode missing
    if "A1Cresult" in df.columns:
        df["A1Cresult"] = df["A1Cresult"].fillna("Not_Measured")

    return df

import pandas as pd


def map_icd9_to_category(code) -> str:
    """
    Map an ICD-9 diagnosis code to a broad clinical category.

    Grouping follows the methodology used in:
    Strack et al. (2014), Impact of HbA1c Measurement on Hospital Readmission Rates.
    """
    if pd.isna(code):
        return "Other"

    code = str(code).strip().upper()

    if code in ["UNKNOWN", "?", ""]:
        return "Other"

    # Diabetes: all 250.xx codes
    if code.startswith("250"):
        return "Diabetes"

    # Supplementary / external cause codes
    if code.startswith("V") or code.startswith("E"):
        return "Other"

    try:
        code_num = float(code)
    except ValueError:
        return "Other"

    if (390 <= code_num <= 459) or (code_num == 785):
        return "Circulatory"
    elif (460 <= code_num <= 519) or (code_num == 786):
        return "Respiratory"
    elif (520 <= code_num <= 579) or (code_num == 787):
        return "Digestive"
    elif (580 <= code_num <= 629) or (code_num == 788):
        return "Genitourinary"
    elif 710 <= code_num <= 739:
        return "Musculoskeletal"
    elif 140 <= code_num <= 239:
        return "Neoplasms"
    elif 800 <= code_num <= 999:
        return "Injury"
    else:
        return "Other"


def add_diagnosis_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create diagnosis-category features from raw ICD-9 code columns.
    """
    df = df.copy()

    diag_cols = ["diag_1", "diag_2", "diag_3"]

    for col in diag_cols:
        if col in df.columns:
            df[f"{col}_category"] = df[col].apply(map_icd9_to_category)

    return df


def add_diabetes_primary_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary flag indicating whether diabetes is the primary diagnosis.
    Requires diag_1_category to exist.
    """
    df = df.copy()

    if "diag_1_category" in df.columns:
        df["diabetes_primary_diag"] = (
            df["diag_1_category"] == "Diabetes"
        ).astype(int)

    return df

import pandas as pd


def engineer_medication_features(
    df: pd.DataFrame,
    medication_cols: list[str]
) -> pd.DataFrame:
    """
    Create aggregated medication features from the diabetes medication columns.

    Definitions
    ----------
    active medication:
        status in {"Steady", "Up", "Down"}
    medication increase:
        status == "Up"
    medication decrease:
        status == "Down"

    Engineered features
    -------------------
    n_active_diabetes_meds : count of active diabetes medications
    n_med_increases        : count of medications with dose increase
    n_med_decreases        : count of medications with dose decrease
    insulin_flag           : 1 if insulin is active, else 0
    insulin_change         : 1 if insulin status is Up/Down, else 0
    """
    df = df.copy()

    active_values = {"Steady", "Up", "Down"}

    # Count active diabetes medications
    df["n_active_diabetes_meds"] = (
        df[medication_cols]
        .isin(active_values)
        .sum(axis=1)
    )

    # Count increases and decreases separately
    df["n_med_increases"] = (
        df[medication_cols]
        .eq("Up")
        .sum(axis=1)
    )

    df["n_med_decreases"] = (
        df[medication_cols]
        .eq("Down")
        .sum(axis=1)
    )

    # Insulin-specific features
    if "insulin" in df.columns:
        df["insulin_flag"] = (df["insulin"] != "No").astype(int)
        df["insulin_change"] = df["insulin"].isin(["Up", "Down"]).astype(int)

    return df

def engineer_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create utilization-based features from prior outpatient, emergency,
    and inpatient visit counts.

    Engineered features
    -------------------
    any_prior_outpatient   : 1 if number_outpatient > 0 else 0
    any_prior_emergency    : 1 if number_emergency > 0 else 0
    any_prior_inpatient    : 1 if number_inpatient > 0 else 0
    total_prior_visits     : sum of outpatient + emergency + inpatient visits
    prior_utilization_score: acuity-weighted utilization burden
                             (inpatient*3 + emergency*2 + outpatient*1)
    """
    df = df.copy()

    if "number_outpatient" in df.columns:
        df["any_prior_outpatient"] = (df["number_outpatient"] > 0).astype(int)

    if "number_emergency" in df.columns:
        df["any_prior_emergency"] = (df["number_emergency"] > 0).astype(int)

    if "number_inpatient" in df.columns:
        df["any_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)

    required_cols = {"number_outpatient", "number_emergency", "number_inpatient"}
    if required_cols.issubset(df.columns):
        df["total_prior_visits"] = (
            df["number_outpatient"]
            + df["number_emergency"]
            + df["number_inpatient"]
        )

        df["prior_utilization_score"] = (
            df["number_inpatient"] * 3
            + df["number_emergency"] * 2
            + df["number_outpatient"] * 1
        )

    return df

def engineer_encounter_intensity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create encounter intensity features normalized by length of stay.

    med_intensity : medications per hospital day
    lab_intensity : lab procedures per hospital day
    """
    df = df.copy()

    if {"num_medications", "time_in_hospital"}.issubset(df.columns):
        df["med_intensity"] = (
            df["num_medications"] / df["time_in_hospital"]
        )

    if {"num_lab_procedures", "time_in_hospital"}.issubset(df.columns):
        df["lab_intensity"] = (
            df["num_lab_procedures"] / df["time_in_hospital"]
        )

    return df

def engineer_admission_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map admission_type_id to clinically interpretable categories.
    """
    
    df = df.copy()

    def map_admission_type(x):

        if x == 1:
            return "Emergency"

        elif x == 2:
            return "Urgent"

        elif x == 3:
            return "Elective"

        elif x in [4, 7, 8]:
            return "Other"

        elif x in [5, 6]:
            return "Unknown"

        else:
            return "Unknown"

    df["admission_type"] = df["admission_type_id"].apply(map_admission_type)

    return df

def engineer_admission_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map admission_source_id into clinically meaningful groups.
    """
    
    df = df.copy()

    def map_admission_source(x):

        if x in [1,2,3]:
            return "Physician_Referral"

        elif x == 7:
            return "Emergency_Room"

        elif x in [4,5,6,17]:
            return "Transfer"

        elif pd.isna(x):
            return "Unknown"

        else:
            return "Other"

    df["admission_source"] = df["admission_source_id"].apply(map_admission_source)

    return df

def engineer_discharge_disposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map discharge_disposition_id into clinically meaningful groups.

    Categories
    ----------
    Home
    Skilled_Nursing
    Home_Health
    Transfer
    Rehab
    Long_Term_Care
    AMA
    Other
    Unknown
    """
    df = df.copy()

    def map_discharge_disposition(x):

        if x == 1:
            return "Home"

        elif x in [3, 15, 24]:
            return "Skilled_Nursing"

        elif x == 6:
            return "Home_Health"

        elif x in [2, 4, 5]:
            return "Transfer"

        elif x == 22:
            return "Rehab"

        elif x == 23:
            return "Long_Term_Care"

        elif x == 7:
            return "AMA"

        elif x in [18, 25]:
            return "Other"

        else:
            return "Unknown"

    if "discharge_disposition_id" in df.columns:
        df["discharge_disposition"] = (
            df["discharge_disposition_id"].apply(map_discharge_disposition)
        )

    return df

def engineer_discharge_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional discharge-related features.
    """

    df = df.copy()

    if "discharge_disposition" in df.columns:

        df["institutional_discharge"] = (
            df["discharge_disposition"].isin([
                "Skilled_Nursing",
                "Rehab",
                "Long_Term_Care",
                "Transfer"
            ])
        ).astype(int)

    return df

def engineer_medical_specialty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group medical_specialty into broader clinical categories.
    """

    df = df.copy()

    def map_medical_specialty(x):

        if pd.isna(x) or x == "?":
            return "Unknown"

        x = str(x)

        if x in ["InternalMedicine", "Family/GeneralPractice", "Hospitalist"]:
            return "Primary_Care"

        elif x == "Emergency/Trauma":
            return "Emergency"

        elif x.startswith("Cardiology"):
            return "Cardiology"

        elif x.startswith("Surgery"):
            return "Surgery"

        elif x.startswith("Orthopedics"):
            return "Orthopedics"

        elif x == "Nephrology":
            return "Nephrology"

        elif x == "Pulmonology":
            return "Pulmonology"

        elif x in ["Psychiatry", "Psychology", "Psychiatry-Child/Adolescent"]:
            return "Mental_Health"

        elif x in ["Radiology", "Radiologist"]:
            return "Radiology"

        elif x in [
            "Neurology",
            "Gastroenterology",
            "Oncology",
            "Hematology/Oncology",
            "Endocrinology",
            "Endocrinology-Metabolism",
            "InfectiousDiseases"
        ]:
            return "Specialty_Medicine"

        else:
            return "Other"

    if "medical_specialty" in df.columns:
        df["medical_specialty_group"] = (
            df["medical_specialty"].apply(map_medical_specialty)
        )

    return df

def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize gender values to Male, Female, or Unknown.

    Maps 'Unknown/Invalid' to 'Unknown' for consistency.
    Three Unknown/Invalid cases exist in training data — retained
    rather than dropped to avoid demographic-based exclusion.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'gender' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with gender values standardized.
    """
    df = df.copy()

    gender_map = {
        'Male': 'Male',
        'Female': 'Female',
        'Unknown/Invalid': 'Unknown'
    }

    if 'gender' in df.columns:
        df['gender'] = df['gender'].map(gender_map).fillna('Unknown')

    return df


def engineer_race(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize race values and keep missing/unknown values explicit.
    """
    df = df.copy()

    if "race" not in df.columns:
        return df

    race_map = {
        "AfricanAmerican": "AfricanAmerican",
        "African American": "AfricanAmerican",
        "Caucasian": "Caucasian",
        "Asian": "Asian",
        "Hispanic": "Hispanic",
        "Other": "Other",
        "?": "Unknown",
        "Unknown/Invalid": "Unknown",
    }

    race_series = df["race"].astype("string").str.strip()
    df["race"] = race_series.map(race_map).fillna("Unknown")

    return df
