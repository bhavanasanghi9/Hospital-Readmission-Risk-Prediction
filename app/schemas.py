from pydantic import BaseModel
from typing import Optional, Literal

# Valid medication status values
MedStatus = Literal["No", "Steady", "Up", "Down"]

class PatientRecord(BaseModel):
    # Identifiers
    encounter_id: int
    patient_nbr: int
    # Demographics
    race: Optional[str] = None
    gender: Literal["Male", "Female"]
    age: str
    # Admission details
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    medical_specialty: Optional[str] = None
    # Encounter metrics
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    # Diagnosis codes
    diag_1: Optional[str] = None
    diag_2: Optional[str] = None
    diag_3: Optional[str] = None
    # Lab results
    max_glu_serum: Optional[str] = None
    A1Cresult: Optional[str] = None
    # Medications
    metformin: MedStatus
    repaglinide: MedStatus
    nateglinide: MedStatus
    chlorpropamide: MedStatus
    glimepiride: MedStatus
    acetohexamide: MedStatus
    glipizide: MedStatus
    glyburide: MedStatus
    tolbutamide: MedStatus
    pioglitazone: MedStatus
    rosiglitazone: MedStatus
    acarbose: MedStatus
    miglitol: MedStatus
    troglitazone: MedStatus
    tolazamide: MedStatus
    examide: MedStatus
    citoglipton: MedStatus
    insulin: MedStatus
    glyburide_metformin: MedStatus
    glipizide_metformin: MedStatus
    glimepiride_pioglitazone: MedStatus
    metformin_rosiglitazone: MedStatus
    metformin_pioglitazone: MedStatus
    # Diabetes flags
    change: Literal["Ch", "No"]
    diabetesMed: Literal["Yes", "No"]

class PredictionResponse(BaseModel):
    encounter_id: int
    risk_score: float
    risk_tier: str