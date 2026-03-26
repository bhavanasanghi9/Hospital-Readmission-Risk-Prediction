import pytest
from pydantic import ValidationError
from app.schemas import PatientRecord


def test_valid_patient_record(sample_patient):
    record = PatientRecord(**sample_patient)
    assert record.gender == "Female"
    assert record.insulin == "Up"
    assert record.encounter_id == 2278392


def test_invalid_gender(sample_patient):
    sample_patient["gender"] = "Unknown"
    with pytest.raises(ValidationError):
        PatientRecord(**sample_patient)


def test_invalid_medication_value(sample_patient):
    sample_patient["insulin"] = "banana"
    with pytest.raises(ValidationError):
        PatientRecord(**sample_patient)


def test_invalid_change_value(sample_patient):
    sample_patient["change"] = "Yes"
    with pytest.raises(ValidationError):
        PatientRecord(**sample_patient)


def test_optional_fields_can_be_none(sample_patient):
    sample_patient["race"] = None
    sample_patient["max_glu_serum"] = None
    sample_patient["A1Cresult"] = None
    sample_patient["medical_specialty"] = None
    record = PatientRecord(**sample_patient)
    assert record.race is None
    assert record.max_glu_serum is None