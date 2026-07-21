from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class PatientCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    external_id: str | None = Field(default=None, max_length=100)
    date_of_birth: date | None = None
    sex: str | None = Field(default=None, max_length=30)


class PatientUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    external_id: str | None = Field(default=None, max_length=100)
    date_of_birth: date | None = None
    sex: str | None = Field(default=None, max_length=30)


class PatientRead(ORMModel):
    id: str
    external_id: str | None
    name: str
    date_of_birth: date | None
    sex: str | None
    created_at: datetime


class UploadRead(ORMModel):
    id: str
    original_filename: str
    content_type: str
    file_type: str
    size_bytes: int
    sha256: str
    status: str
    created_at: datetime


class ReportRead(ORMModel):
    id: str
    patient_id: str
    upload_id: str
    report_type: str
    report_date: date | None
    extracted_text: str | None
    structured_data: dict[str, Any] | None
    extraction_method: str | None
    status: str
    created_at: datetime


class LabValueRead(ORMModel):
    id: str
    test_code: str | None
    test_name: str
    value_numeric: float | None
    value_text: str | None
    unit: str | None
    reference_low: float | None
    reference_high: float | None
    flag: str | None
    observed_at: datetime | None


class LabReportRead(ORMModel):
    id: str
    report_id: str
    patient_id: str
    collected_at: datetime | None
    values: list[LabValueRead]


class MRIResultRead(ORMModel):
    id: str
    model_name: str
    model_version: str | None
    tumor_volume_mm3: float
    confidence_score: float | None
    localization: dict[str, Any]
    findings: dict[str, Any]
    mask_path: str
    overlay_path: str | None


class MRIRead(ORMModel):
    id: str
    patient_id: str
    upload_id: str
    modality: str
    format: str
    study_date: date | None
    status: str
    metadata_json: dict[str, Any] | None
    created_at: datetime
    result: MRIResultRead | None = None


class PatientDetail(PatientRead):
    reports: list[ReportRead]
    mri_scans: list[MRIRead]


class UploadResponse(BaseModel):
    upload: UploadRead
    resource_id: str
    duplicate: bool = False


class ErrorResponse(BaseModel):
    error: str
    message: str

