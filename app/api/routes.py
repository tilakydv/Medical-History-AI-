from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.deps import DB
from app.core.config import get_settings
from app.core.exceptions import NotFoundError
from app.models import LaboratoryReport, MRIResult, MRIScan, Patient, Report, Upload
from app.schemas.common import (LabReportRead, MRIRead, PatientCreate, PatientDetail,
                                PatientRead, ReportRead, UploadRead, UploadResponse)
from app.services.integration_service import LLMIntegrationService
from app.services.lab_service import LaboratoryService
from app.services.mri_service import MRIService
from app.services.ocr_service import OCRService
from app.services.upload_service import UploadService

router = APIRouter()


@router.post("/patients", response_model=PatientRead, status_code=201, tags=["patients"])
def create_patient(payload: PatientCreate, db: DB) -> Patient:
    patient = Patient(**payload.model_dump())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@router.get("/patients", response_model=list[PatientRead], tags=["patients"])
def list_patients(db: DB) -> list[Patient]:
    return list(db.scalars(select(Patient).order_by(Patient.created_at.desc())).all())


@router.post("/upload-report", response_model=UploadResponse, tags=["reports"])
async def upload_report(db: DB, patient_id: Annotated[str, Form()],
                        file: Annotated[UploadFile, File()],
                        report_type: Annotated[str, Form()] = "clinical") -> UploadResponse:
    upload, report, duplicate = await UploadService(db, get_settings()).save_report(
        patient_id, file, report_type)
    return UploadResponse(upload=UploadRead.model_validate(upload), resource_id=report.id,
                          duplicate=duplicate)


@router.post("/upload-mri", response_model=UploadResponse, tags=["mri"])
async def upload_mri(db: DB, patient_id: Annotated[str, Form()],
                     file: Annotated[UploadFile, File()]) -> UploadResponse:
    upload, scan, duplicate = await UploadService(db, get_settings()).save_mri(patient_id, file)
    if not duplicate:
        scan.metadata_json = MRIService(get_settings()).inspect(Path(upload.stored_path))
        scan.format = str(scan.metadata_json["format"])
        db.commit()
    return UploadResponse(upload=UploadRead.model_validate(upload), resource_id=scan.id,
                          duplicate=duplicate)


@router.post("/extract-report", response_model=ReportRead, tags=["reports"])
def extract_report(report_id: Annotated[str, Query()], db: DB) -> Report:
    report = db.get(Report, report_id)
    if not report:
        raise NotFoundError("Report not found")
    upload = db.get(Upload, report.upload_id)
    if not upload:
        raise NotFoundError("Source upload not found")
    result = OCRService(get_settings().ocr_lang).extract(Path(upload.stored_path))
    report.extracted_text = result.text
    report.extraction_method = result.method
    report.structured_data = {"pages": result.pages}
    report.status = "extracted"
    upload.status = "processed"
    if report.report_type.lower() in {"lab", "laboratory", "pathology"}:
        LaboratoryService().persist(db, report.id, report.patient_id, result.text)
    db.commit()
    db.refresh(report)
    return report


@router.post("/analyze-mri", response_model=MRIRead, tags=["mri"])
def analyze_mri(mri_id: Annotated[str, Query()], db: DB) -> MRIScan:
    scan = db.get(MRIScan, mri_id)
    if not scan:
        raise NotFoundError("MRI scan not found")
    if scan.result:
        return scan
    upload = db.get(Upload, scan.upload_id)
    if not upload:
        raise NotFoundError("Source upload not found")
    output_dir = get_settings().upload_dir / scan.patient_id / "mri_results" / scan.id
    values = MRIService(get_settings()).analyze(Path(upload.stored_path), output_dir)
    result = MRIResult(mri_scan_id=scan.id, **values)
    db.add(result)
    scan.status = "analyzed"
    upload.status = "processed"
    db.commit()
    return db.scalar(select(MRIScan).where(MRIScan.id == scan.id)
                     .options(selectinload(MRIScan.result)))  # type: ignore[return-value]


@router.get("/patient/{patient_id}", response_model=PatientDetail, tags=["patients"])
def get_patient(patient_id: str, db: DB) -> Patient:
    patient = db.scalar(select(Patient).where(Patient.id == patient_id).options(
        selectinload(Patient.reports),
        selectinload(Patient.mri_scans).selectinload(MRIScan.result)))
    if not patient:
        raise NotFoundError("Patient not found")
    return patient


@router.get("/reports/{report_id}", response_model=ReportRead, tags=["reports"])
def get_report(report_id: str, db: DB) -> Report:
    report = db.get(Report, report_id)
    if not report:
        raise NotFoundError("Report not found")
    return report


@router.get("/patients/{patient_id}/reports", tags=["reports"])
def patient_reports(patient_id: str, db: DB) -> list[dict[str, Any]]:
    """List persisted reports with their source filename for patient-facing selection."""
    if not db.get(Patient, patient_id):
        raise NotFoundError("Patient not found")
    rows = db.execute(
        select(Report, Upload)
        .join(Upload, Upload.id == Report.upload_id)
        .where(Report.patient_id == patient_id)
        .order_by(Report.created_at.desc())
    ).all()
    return [
        {
            **ReportRead.model_validate(report).model_dump(mode="json"),
            "original_filename": upload.original_filename,
            "upload_status": upload.status,
        }
        for report, upload in rows
    ]


@router.get("/reports/{report_id}/lab-overview", tags=["laboratory"])
def report_lab_overview(report_id: str, db: DB) -> dict[str, Any]:
    report = db.get(Report, report_id)
    if not report:
        raise NotFoundError("Report not found")
    if not report.extracted_text:
        raise NotFoundError("Extract the report before requesting its overview")
    return LaboratoryService().overview(report.extracted_text)


@router.get("/mri/{mri_id}", response_model=MRIRead, tags=["mri"])
def get_mri(mri_id: str, db: DB) -> MRIScan:
    scan = db.scalar(select(MRIScan).where(MRIScan.id == mri_id)
                     .options(selectinload(MRIScan.result)))
    if not scan:
        raise NotFoundError("MRI scan not found")
    return scan


@router.get("/mri/{mri_id}/overlay", tags=["mri"])
def get_mri_overlay(mri_id: str, db: DB) -> FileResponse:
    result = db.scalar(select(MRIResult).where(MRIResult.mri_scan_id == mri_id))
    if not result or not result.overlay_path:
        raise NotFoundError("MRI overlay not found")
    path = Path(result.overlay_path)
    if not path.is_file():
        raise NotFoundError("MRI overlay file not found")
    return FileResponse(path, media_type="image/png", filename=f"{mri_id}-overlay.png")


@router.get("/labs/{lab_id}", response_model=LabReportRead, tags=["laboratory"])
def get_labs(lab_id: str, db: DB) -> LaboratoryReport:
    lab = db.scalar(select(LaboratoryReport).where(LaboratoryReport.id == lab_id)
                    .options(selectinload(LaboratoryReport.values)))
    if not lab:
        raise NotFoundError("Laboratory report not found")
    return lab


@router.get("/patients/{patient_id}/lab-trends", tags=["laboratory"])
def lab_trends(patient_id: str, db: DB) -> dict[str, Any]:
    if not db.get(Patient, patient_id):
        raise NotFoundError("Patient not found")
    return {"patient_id": patient_id, "series": LaboratoryService().trends(db, patient_id)}


@router.get("/integration/reports/{report_id}", tags=["integration"])
def report_integration(report_id: str, db: DB) -> dict[str, Any]:
    report = db.scalar(select(Report).where(Report.id == report_id)
                       .options(selectinload(Report.lab_report)
                                .selectinload(LaboratoryReport.values)))
    if not report:
        raise NotFoundError("Report not found")
    return LLMIntegrationService.report_payload(report)


@router.get("/integration/mri/{mri_id}", tags=["integration"])
def mri_integration(mri_id: str, db: DB) -> dict[str, Any]:
    result = db.scalar(select(MRIResult).where(MRIResult.mri_scan_id == mri_id))
    if not result:
        raise NotFoundError("MRI result not found")
    return LLMIntegrationService.mri_payload(result)


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}
