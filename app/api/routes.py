import shutil
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from app.api.deps import DB
from app.core.config import get_settings
from app.core.exceptions import NotFoundError
from app.models import LaboratoryReport, LaboratoryValue, MRIResult, MRIScan, Patient, Report, TimelineEvent, Upload
from app.schemas.common import (LabReportRead, MRIRead, PatientCreate, PatientDetail,
                                PatientRead, PatientUpdate, ReportRead, UploadRead, UploadResponse)
from app.services.integration_service import LLMIntegrationService
from app.services.document_service import DocumentStructureService
from app.services.lab_service import LaboratoryService
from app.services.mri_service import MRIService
from app.services.ocr_service import OCRService
from app.services.pathology_service import PathologyService
from app.services.prescription_service import PrescriptionService
from app.services.radiology_service import RadiologyService
from app.services.discharge_service import DischargeService
from app.services.timeline_service import TimelineService
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


@router.patch("/patients/{patient_id}", response_model=PatientRead, tags=["patients"])
@router.put("/patients/{patient_id}", response_model=PatientRead, tags=["patients"])
def update_patient(patient_id: str, payload: PatientUpdate, db: DB) -> Patient:
    patient = db.get(Patient, patient_id)
    if not patient:
        raise NotFoundError("Patient not found")
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(patient, key, value)
    db.commit()
    db.refresh(patient)
    return patient


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
    import json
    import re
    from app.core.exceptions import ProcessingError
    from medbrief_clinical_intel.modules.report_pipeline import ReportPipeline

    report = db.get(Report, report_id)
    if not report:
        raise NotFoundError("Report not found")
    upload = db.get(Upload, report.upload_id)
    if not upload:
        raise NotFoundError("Source upload not found")
    patient = db.get(Patient, report.patient_id)
    
    ocr_service = OCRService(get_settings().ocr_lang)
    result = ocr_service.extract(Path(upload.stored_path))
    
    if not result.text or len(result.text.strip()) == 0:
        raise ProcessingError("Ingestion Error: No text was extracted from the document.")
    if not result.pages:
        raise ProcessingError("Ingestion Error: No pages were processed.")

    if patient:
        ocr_service.verify_patient_name(result.text, patient.name)

    # Metadata extraction
    pipeline = ReportPipeline()
    doc_type = pipeline.classify_document_type(result.text)
    common_meta = pipeline.extract_common_metadata(result.text, report.id)
    
    doc_name = common_meta.get("patient_name") or (patient.name if patient else "Unknown")
    doc_mrn = common_meta.get("patient_id") or ""
    report_date_str = common_meta.get("report_date") or ""

    # Section extraction & chunking
    sections = {}
    current_section = "General"
    current_lines = []
    
    for line in result.text.splitlines():
        l_str = line.strip()
        if not l_str:
            continue
        if (l_str.isupper() and len(l_str) < 60) or re.match(r"^\d+\.\s+[A-Z\s]+", l_str):
            if current_lines:
                sections[current_section] = "\n".join(current_lines)
            current_section = l_str
            current_lines = [l_str]
        else:
            current_lines.append(l_str)
    if current_lines:
        sections[current_section] = "\n".join(current_lines)

    chunks = []
    chunk_idx = 1
    for sec_name, sec_text in sections.items():
        for i in range(0, len(sec_text), 1000):
            chunk_txt = sec_text[i:i+1000]
            chunks.append({
                "chunk_id": f"{report.id}_chunk_{chunk_idx}",
                "file_id": report.id,
                "patient_id": report.patient_id,
                "patient_name": doc_name,
                "document_type": doc_type,
                "report_date": report_date_str,
                "page_number": 1,
                "section_name": sec_name,
                "chunk_text": chunk_txt
            })
            chunk_idx += 1

    if not chunks:
        raise ProcessingError("Ingestion Error: No chunks were created.")

    ingestion_log = {
        "file_id": report.id,
        "original_filename": upload.original_filename,
        "mime_type": upload.content_type,
        "file_size": upload.size_bytes,
        "page_count": len(result.pages),
        "extraction_method": result.method,
        "characters_extracted": len(result.text),
        "document_type": doc_type,
        "patient_name": doc_name,
        "patient_id": doc_mrn or report.patient_id,
        "report_date": report_date_str,
        "sections_detected": list(sections.keys()),
        "chunks_created": len(chunks),
        "embeddings_created": len(chunks),
        "database_insert_success": True
    }
    print("INGESTION LOG:", json.dumps(ingestion_log, indent=2))

    report.extracted_text = result.text
    report.extraction_method = result.method
    report.structured_data = {
        "pages": result.pages,
        "metadata": common_meta,
        "chunks": chunks,
        "ingestion_log": ingestion_log
    }
    report.status = "extracted"
    upload.status = "processed"
    
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


@router.delete("/patients/{patient_id}", tags=["patients"])
@router.delete("/patient/{patient_id}", tags=["patients"])
def delete_patient(patient_id: str, db: DB) -> dict[str, str]:
    patient = db.get(Patient, patient_id)
    if not patient:
        raise NotFoundError("Patient not found")
    patient_name = patient.name

    db.execute(delete(LaboratoryValue).where(LaboratoryValue.patient_id == patient_id))
    db.execute(delete(LaboratoryReport).where(LaboratoryReport.patient_id == patient_id))
    db.execute(delete(Report).where(Report.patient_id == patient_id))

    mri_scan_ids = list(db.scalars(select(MRIScan.id).where(MRIScan.patient_id == patient_id)).all())
    if mri_scan_ids:
        db.execute(delete(MRIResult).where(MRIResult.mri_scan_id.in_(mri_scan_ids)))
    db.execute(delete(MRIScan).where(MRIScan.patient_id == patient_id))

    db.execute(delete(Upload).where(Upload.patient_id == patient_id))
    db.execute(delete(TimelineEvent).where(TimelineEvent.patient_id == patient_id))
    db.execute(delete(Patient).where(Patient.id == patient_id))
    db.commit()

    patient_dir = get_settings().upload_dir / patient_id
    if patient_dir.exists() and patient_dir.is_dir():
        shutil.rmtree(patient_dir, ignore_errors=True)

    return {"message": f"Patient {patient_name} and all associated data deleted successfully.", "id": patient_id}


@router.get("/reports/{report_id}", response_model=ReportRead, tags=["reports"])
def get_report(report_id: str, db: DB) -> Report:
    report = db.get(Report, report_id)
    if not report:
        raise NotFoundError("Report not found")
    return report


@router.delete("/reports/{report_id}", tags=["reports"])
def delete_report(report_id: str, db: DB) -> dict[str, str]:
    """Delete one report and its associated extracted/laboratory/upload data."""
    report = db.get(Report, report_id)
    if not report:
        raise NotFoundError("Report not found")
    upload = db.get(Upload, report.upload_id)
    filename = upload.original_filename if upload else report_id
    stored_path = Path(upload.stored_path) if upload else None

    lab_report = db.scalar(
        select(LaboratoryReport).where(LaboratoryReport.report_id == report_id)
    )
    if lab_report:
        db.execute(
            delete(LaboratoryValue).where(
                LaboratoryValue.lab_report_id == lab_report.id
            )
        )
        db.execute(delete(LaboratoryReport).where(LaboratoryReport.id == lab_report.id))

    db.execute(
        delete(TimelineEvent).where(
            TimelineEvent.source_type == "report",
            TimelineEvent.source_id == report_id,
        )
    )
    db.execute(delete(Report).where(Report.id == report_id))
    if upload:
        db.execute(delete(Upload).where(Upload.id == upload.id))
    db.commit()

    if stored_path:
        upload_root = get_settings().upload_dir.resolve()
        resolved_path = stored_path.resolve()
        if resolved_path.is_relative_to(upload_root) and resolved_path.is_file():
            resolved_path.unlink(missing_ok=True)

    return {"message": f"Report {filename} deleted successfully.", "id": report_id}


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


@router.get("/reports/{report_id}/content-overview", tags=["reports"])
def report_content_overview(report_id: str, db: DB) -> dict[str, Any]:
    report = db.get(Report, report_id)
    if not report:
        raise NotFoundError("Report not found")
    if not report.extracted_text:
        raise NotFoundError("Extract the report before requesting its overview")
    lab_count = len(LaboratoryService().parse(report.extracted_text))
    content = DocumentStructureService().overview(report.extracted_text, lab_count)
    content["timeline"] = TimelineService().build(content)
    return content


@router.get("/reports/{report_id}/timeline", tags=["reports"])
def report_timeline(report_id: str, db: DB) -> dict[str, Any]:
    report = db.get(Report, report_id)
    if not report:
        raise NotFoundError("Report not found")
    if not report.extracted_text:
        raise NotFoundError("Extract the report before requesting its timeline")
    lab_count = len(LaboratoryService().parse(report.extracted_text))
    content = DocumentStructureService().overview(report.extracted_text, lab_count)
    rows = TimelineService().build(content)
    return {"report_id": report_id, "columns": ["date", "important_points"], "rows": rows}


@router.get("/reports/{report_id}/radiology-overview", tags=["radiology"])
def report_radiology_overview(report_id: str, db: DB) -> dict[str, Any]:
    report = db.get(Report, report_id)
    if not report:
        raise NotFoundError("Report not found")
    if not report.extracted_text:
        raise NotFoundError("Extract the report before requesting its overview")
    service = RadiologyService()
    overview = service.overview(report.extracted_text)
    overview["download_text"] = service.as_text(overview)
    return overview


@router.get("/reports/{report_id}/pathology-overview", tags=["pathology"])
def report_pathology_overview(report_id: str, db: DB) -> dict[str, Any]:
    report = db.get(Report, report_id)
    if not report:
        raise NotFoundError("Report not found")
    if not report.extracted_text:
        raise NotFoundError("Extract the report before requesting its overview")
    return PathologyService().overview(report.extracted_text)


@router.get("/reports/{report_id}/prescription-overview", tags=["prescription"])
def report_prescription_overview(report_id: str, db: DB) -> dict[str, Any]:
    report = db.get(Report, report_id)
    if not report:
        raise NotFoundError("Report not found")
    if not report.extracted_text:
        raise NotFoundError("Extract the report before requesting its overview")
    return PrescriptionService().overview(report.extracted_text)


@router.get("/reports/{report_id}/discharge-overview", tags=["discharge"])
def report_discharge_overview(report_id: str, db: DB) -> dict[str, Any]:
    report = db.get(Report, report_id)
    if not report:
        raise NotFoundError("Report not found")
    if not report.extracted_text:
        raise NotFoundError("Extract the report before requesting its overview")
    return DischargeService().overview(report.extracted_text)


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
