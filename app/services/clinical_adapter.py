from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.exceptions import NotFoundError, ProcessingError
from app.models import MRIResult, MRIScan, Patient, Report
from medbrief_clinical_intel.schemas.input_models import (
    DocumentRecord,
    MRIFindingsInput,
    PatientRecordInput,
)
from medbrief_clinical_intel.service import MedBriefClinicalService


class ClinicalIntelligenceAdapter:
    """Maps persisted backend records to the partner-owned clinical intelligence API."""

    def __init__(self, service: MedBriefClinicalService | None = None) -> None:
        self.service = service or MedBriefClinicalService()

    def build_record(self, db: Session, patient_id: str) -> PatientRecordInput:
        patient = db.scalar(
            select(Patient)
            .where(Patient.id == patient_id)
            .options(
                selectinload(Patient.reports),
                selectinload(Patient.mri_scans).selectinload(MRIScan.result),
            )
        )
        if not patient:
            raise NotFoundError("Patient not found")

        documents = [self._document(report) for report in patient.reports if report.extracted_text]
        latest_result = next(
            (scan.result for scan in sorted(patient.mri_scans, key=lambda item: item.created_at,
                                             reverse=True) if scan.result),
            None,
        )
        return PatientRecordInput(
            patient_id=patient.id,
            patient_name=patient.name,
            age=self._age(patient.date_of_birth),
            gender=patient.sex,
            documents=documents,
            mri_findings=self._mri(latest_result) if latest_result else None,
            metadata={"source": "medbrief_backend", "schema_version": "1.0"},
        )

    @staticmethod
    def _document(report: Report) -> DocumentRecord:
        return DocumentRecord(
            document_id=report.id,
            date=report.report_date.isoformat() if report.report_date else None,
            document_type=report.report_type,
            extracted_text=report.extracted_text or "",
            language="en",
        )

    @staticmethod
    def _mri(result: MRIResult) -> MRIFindingsInput:
        location = result.localization.get("anatomical_region")
        if not location and result.localization.get("centroid_voxel"):
            location = f"Voxel centroid {result.localization['centroid_voxel']}"
        return MRIFindingsInput(
            tumor_detected=bool(result.findings.get("segmentation_present")),
            tumor_type=result.findings.get("tumor_type"),
            location=location,
            volume_mm3=result.tumor_volume_mm3,
            volume_cm3=result.tumor_volume_mm3 / 1000,
            edema_present=result.findings.get("edema_present"),
            midline_shift=result.findings.get("midline_shift"),
            contrast_enhancement=result.findings.get("contrast_enhancement"),
            additional_notes=result.findings.get("disclaimer"),
        )

    @staticmethod
    def _age(born: date | None) -> int | None:
        if not born:
            return None
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    def execute(self, operation: str, record: PatientRecordInput, **kwargs: Any) -> Any:
        operations = {
            "summary": self.service.summarize,
            "timeline": self.service.get_timeline,
            "medications-allergies": self.service.analyze_medications,
            "contradictions": self.service.detect_contradictions,
            "missing-info": self.service.detect_missing_info,
            "full-analysis": self.service.run_full_clinical_pipeline,
        }
        handler = operations.get(operation)
        if not handler:
            raise NotFoundError("Clinical operation not found")
        try:
            return handler(record, **kwargs)
        except Exception as exc:
            raise ProcessingError(f"Clinical intelligence operation '{operation}' failed") from exc

