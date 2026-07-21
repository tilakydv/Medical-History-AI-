from typing import Any

from app.models import MRIResult, Report


class LLMIntegrationService:
    """Stable, normalized contracts consumed by the separately owned LLM modules."""

    @staticmethod
    def report_payload(report: Report) -> dict[str, Any]:
        labs = report.lab_report.values if report.lab_report else []
        return {"schema_version": "1.0", "source": {"type": "report", "id": report.id},
                "patient_id": report.patient_id, "document": {"type": report.report_type,
                "date": report.report_date.isoformat() if report.report_date else None,
                "clean_text": report.extracted_text},
                "laboratory_values": [{"name": v.test_name, "value": v.value_numeric
                if v.value_numeric is not None else v.value_text, "unit": v.unit,
                "reference_range": {"low": v.reference_low, "high": v.reference_high},
                "flag": v.flag, "observed_at": v.observed_at.isoformat() if v.observed_at else None}
                for v in labs]}

    @staticmethod
    def mri_payload(result: MRIResult) -> dict[str, Any]:
        return {"schema_version": "1.0", "source": {"type": "mri", "id": result.mri_scan_id},
                "model": {"name": result.model_name, "version": result.model_version},
                "segmentation": {"tumor_volume_mm3": result.tumor_volume_mm3,
                "confidence": result.confidence_score, "localization": result.localization},
                "findings": result.findings}
