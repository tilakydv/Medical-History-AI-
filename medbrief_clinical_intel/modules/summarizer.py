"""
Feature 1: AI Patient Summarization Module.
Receives extracted patient information (JSON/text) and structured MRI findings,
generating a concise doctor-friendly summary strictly grounded in records.
"""

import json
from typing import Optional
from medbrief_clinical_intel.schemas.input_models import PatientRecordInput, MRIFindingsInput
from medbrief_clinical_intel.schemas.summary import PatientSummaryResponse
from medbrief_clinical_intel.llm.client import QwenLLMClient
from medbrief_clinical_intel.llm.prompt_templates import SystemPrompts, ClinicalPromptBuilder
from medbrief_clinical_intel.llm.parser import JSONOutputParser


class PatientSummarizer:
    """Module 1: AI Patient Summarizer."""

    def __init__(self, llm_client: Optional[QwenLLMClient] = None):
        self.llm_client = llm_client or QwenLLMClient()

    def generate_summary(self, patient_record: PatientRecordInput) -> PatientSummaryResponse:
        """Generate structured patient summary."""
        # 1. Format patient context
        context_str = self._format_context(patient_record)

        # 2. Schema definition for Qwen
        schema_json = json.dumps(PatientSummaryResponse.model_json_schema(), indent=2)

        # 3. Build prompt and invoke Qwen LLM
        prompt = ClinicalPromptBuilder.build_summarizer_prompt(context_str, schema_json)
        response_text = self.llm_client.generate(
            prompt, system_prompt=SystemPrompts.CLINICAL_INTEL_SYSTEM
        )

        # 4. Parse response
        model_result = JSONOutputParser.parse_to_model(response_text, PatientSummaryResponse)
        if model_result:
            model_result.patient_id = patient_record.patient_id
            return model_result

        # Fallback if raw JSON structure didn't parse perfectly
        parsed_dict = JSONOutputParser.parse_json(response_text)
        if isinstance(parsed_dict, dict):
            parsed_dict["patient_id"] = patient_record.patient_id
            return PatientSummaryResponse.model_validate(parsed_dict)

        # Fallback default
        return self._create_fallback_summary(patient_record)

    def _format_context(self, record: PatientRecordInput) -> str:
        """Format patient docs and MRI findings into prompt context."""
        parts = [
            f"Patient ID: {record.patient_id}",
            f"Demographics: Age {record.age or 'N/A'}, Gender {record.gender or 'N/A'}"
        ]

        if record.mri_findings:
            mri = record.mri_findings
            parts.append(
                f"\n--- Structured Brain MRI Findings (nnU-Net / MONAI Pipeline) ---\n"
                f"Tumor Detected: {mri.tumor_detected}\n"
                f"Tumor Type: {mri.tumor_type or 'Not specified'}\n"
                f"Location: {mri.location or 'Not specified'}\n"
                f"Volume (cmÂ³): {mri.volume_cm3 or 'N/A'} (Volume mmÂ³: {mri.volume_mm3 or 'N/A'})\n"
                f"Peritumoral Edema: {mri.edema_present}\n"
                f"Midline Shift: {mri.midline_shift}\n"
                f"Contrast Enhancement: {mri.contrast_enhancement or 'N/A'}\n"
                f"Additional Radiological Notes: {mri.additional_notes or 'None'}"
            )

        if record.documents:
            parts.append("\n--- Extracted Document Records (OCR / PyMuPDF) ---")
            for doc in record.documents:
                parts.append(
                    f"\nDocument ID: {doc.document_id} | Date: {doc.date or 'Unknown'} | Type: {doc.document_type}\n"
                    f"Text Content:\n{doc.extracted_text}\n"
                )

        return "\n".join(parts)

    def _create_fallback_summary(self, record: PatientRecordInput) -> PatientSummaryResponse:
        """Create fallback summary structure."""
        mri_str = None
        if record.mri_findings:
            mri_str = f"Tumor Detected: {record.mri_findings.tumor_detected}, Location: {record.mri_findings.location}, Volume: {record.mri_findings.volume_cm3} cmÂ³"
            
        return PatientSummaryResponse(
            patient_id=record.patient_id,
            patient_overview=f"Patient {record.patient_id}, Age: {record.age or 'N/A'}, Gender: {record.gender or 'N/A'}.",
            medical_history=[],
            diagnoses=[],
            current_conditions=[],
            mri_findings_summary=mri_str,
            laboratory_observations=[],
            current_medications=[],
            allergies=[],
            recommended_follow_up=[],
            narrative_summary_markdown=f"### Patient Summary Brief\n\nPatient ID: {record.patient_id}\nRecords parsed."
        )


