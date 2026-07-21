"""
Feature 1: AI Patient Summarization Module.
Receives extracted patient information (JSON/text) and structured MRI findings,
generating a concise doctor-friendly summary strictly grounded in records.
"""

import json
import re
from typing import Optional
from medbrief_clinical_intel.schemas.input_models import PatientRecordInput
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
        # Development mode must never manufacture clinical facts. When a real model is not
        # configured, return a deterministic summary made only from persisted source records.
        if self.llm_client.config.use_mock_llm or not self.llm_client.is_loaded:
            return self._create_grounded_summary(patient_record)

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

    @staticmethod
    def _clean_lines(text: str) -> list[str]:
        lines: list[str] = []
        for raw in text.splitlines():
            line = re.sub(r"\s+", " ", raw).strip(" -\t")
            if len(line) >= 3 and line not in lines:
                lines.append(line)
        return lines

    def _create_grounded_summary(self, record: PatientRecordInput) -> PatientSummaryResponse:
        """Create an extractive summary without inventing facts or recommendations."""
        overview = (
            f"{record.patient_name or 'Patient'} (ID: {record.patient_id}), "
            f"age {record.age if record.age is not None else 'not recorded'}, "
            f"sex {record.gender or 'not recorded'}. "
            f"Summary compiled from {len(record.documents)} extracted report(s)."
        )
        lab_lines: list[str] = []
        medication_lines: list[str] = []
        allergy_lines: list[str] = []
        follow_up_lines: list[str] = []
        source_sections: list[str] = []

        for document in record.documents:
            lines = self._clean_lines(document.extracted_text)
            label = f"{document.document_type.title()} ({document.date or 'date not recorded'})"
            excerpts = lines[:20]
            source_sections.append(
                f"#### {label}\n" + ("\n".join(f"- {line}" for line in excerpts)
                                      if excerpts else "- No readable text extracted.")
            )
            if document.document_type.lower() in {"lab", "laboratory", "pathology"}:
                lab_lines.extend(lines[:20])
            medication_lines.extend(
                line for line in lines
                if re.search(r"\b(medication|medicine|tablet|capsule|dose|mg|mcg)\b", line, re.I)
            )
            allergy_lines.extend(
                line for line in lines if re.search(r"\b(allerg|nkda)\b", line, re.I)
            )
            follow_up_lines.extend(
                line for line in lines
                if re.search(r"\b(follow[- ]?up|review|return|recheck)\b", line, re.I)
            )

        mri_summary = None
        if record.mri_findings:
            mri = record.mri_findings
            parts = [f"Segmentation present: {'yes' if mri.tumor_detected else 'no'}"]
            if mri.location:
                parts.append(f"location: {mri.location}")
            if mri.volume_cm3 is not None:
                parts.append(f"volume: {mri.volume_cm3:.3f} cm³")
            if mri.additional_notes:
                parts.append(mri.additional_notes)
            mri_summary = "; ".join(parts)

        markdown = "### Grounded Patient Summary\n\n" + overview
        if source_sections:
            markdown += "\n\n### Extracted source information\n\n" + "\n\n".join(source_sections)
        else:
            markdown += "\n\n_No extracted reports are available for this patient._"
        if mri_summary:
            markdown += f"\n\n### Structured MRI result\n\n{mri_summary}"
        markdown += (
            "\n\n> This extractive summary contains only stored patient details and text "
            "recovered from uploaded records. It does not add diagnoses or medical advice."
        )

        return PatientSummaryResponse(
            patient_id=record.patient_id,
            patient_overview=overview,
            medical_history=[],
            diagnoses=[],
            current_conditions=[],
            mri_findings_summary=mri_summary,
            laboratory_observations=list(dict.fromkeys(lab_lines))[:20],
            current_medications=list(dict.fromkeys(medication_lines))[:20],
            allergies=list(dict.fromkeys(allergy_lines))[:20],
            recommended_follow_up=list(dict.fromkeys(follow_up_lines))[:20],
            narrative_summary_markdown=markdown,
        )

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
