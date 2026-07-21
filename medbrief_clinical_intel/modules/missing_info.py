"""
Feature 6: Missing Information Detection Module.
Identifies clinically important patient information missing from current records
(e.g., prior MRI baseline, family history, smoking status, vitals, lab values).
"""

import json
from typing import Optional, List
from medbrief_clinical_intel.schemas.input_models import PatientRecordInput
from medbrief_clinical_intel.schemas.missing_info import MissingInfoChecklist, MissingInfoItem
from medbrief_clinical_intel.llm.client import QwenLLMClient
from medbrief_clinical_intel.llm.prompt_templates import SystemPrompts, ClinicalPromptBuilder
from medbrief_clinical_intel.llm.parser import JSONOutputParser


class MissingInfoDetector:
    """Module 6: Missing Information Detector."""

    def __init__(self, llm_client: Optional[QwenLLMClient] = None):
        self.llm_client = llm_client or QwenLLMClient()

    def detect_missing_information(
        self, patient_record: PatientRecordInput
    ) -> MissingInfoChecklist:
        """Evaluate records and construct checklist of missing clinical data."""
        context_str = self._format_context(patient_record)
        schema_json = json.dumps(MissingInfoChecklist.model_json_schema(), indent=2)

        prompt = ClinicalPromptBuilder.build_missing_info_prompt(context_str, schema_json)
        response_text = self.llm_client.generate(
            prompt, system_prompt=SystemPrompts.CLINICAL_INTEL_SYSTEM
        )

        model_result = JSONOutputParser.parse_to_model(response_text, MissingInfoChecklist)
        if not model_result:
            parsed_dict = JSONOutputParser.parse_json(response_text)
            if isinstance(parsed_dict, dict):
                model_result = MissingInfoChecklist.model_validate(parsed_dict)

        if model_result:
            model_result.patient_id = patient_record.patient_id
            return model_result

        # Programmatic fallback check for essential items
        return self._create_fallback_checklist(patient_record)

    def _format_context(self, record: PatientRecordInput) -> str:
        parts = [f"Patient ID: {record.patient_id}"]
        if record.mri_findings:
            parts.append(f"MRI Report Present: {record.mri_findings.tumor_detected}")
        for doc in record.documents:
            parts.append(f"Document ID: {doc.document_id}\n{doc.extracted_text}")
        return "\n".join(parts)

    def _create_fallback_checklist(self, record: PatientRecordInput) -> MissingInfoChecklist:
        combined_text = " ".join([doc.extracted_text.lower() for doc in record.documents])
        items = []

        if "prior mri" not in combined_text and "previous mri" not in combined_text:
            items.append(
                MissingInfoItem(
                    field_name="Previous Baseline MRI",
                    category="Imaging History",
                    clinical_significance="Required to compare tumor progression rate over time.",
                    status="Missing"
                )
            )

        if "family history" not in combined_text:
            items.append(
                MissingInfoItem(
                    field_name="Family History",
                    category="Social/Family History",
                    clinical_significance="Essential for hereditary risk assessment.",
                    status="Missing"
                )
            )

        if "smoking" not in combined_text and "tobacco" not in combined_text:
            items.append(
                MissingInfoItem(
                    field_name="Smoking History",
                    category="Social/Family History",
                    clinical_significance="Vascular risk factor evaluation.",
                    status="Missing"
                )
            )

        if "blood pressure" not in combined_text and "bp:" not in combined_text:
            items.append(
                MissingInfoItem(
                    field_name="Blood Pressure Vitals",
                    category="Vitals",
                    clinical_significance="Baseline vitals for treatment safety.",
                    status="Missing"
                )
            )

        if "hba1c" not in combined_text:
            items.append(
                MissingInfoItem(
                    field_name="Recent HbA1c",
                    category="Lab Parameters",
                    clinical_significance="Glycemic status check for diabetic management.",
                    status="Missing"
                )
            )

        return MissingInfoChecklist(
            patient_id=record.patient_id,
            missing_items_count=len(items),
            items=items,
            recommendations_text=f"Identified {len(items)} missing clinical record items."
        )


