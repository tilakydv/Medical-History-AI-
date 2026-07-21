"""
Feature 4: Medication & Allergy Intelligence Module.
Extracts medication details (name, dosage, frequency, duration, status)
and allergies (drug, food, environmental) into structured JSON and narrative summary.
"""

import json
from typing import Optional
from medbrief_clinical_intel.schemas.input_models import PatientRecordInput
from medbrief_clinical_intel.schemas.medication import MedicationAllergyReport, MedicationItem, AllergyItem
from medbrief_clinical_intel.llm.client import QwenLLMClient
from medbrief_clinical_intel.llm.prompt_templates import SystemPrompts, ClinicalPromptBuilder
from medbrief_clinical_intel.llm.parser import JSONOutputParser


class MedicationIntelligence:
    """Module 4: Medication & Allergy Intelligence Extractor."""

    def __init__(self, llm_client: Optional[QwenLLMClient] = None):
        self.llm_client = llm_client or QwenLLMClient()

    def analyze_medications_and_allergies(
        self, patient_record: PatientRecordInput
    ) -> MedicationAllergyReport:
        """Extract and structure all medications and allergies."""
        context_str = self._format_context(patient_record)
        schema_json = json.dumps(MedicationAllergyReport.model_json_schema(), indent=2)

        prompt = ClinicalPromptBuilder.build_medication_allergy_prompt(context_str, schema_json)
        response_text = self.llm_client.generate(
            prompt, system_prompt=SystemPrompts.CLINICAL_INTEL_SYSTEM
        )

        model_result = JSONOutputParser.parse_to_model(response_text, MedicationAllergyReport)
        if not model_result:
            parsed_dict = JSONOutputParser.parse_json(response_text)
            if isinstance(parsed_dict, dict):
                model_result = MedicationAllergyReport.model_validate(parsed_dict)

        if model_result:
            model_result.patient_id = patient_record.patient_id
            return model_result

        return self._create_fallback_report(patient_record)

    def _format_context(self, record: PatientRecordInput) -> str:
        parts = [f"Patient ID: {record.patient_id}"]
        for doc in record.documents:
            parts.append(f"Doc ID: {doc.document_id} | Date: {doc.date or 'N/A'}\n{doc.extracted_text}")
        return "\n".join(parts)

    def _create_fallback_report(self, record: PatientRecordInput) -> MedicationAllergyReport:
        return MedicationAllergyReport(
            patient_id=record.patient_id,
            current_medications=[],
            previous_medications=[],
            drug_allergies=[],
            food_allergies=[],
            environmental_allergies=[],
            summary_text="No active medications or allergies recorded in uploaded text."
        )


