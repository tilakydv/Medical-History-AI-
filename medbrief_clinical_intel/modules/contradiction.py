"""
Feature 5: Clinical Contradiction Detection Module.
Compares multi-source or multi-date reports to identify conflicting clinical information
(e.g., allergy mismatch, medication mismatch, diagnosis date conflicts, surgical history conflicts).
"""

import json
from typing import Optional, List
from medbrief_clinical_intel.schemas.input_models import PatientRecordInput
from medbrief_clinical_intel.schemas.contradiction import ContradictionReport, ContradictionItem
from medbrief_clinical_intel.llm.client import QwenLLMClient
from medbrief_clinical_intel.llm.prompt_templates import SystemPrompts, ClinicalPromptBuilder
from medbrief_clinical_intel.llm.parser import JSONOutputParser


class ContradictionDetector:
    """Module 5: Clinical Contradiction Detector."""

    def __init__(self, llm_client: Optional[QwenLLMClient] = None):
        self.llm_client = llm_client or QwenLLMClient()

    def detect_contradictions(self, patient_record: PatientRecordInput) -> ContradictionReport:
        """Analyze records for cross-document clinical contradictions."""
        context_str = self._format_context(patient_record)
        schema_json = json.dumps(ContradictionReport.model_json_schema(), indent=2)

        prompt = ClinicalPromptBuilder.build_contradiction_prompt(context_str, schema_json)
        response_text = self.llm_client.generate(
            prompt, system_prompt=SystemPrompts.CLINICAL_INTEL_SYSTEM
        )

        model_result = JSONOutputParser.parse_to_model(response_text, ContradictionReport)
        if not model_result:
            parsed_dict = JSONOutputParser.parse_json(response_text)
            if isinstance(parsed_dict, dict):
                model_result = ContradictionReport.model_validate(parsed_dict)

        if model_result:
            model_result.patient_id = patient_record.patient_id
            return model_result

        # Also run deterministic programmatic checks on common patterns
        rule_based_conflicts = self._rule_based_check(patient_record)
        has_conflicts = len(rule_based_conflicts) > 0

        return ContradictionReport(
            patient_id=patient_record.patient_id,
            contradictions_found=has_conflicts,
            contradictions=rule_based_conflicts,
            summary_text="No contradictions detected across uploaded documents." if not has_conflicts else f"{len(rule_based_conflicts)} clinical contradiction(s) require clinician review."
        )

    def _format_context(self, record: PatientRecordInput) -> str:
        parts = [f"Patient ID: {record.patient_id}"]
        if record.mri_findings:
            parts.append(f"MRI Report | Location: {record.mri_findings.location}")
        for doc in record.documents:
            parts.append(f"Document ID: {doc.document_id} | Date: {doc.date or 'Unknown'}\n{doc.extracted_text}")
        return "\n".join(parts)

    def _rule_based_check(self, record: PatientRecordInput) -> List[ContradictionItem]:
        """Programmatic heuristic checks for common contradictions."""
        items = []
        doc_texts = [(doc.document_id, doc.extracted_text.lower()) for doc in record.documents]

        # Check NKDA vs Allergy documented
        nkda_docs = [doc_id for doc_id, txt in doc_texts if "nkda" in txt or "no known drug allergies" in txt]
        allergy_docs = [doc_id for doc_id, txt in doc_texts if "allergy:" in txt or "allergic to" in txt]

        if nkda_docs and allergy_docs:
            items.append(
                ContradictionItem(
                    category="Allergy Mismatch",
                    description="One document records No Known Drug Allergies (NKDA), while another documents a specific drug allergy.",
                    record_a_source=nkda_docs[0],
                    record_a_statement="Documented NKDA",
                    record_b_source=allergy_docs[0],
                    record_b_statement="Documented specific allergy",
                    severity="High",
                    requires_clinician_review=True
                )
            )

        return items


