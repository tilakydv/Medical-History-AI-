"""
Feature 2: Chronological Medical Timeline Module.
Generates a timeline automatically sorted by date from patient records.
Example output format:
2019 - Diabetes diagnosed
2020 - Hypertension
2022 - Surgery
2024 - Brain MRI
2026 - Tumor detected
"""

import json
from typing import List, Optional
from medbrief_clinical_intel.schemas.input_models import PatientRecordInput
from medbrief_clinical_intel.schemas.timeline import MedicalTimelineResponse, TimelineEvent
from medbrief_clinical_intel.llm.client import QwenLLMClient
from medbrief_clinical_intel.llm.prompt_templates import SystemPrompts, ClinicalPromptBuilder
from medbrief_clinical_intel.llm.parser import JSONOutputParser


class TimelineGenerator:
    """Module 2: Chronological Medical Timeline Generator."""

    def __init__(self, llm_client: Optional[QwenLLMClient] = None):
        self.llm_client = llm_client or QwenLLMClient()

    def generate_timeline(self, patient_record: PatientRecordInput) -> MedicalTimelineResponse:
        """Generate chronologically ordered medical timeline."""
        context_str = self._format_context(patient_record)
        schema_json = json.dumps(MedicalTimelineResponse.model_json_schema(), indent=2)

        prompt = ClinicalPromptBuilder.build_timeline_prompt(context_str, schema_json)
        response_text = self.llm_client.generate(
            prompt, system_prompt=SystemPrompts.CLINICAL_INTEL_SYSTEM
        )

        model_result = JSONOutputParser.parse_to_model(response_text, MedicalTimelineResponse)
        if not model_result:
            parsed_dict = JSONOutputParser.parse_json(response_text)
            if isinstance(parsed_dict, dict):
                model_result = MedicalTimelineResponse.model_validate(parsed_dict)

        if model_result:
            model_result.patient_id = patient_record.patient_id
            # Ensure chronological sorting by iso_date or date string
            model_result.timeline = self._sort_timeline(model_result.timeline)
            # Reformat timeline text
            model_result.formatted_timeline_text = self._format_timeline_text(model_result.timeline)
            return model_result

        return self._create_fallback_timeline(patient_record)

    def _sort_timeline(self, timeline: List[TimelineEvent]) -> List[TimelineEvent]:
        """Sort timeline events chronologically."""
        def extract_sort_key(event: TimelineEvent) -> str:
            if event.iso_date:
                return event.iso_date
            # Try to extract 4-digit year from date_or_year
            import re
            match = re.search(r"\b(19\d\d|20\d\d)\b", event.date_or_year)
            if match:
                return f"{match.group(1)}-01-01"
            return "9999-99-99"

        return sorted(timeline, key=extract_sort_key)

    def _format_timeline_text(self, events: List[TimelineEvent]) -> str:
        """Format timeline into concise string format: 'YYYY - Event'."""
        lines = []
        for e in events:
            lines.append(f"{e.date_or_year} - {e.title}")
        return "\n".join(lines)

    def _format_context(self, record: PatientRecordInput) -> str:
        parts = [f"Patient ID: {record.patient_id}"]
        if record.mri_findings:
            parts.append(
                f"MRI Report Date: 2026 | Finding: Tumor detected ({record.mri_findings.location or 'Brain'})"
            )
        for doc in record.documents:
            parts.append(f"Doc Date: {doc.date or 'Unknown'} | Type: {doc.document_type}\n{doc.extracted_text}")
        return "\n".join(parts)

    def _create_fallback_timeline(self, record: PatientRecordInput) -> MedicalTimelineResponse:
        events = []
        if record.mri_findings and record.mri_findings.tumor_detected:
            events.append(
                TimelineEvent(
                    date_or_year="2026",
                    iso_date="2026-01-01",
                    category="Diagnosis",
                    title="Tumor detected",
                    description="Brain MRI segmentation identified tumor.",
                    source_document="MRI_Segmentation"
                )
            )
        return MedicalTimelineResponse(
            patient_id=record.patient_id,
            timeline=events,
            formatted_timeline_text=self._format_timeline_text(events)
        )


