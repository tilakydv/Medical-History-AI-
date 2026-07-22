"""
Feature 1: AI Patient Summarization Module.
Receives extracted patient information (JSON/text) and structured MRI findings,
generating a concise doctor-friendly summary strictly grounded in records.
"""

import json
import re
from collections import defaultdict
from typing import Any, Optional
from app.services.discharge_service import DischargeService
from app.services.document_service import DocumentStructureService
from app.services.lab_service import LaboratoryService
from app.services.pathology_service import PathologyService
from app.services.prescription_service import PrescriptionService
from app.services.radiology_service import RadiologyService
from app.services.timeline_service import TimelineService
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
            return self._apply_grounding_guards(model_result, patient_record)

        # Fallback if raw JSON structure didn't parse perfectly
        parsed_dict = JSONOutputParser.parse_json(response_text)
        if isinstance(parsed_dict, dict):
            parsed_dict["patient_id"] = patient_record.patient_id
            return self._apply_grounding_guards(
                PatientSummaryResponse.model_validate(parsed_dict), patient_record
            )

        # Fallback default
        return self._create_fallback_summary(patient_record)

    def _apply_grounding_guards(
        self, result: PatientSummaryResponse, record: PatientRecordInput,
    ) -> PatientSummaryResponse:
        """Apply deterministic safety, chronology, and traceability fields to LLM output."""
        grounded = self._create_grounded_summary(record)
        result.patient_overview = grounded.patient_overview
        result.medical_history = grounded.medical_history
        result.diagnoses = grounded.diagnoses
        result.current_conditions = grounded.current_conditions
        result.mri_findings_summary = grounded.mri_findings_summary
        result.laboratory_observations = grounded.laboratory_observations
        result.current_medications = grounded.current_medications
        result.allergies = grounded.allergies
        result.recommended_follow_up = grounded.recommended_follow_up
        result.demographic_conflicts = grounded.demographic_conflicts
        result.chronological_timeline = grounded.chronological_timeline
        result.report_type_findings = grounded.report_type_findings
        result.extraction_warnings = grounded.extraction_warnings
        result.narrative_summary_markdown = grounded.narrative_summary_markdown
        return result

    @staticmethod
    def _clean_lines(text: str) -> list[str]:
        lines: list[str] = []
        for raw in text.splitlines():
            line = PatientSummarizer._clean_text(raw).strip(" -\t")
            if len(line) >= 3 and line not in lines:
                lines.append(line)
        return lines

    def _create_grounded_summary(self, record: PatientRecordInput) -> PatientSummaryResponse:
        """Create a concise report-type-aware summary without inventing facts."""
        overview = (
            f"{record.patient_name or 'Patient'} (ID: {record.patient_id}), "
            f"age {record.age if record.age is not None else 'not recorded'}, "
            f"sex {record.gender or 'not recorded'}. "
            f"Summary compiled from {len(record.documents)} extracted report(s)."
        )
        medical_history: list[str] = []
        diagnoses: list[str] = []
        current_conditions: list[str] = []
        lab_lines: list[str] = []
        medication_lines: list[str] = []
        allergy_lines: list[str] = []
        follow_up_lines: list[str] = []
        document_demographics: list[str] = []
        demographic_conflicts: list[str] = []
        report_type_findings: dict[str, list[str]] = defaultdict(list)
        timeline_by_date: dict[str, list[str]] = defaultdict(list)
        extraction_warnings = [
            "Automated extraction can preserve spelling or OCR errors from uploaded files; verify unusual wording against the original report.",
            "Medication entries remain separated by report date and status; do not interpret the combined list as one current treatment plan.",
        ]

        for document in record.documents:
            lines = self._clean_lines(document.extracted_text)
            joined_text = " ".join(lines)
            report_type = document.document_type.lower()
            report_label = self._report_label(report_type)
            report_date = self._document_date(document.date, document.extracted_text)
            before = {
                "history": len(medical_history), "diagnoses": len(diagnoses),
                "conditions": len(current_conditions), "labs": len(lab_lines),
                "medications": len(medication_lines), "follow_up": len(follow_up_lines),
                "allergies": len(allergy_lines),
            }
            doc_age, doc_sex = self._extract_demographics(joined_text)
            if doc_age is not None or doc_sex:
                display = " / ".join(value for value in (
                    f"{doc_age} years" if doc_age is not None else "",
                    doc_sex or "",
                ) if value)
                document_demographics.append(display)
                if record.age is not None and doc_age is not None and record.age != doc_age:
                    demographic_conflicts.append(
                        f"Registered age is {record.age}; {report_label} records age {doc_age}."
                    )
                if record.gender and doc_sex and not self._same_sex(record.gender, doc_sex):
                    demographic_conflicts.append(
                        f"Registered sex is {record.gender}; {report_label} records sex {doc_sex}."
                    )
            if report_type in {"laboratory", "lab"}:
                parsed = LaboratoryService().parse(document.extracted_text)
                abnormal = [value for value in parsed if value.flag in {"H", "L", "A"}]
                selected = abnormal or parsed[:8]
                lab_lines.extend(self._format_lab(value) for value in selected)
            elif report_type == "prescription":
                prescription = PrescriptionService().overview(document.extracted_text)
                report_date = self._document_date(
                    document.date, document.extracted_text, prescription.get("details")
                )
                medication_lines.extend(
                    self._tag_point(self._medication_summary(item), "Prescribed", report_date)
                    for item in prescription["medications"]
                )
                follow_up_lines.extend(prescription.get("precautions", []))
            elif report_type == "radiology":
                radiology = RadiologyService().overview(document.extracted_text)
                report_date = self._document_date(
                    document.date, document.extracted_text, radiology.get("metadata")
                )
                if radiology.get("conclusion"):
                    diagnoses.extend(self._compact_points(radiology["conclusion"], 3))
                current_conditions.extend(radiology.get("findings", [])[:5])
                follow_up_lines.extend(radiology.get("recommendations", [])[:4])
            elif report_type == "pathology":
                pathology = PathologyService().overview(document.extracted_text)
                report_date = self._document_date(
                    document.date, document.extracted_text, pathology.get("details")
                )
                for heading, text in pathology.get("sections", {}).items():
                    if re.search(r"diagnos|summary", heading, re.I):
                        diagnoses.extend(self._compact_points(text, 4))
            elif report_type == "discharge":
                discharge = DischargeService().overview(document.extracted_text)
                report_date = self._document_date(
                    document.date, document.extracted_text, discharge.get("details")
                )
                for heading, text in discharge.get("sections", {}).items():
                    if "diagnos" in heading.lower():
                        diagnoses.extend(self._compact_points(text, 4))
                    elif "course" in heading.lower():
                        current_conditions.extend(self._compact_points(text, 3))
                    elif "follow" in heading.lower() or "instruction" in heading.lower():
                        follow_up_lines.extend(self._compact_points(text, 6))
                medication_lines.extend(
                    self._tag_point(self._medication_summary(item), "Discharge medication", report_date)
                    for item in discharge.get("medications", [])
                )
            else:
                structure = DocumentStructureService().overview(document.extracted_text)
                for heading, text in structure.get("sections", {}).items():
                    normalized = heading.lower()
                    if re.search(r"past medical|past surgical|medical, surgical|social history", normalized):
                        medical_history.extend(self._compact_points(text, 5))
                    elif re.search(r"diagnos|assessment|impression", normalized):
                        diagnoses.extend(self._compact_points(text, 5))
                    elif re.search(r"chief complaint|present illness|review of systems|physical", normalized):
                        current_conditions.extend(self._compact_points(text, 5))
                    elif re.search(r"follow|plan|management|education", normalized):
                        follow_up_lines.extend(self._compact_points(text, 6))

            allergy_lines.extend(
                self._compact_points(line, 2)[0]
                for line in lines if re.search(r"\b(?:allerg\w*|nkda)\b", line, re.I)
                and self._compact_points(line, 2)
            )

            contributed = (
                diagnoses[before["diagnoses"]:] + current_conditions[before["conditions"]:] +
                lab_lines[before["labs"]:] + medication_lines[before["medications"]:] +
                follow_up_lines[before["follow_up"]:] + medical_history[before["history"]:]
            )
            report_type_findings[report_label].extend(
                self._strip_trace(point) for point in contributed if point
            )
            timeline_content = DocumentStructureService().overview(document.extracted_text)
            added_timeline = False
            for row in TimelineService().build(timeline_content):
                timeline_points = self._clean_timeline_points(row["important_points"])
                if timeline_points:
                    timeline_by_date[row["date"]].extend(timeline_points)
                    added_timeline = True
            if report_date and not added_timeline:
                fallback_points = self._unique(
                    [self._strip_trace(point) for point in contributed], 2
                )
                if fallback_points:
                    timeline_by_date[report_date].extend(fallback_points)
            trace_label = report_label
            for values, key in (
                (medical_history, "history"), (diagnoses, "diagnoses"),
                (current_conditions, "conditions"), (lab_lines, "labs"),
                (follow_up_lines, "follow_up"),
            ):
                values[before[key]:] = [
                    self._tag_point(point, trace_label, report_date)
                    for point in values[before[key]:]
                ]

            # Preserve concise clinically meaningful lines when a sparse or unusual report
            # does not match a known structure (for example a one-line lab report).
            if not any((medical_history, diagnoses, current_conditions, lab_lines,
                        medication_lines, allergy_lines, follow_up_lines)):
                current_conditions.extend(self._meaningful_lines(lines, 6))

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

        medical_history = self._unique([
            point for point in medical_history
            if not re.match(r"^(?:allerg|no known .*allerg)", self._strip_trace(point), re.IGNORECASE)
        ], 4)
        diagnoses = self._unique([
            point for point in diagnoses
            if not re.match(r"^(?:order|collect|obtain|recommend)\b",
                            self._strip_trace(point), re.IGNORECASE)
        ], 7)
        current_conditions = self._unique([
            point for point in current_conditions
            if not re.match(r"^(?:negative for|denies|no evidence of)\b",
                            self._strip_trace(point), re.IGNORECASE)
        ], 5)
        lab_lines = self._unique(lab_lines, 7)
        medication_lines = self._dedupe_medications(medication_lines, 8)
        allergy_lines = self._normalize_allergies(allergy_lines)
        follow_up_lines = self._unique(follow_up_lines, 6)
        document_demographics = self._merge_demographics(document_demographics)
        demographic_conflicts = self._unique(demographic_conflicts, 8)
        report_type_findings = {
            label: self._unique(points, 3)
            for label, points in report_type_findings.items() if points
        }
        chronological_timeline = [{
            "date": date,
            "important_points": self._unique(points, 5),
        } for date, points in sorted(timeline_by_date.items()) if points]
        if document_demographics:
            overview += " Uploaded report demographics: " + ", ".join(document_demographics) + "."
        if demographic_conflicts:
            overview += " Demographic conflict detected; review the warnings below."

        markdown = "### Grounded Patient Summary\n\n" + overview
        summary_sections = [
            ("Demographic conflicts requiring verification", demographic_conflicts),
            ("Medical history", medical_history),
            ("Diagnoses and clinical impressions", diagnoses),
            ("Current conditions and important findings", current_conditions),
            ("Laboratory observations", lab_lines),
            ("Documented medications by report date and status", medication_lines),
            ("Allergies", allergy_lines),
            ("Documented follow-up and precautions", follow_up_lines),
        ]
        for heading, points in summary_sections:
            if points:
                markdown += f"\n\n### {heading}\n\n" + "\n".join(
                    f"- {point}" for point in points
                )
        if report_type_findings:
            markdown += "\n\n### Key findings by report type\n"
            for label, points in report_type_findings.items():
                markdown += f"\n#### {label}\n" + "\n".join(
                    f"- {point}" for point in points
                )
        markdown += "\n\n### Extraction and verification warnings\n" + "\n".join(
            f"- {warning}" for warning in extraction_warnings
        )
        if not record.documents:
            markdown += "\n\n_No extracted reports are available for this patient._"
        if mri_summary:
            markdown += f"\n\n### Structured MRI result\n\n- {mri_summary}"
        markdown += (
            "\n\n> This extractive summary contains only stored patient details and text "
            "recovered from uploaded records. It does not add diagnoses or medical advice."
        )

        return PatientSummaryResponse(
            patient_id=record.patient_id,
            patient_overview=overview,
            medical_history=medical_history,
            diagnoses=diagnoses,
            current_conditions=current_conditions,
            mri_findings_summary=mri_summary,
            laboratory_observations=lab_lines,
            current_medications=medication_lines,
            allergies=allergy_lines,
            recommended_follow_up=follow_up_lines,
            demographic_conflicts=demographic_conflicts,
            chronological_timeline=chronological_timeline,
            report_type_findings=report_type_findings,
            extraction_warnings=extraction_warnings,
            narrative_summary_markdown=markdown,
        )

    @staticmethod
    def _unique(values: list[str], maximum: int) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            clean = PatientSummarizer._clean_text(value).strip(" -")
            key = clean.casefold()
            if clean and key not in seen:
                output.append(clean)
                seen.add(key)
            if len(output) >= maximum:
                break
        return output

    @staticmethod
    def _report_label(report_type: str) -> str:
        labels = {
            "lab": "Laboratory", "laboratory": "Laboratory",
            "clinical": "Clinical", "prescription": "Prescription",
            "radiology": "Radiology", "pathology": "Pathology",
            "discharge": "Discharge",
        }
        return labels.get(report_type.lower(), report_type.replace("_", " ").title() or "Report")

    @staticmethod
    def _clean_text(value: Any) -> str:
        text = str(value)
        replacements = {
            "\ufffd": "", "â€¢": "-", "â—¦": "-", "â€“": "-", "â€”": "-",
            "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u00d7": "x",
            "\u00b5": "u", "\u03bc": "u", "\u00b0": " degrees ",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        text = re.sub(r":(?=\S)", ": ", text)
        text = text.replace(".;", ".")
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _clean_timeline_points(cls, points: list[str]) -> list[str]:
        administrative = re.compile(
            r"\b(?:phone|fax|mrn|patient name|accession|diplomate|license|"
            r"report finalized|confidential|medical center parkway)\b",
            re.IGNORECASE,
        )
        output = []
        for point in points:
            clean = cls._clean_text(point)
            if len(clean) < 12 or administrative.search(clean):
                continue
            output.append(clean)
        return cls._unique(output, 5)

    @classmethod
    def _merge_demographics(cls, values: list[str]) -> list[str]:
        unique = cls._unique(values, 8)
        detailed = [value for value in unique if "/" in value]
        if not detailed:
            return unique[:4]
        ages = {match.group(1) for value in detailed
                if (match := re.search(r"\b(\d{1,3})\b", value))}
        return [value for value in unique
                if "/" in value or not any(re.search(rf"\b{age}\b", value) for age in ages)][:4]

    @staticmethod
    def _extract_demographics(text: str) -> tuple[int | None, str | None]:
        age = None
        sex = None
        age_match = re.search(
            r"\bAge(?:\s*/\s*Sex)?\s*:\s*(\d{1,3})(?:\s*(?:Yrs?|Years?))?",
            text, re.IGNORECASE,
        ) or re.search(r"\b(\d{1,3})[- ]year[- ]old\b", text, re.IGNORECASE)
        if age_match:
            age = int(age_match.group(1))
        combined = re.search(
            r"\bAge\s*/\s*Sex\s*:\s*\d{1,3}\s*(?:Yrs?|Years?)?\s*/\s*([A-Za-z-]+)",
            text, re.IGNORECASE,
        )
        sex_match = combined or re.search(
            r"\b(?:Sex|Gender)\s*:\s*(Male|Female|Non-Binary|M|F)\b",
            text, re.IGNORECASE,
        )
        if sex_match:
            sex = sex_match.group(1)
            sex = {"m": "Male", "f": "Female"}.get(sex.casefold(), sex.title())
        return age, sex

    @staticmethod
    def _same_sex(left: str, right: str) -> bool:
        aliases = {
            "m": "male", "male": "male", "f": "female", "female": "female",
            "nonbinary": "non-binary", "non-binary": "non-binary",
        }
        normalize = lambda value: aliases.get(
            re.sub(r"[^a-z-]", "", value.casefold()), value.casefold()
        )
        return normalize(left) == normalize(right)

    @staticmethod
    def _document_date(
        stored_date: str | None, text: str, details: dict[str, Any] | None = None,
    ) -> str | None:
        candidates = []
        if stored_date and str(stored_date).lower() not in {"none", "not recorded", "unknown"}:
            candidates.append(str(stored_date))
        if details:
            candidates.extend(
                str(value) for key, value in details.items()
                if "date" in str(key).lower() and value
            )
        date_match = TimelineService.DATE_PATTERN.search(text)
        if date_match:
            candidates.append(date_match.group(0))
        for value in candidates:
            match = TimelineService.DATE_PATTERN.search(value)
            date_value = match.group(0) if match else value[:10]
            parsed = TimelineService._date_key(date_value)
            if parsed.year != 9999:
                return parsed.strftime("%Y-%m-%d")
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_value):
                return date_value
        return None

    @staticmethod
    def _tag_point(point: str, label: str, date: str | None) -> str:
        context = f"{label} - {date}" if date else label
        return f"[{context}] {point}"

    @staticmethod
    def _strip_trace(point: str) -> str:
        return re.sub(r"^\[[^]]+\]\s*", "", point).strip()

    @classmethod
    def _dedupe_medications(cls, values: list[str], maximum: int) -> list[str]:
        """Remove only equivalent medication orders; retain differing doses with context."""
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            clean = cls._clean_text(value).strip(" -")
            clinical = cls._strip_trace(clean)
            identity = re.sub(r"[^a-z0-9]+", " ", clinical.casefold()).strip()
            if identity and identity not in seen:
                output.append(clean)
                seen.add(identity)
            if len(output) >= maximum:
                break
        return output

    @classmethod
    def _normalize_allergies(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        negative: list[str] = []
        for value in values:
            point = re.sub(r"^(?:allerg(?:y|ies)|drug allergies?)\s*:\s*", "", value,
                           flags=re.IGNORECASE).strip(" -")
            if re.search(r"\b(?:nkda|no known (?:drug )?allerg)", point, re.IGNORECASE):
                negative.append(point)
            elif point:
                cleaned.append(point)
        # A documented positive allergy supersedes broad NKDA-style statements.
        return cls._unique(cleaned if cleaned else negative, 5)

    @staticmethod
    def _compact_points(text: str, maximum: int) -> list[str]:
        clean = PatientSummarizer._clean_text(text)
        clean = re.sub(r"(?:(?<=:)|^)\s*\d+[.)]\s*", "", clean)
        clean = re.sub(r"\s+\d+[.)]\s+", "; ", clean)
        protected = clean.replace("Dr.", "Dr<dot>").replace("M.D.", "MD")
        sentences = re.split(r"(?<=[.!?])\s+", protected)
        return [item.replace("<dot>", ".").strip(" -")
                for item in sentences if len(item.strip()) >= 4][:maximum]

    @staticmethod
    def _meaningful_lines(lines: list[str], maximum: int) -> list[str]:
        excluded = re.compile(
            r"(?:hospital|clinic|laborator|phone|patient name|mrn|confidential|page \d|"
            r"report date|department|medical center)", re.IGNORECASE,
        )
        return [line for line in lines if len(line) >= 12 and not excluded.search(line)][:maximum]

    @staticmethod
    def _medication_summary(item: dict[str, str]) -> str:
        name = item.get("name") or item.get("medication") or "Medication"
        details = [item.get(key, "") for key in
                   ("dose", "directions", "route / frequency", "instructions")]
        suffix = "; ".join(value for value in details if value)
        return f"{name}: {suffix}" if suffix else name

    @staticmethod
    def _format_lab(value: object) -> str:
        name = getattr(value, "test_name", "Laboratory result")
        numeric = getattr(value, "value_numeric", None)
        text = getattr(value, "value_text", None)
        unit = getattr(value, "unit", None) or ""
        flag = getattr(value, "flag", None)
        result = text if text is not None else numeric
        status = {"H": "High", "L": "Low", "A": "Abnormal"}.get(flag, "Within range")
        return PatientSummarizer._clean_text(
            f"{name}: {result} {unit} ({status})"
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
