import re
from typing import Any


class PathologyService:
    """Structure narrative pathology reports without inventing clinical content."""

    DETAIL_LABELS = {
        "patient": "Patient",
        "patient name": "Patient",
        "mrn / patient id": "MRN / Patient ID",
        "mrn": "MRN / Patient ID",
        "accession no": "Accession number",
        "accession number": "Accession number",
        "specimen received": "Specimen received",
        "submitting md": "Submitting clinician",
        "report date": "Report date",
        "date of birth": "Date of birth",
        "medical record number": "Medical record number",
        "gender": "Gender",
        "specimen collection date/time": "Specimen collection date/time",
        "specimen receipt date/time": "Specimen receipt date/time",
        "case number": "Case number",
    }
    SECTION_LABELS = {
        "specimen label(s)": "Specimen",
        "patient history": "Patient history",
        "pre-operative diagnosis": "Pre-operative diagnosis",
        "preoperative diagnosis": "Pre-operative diagnosis",
        "diagnosis": "Diagnosis",
        "note": "Note",
        "special test results": "Special test results",
        "gross description": "Gross description",
        "microscopic description": "Microscopic description",
        "comment": "Comment",
        "clinical indication & specimen origin": "Clinical indication and specimen origin",
        "macroscopic / gross description": "Macroscopic / gross description",
        "microscopic examination & histological analysis": "Microscopic examination and histological analysis",
        "immunohistochemistry (ihc) & special stains": "Immunohistochemistry and special stains",
        "pathological summary & final diagnosis": "Pathological summary and final diagnosis",
    }

    def overview(self, text: str) -> dict[str, Any]:
        lines = self._clean_lines(text)
        details = self._details(lines)
        sections = self._sections(lines)
        title = next(
            (line for line in lines if "pathology report" in line.lower()),
            "Pathology Report",
        )
        overview = {
            "title": title,
            "details": details,
            "sections": sections,
            "disclaimer": (
                "Structured directly from the uploaded pathology report. Verify all details "
                "against the original document and obtain qualified clinical review."
            ),
        }
        overview["download_text"] = self.as_text(overview)
        return overview

    def _details(self, lines: list[str]) -> dict[str, str]:
        details: dict[str, str] = {}
        for index, line in enumerate(lines):
            label, inline_value = self._label_and_value(line)
            display = self.DETAIL_LABELS.get(label)
            if not display:
                continue
            value = inline_value
            if not value and index + 1 < len(lines):
                candidate = lines[index + 1]
                next_label, _ = self._label_and_value(candidate)
                if next_label not in self.SECTION_LABELS and (
                        ":" not in candidate or next_label not in self.DETAIL_LABELS):
                    value = candidate
            if value:
                details[display] = value
        return details

    def _sections(self, lines: list[str]) -> dict[str, str]:
        sections: dict[str, str] = {}
        current: str | None = None
        content: list[str] = []
        for line in lines:
            if current and re.match(r"^Dr\.\s", line, re.IGNORECASE):
                break
            label, inline_value = self._label_and_value(line)
            display = self.SECTION_LABELS.get(label)
            if display:
                if current:
                    sections[current] = self._join_content(content)
                current = display
                content = [inline_value] if inline_value else []
            elif current and label not in self.DETAIL_LABELS:
                content.append(line)
        if current:
            sections[current] = self._join_content(content)
        return {key: value for key, value in sections.items() if value}

    @staticmethod
    def _join_content(lines: list[str]) -> str:
        output = ""
        for line in lines:
            if line.startswith("- "):
                output = f"{output.rstrip()}\n{line}".strip()
            else:
                output = f"{output} {line}".strip()
        return output

    def as_text(self, overview: dict[str, Any]) -> str:
        output = [overview["title"]]
        if overview["details"]:
            output.append("REPORT DETAILS\n" + "\n".join(
                f"{key}: {value}" for key, value in overview["details"].items()
            ))
        for heading, content in overview["sections"].items():
            output.append(f"{heading.upper()}\n{content}")
        output.append(overview["disclaimer"])
        return "\n\n".join(output)

    @staticmethod
    def _clean_lines(text: str) -> list[str]:
        output: list[str] = []
        for raw in text.replace("\r", "").splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            line = re.sub(r"^[\ufffd•▪◦]\s*", "- ", line)
            line = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", "", line, flags=re.IGNORECASE).strip()
            if "confidential medical record" in line.lower():
                continue
            if line and (not output or line != output[-1]):
                output.append(line)
        return output

    @staticmethod
    def _label_and_value(line: str) -> tuple[str, str]:
        line = re.sub(r"^\s*\d+[.)]\s*", "", line.strip())
        if ":" in line:
            label, value = line.split(":", 1)
            return label.strip().lower(), value.strip()
        return line.strip().lower(), ""
