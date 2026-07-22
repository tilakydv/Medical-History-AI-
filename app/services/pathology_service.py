import re
from typing import Any


class PathologyService:
    """Structure narrative pathology reports without inventing clinical content."""

    DETAIL_LABELS = {
        "patient": "Patient",
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
                next_label, _ = self._label_and_value(lines[index + 1])
                if next_label not in self.DETAIL_LABELS and next_label not in self.SECTION_LABELS:
                    value = lines[index + 1]
            if value:
                details[display] = value
        return details

    def _sections(self, lines: list[str]) -> dict[str, str]:
        sections: dict[str, str] = {}
        current: str | None = None
        content: list[str] = []
        for line in lines:
            label, inline_value = self._label_and_value(line)
            display = self.SECTION_LABELS.get(label)
            if display:
                if current:
                    sections[current] = " ".join(content).strip()
                current = display
                content = [inline_value] if inline_value else []
            elif current and label not in self.DETAIL_LABELS:
                content.append(line)
        if current:
            sections[current] = " ".join(content).strip()
        return {key: value for key, value in sections.items() if value}

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
            if line and (not output or line != output[-1]):
                output.append(line)
        return output

    @staticmethod
    def _label_and_value(line: str) -> tuple[str, str]:
        if ":" in line:
            label, value = line.split(":", 1)
            return label.strip().lower(), value.strip()
        return line.strip().lower(), ""
