import re
from typing import Any


class RadiologyService:
    """Structure radiology report text without adding clinical interpretation."""

    FIELD_LABELS = {
        "date": "Report date",
        "referring veterinary surgeon": "Referring clinician",
        "referring clinician": "Referring clinician",
        "hospital": "Hospital",
        "email address": "Email",
        "patient name and surname": "Patient name",
        "patient name": "Patient name",
        "species (canine/feline)": "Species",
        "species": "Species",
        "breed": "Breed",
        "age": "Age",
        "sex": "Sex",
        "body areas scanned and charged": "Body areas scanned",
        "body areas scanned": "Body areas scanned",
        "service required": "Service",
    }
    FOOTER_MARKERS = (
        "is a trading name of",
        "registered office",
        "all copyrights in the name",
        "company number",
    )

    def overview(self, text: str) -> dict[str, Any]:
        lines = self._clean_lines(text)
        metadata: dict[str, str] = {}
        for line in lines:
            if ":" not in line:
                continue
            label, value = (part.strip() for part in line.split(":", 1))
            display_label = self.FIELD_LABELS.get(label.lower())
            if display_label and value:
                metadata[display_label] = value

        history = self._section(
            lines,
            ("relevant clinical history",),
            ("report", "main findings", "findings", "conclusion"),
        )
        findings_lines = self._section(
            lines,
            ("main findings", "imaging findings", "findings"),
            ("conclusion", "impression", "recommendations"),
        )
        findings = self._bullets(findings_lines)
        conclusion_lines = self._section(
            lines,
            ("conclusion & recommendations", "conclusion and recommendations",
             "conclusion", "impression"),
            ("best regards", "signed", "image 1", "figure 1"),
        )
        conclusion, recommendations = self._split_recommendations(conclusion_lines)
        image_references = [line for line in lines
                            if re.match(r"^(?:image|figure)\s+\d+", line, re.IGNORECASE)]

        return {
            "title": self._report_title(lines),
            "metadata": metadata,
            "clinical_history": self._paragraph(history),
            "findings": findings,
            "conclusion": conclusion,
            "recommendations": recommendations,
            "image_references": image_references,
            "disclaimer": (
                "Structured directly from the uploaded radiology report. Verify all details "
                "against the original document and obtain qualified clinical review."
            ),
        }

    def as_text(self, overview: dict[str, Any]) -> str:
        sections = [overview["title"]]
        if overview["metadata"]:
            sections.append("REPORT DETAILS\n" + "\n".join(
                f"{key}: {value}" for key, value in overview["metadata"].items()
            ))
        if overview["clinical_history"]:
            sections.append("CLINICAL HISTORY\n" + overview["clinical_history"])
        if overview["findings"]:
            sections.append("MAIN FINDINGS\n" + "\n".join(
                f"- {finding}" for finding in overview["findings"]
            ))
        if overview["conclusion"]:
            sections.append("CONCLUSION\n" + overview["conclusion"])
        if overview["recommendations"]:
            sections.append("RECOMMENDATIONS\n" + "\n".join(
                f"- {item}" for item in overview["recommendations"]
            ))
        if overview["image_references"]:
            sections.append("IMAGE REFERENCES\n" + "\n".join(
                f"- {item}" for item in overview["image_references"]
            ))
        sections.append(overview["disclaimer"])
        return "\n\n".join(sections)

    def _clean_lines(self, text: str) -> list[str]:
        output: list[str] = []
        for raw in text.replace("\r", "").splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            if not line or any(marker in line.lower() for marker in self.FOOTER_MARKERS):
                continue
            if output and line == output[-1]:
                continue
            output.append(line)
        return output

    @staticmethod
    def _is_heading(line: str, headings: tuple[str, ...]) -> bool:
        normalized = line.lower().strip(" :&-")
        return any(normalized == heading or normalized.startswith(heading)
                   for heading in headings)

    def _section(self, lines: list[str], starts: tuple[str, ...],
                 ends: tuple[str, ...]) -> list[str]:
        start = next((index + 1 for index, line in enumerate(lines)
                      if self._is_heading(line, starts)), None)
        if start is None:
            return []
        end = next((index for index in range(start, len(lines))
                    if self._is_heading(lines[index], ends)), len(lines))
        return lines[start:end]

    @staticmethod
    def _bullets(lines: list[str]) -> list[str]:
        if not lines:
            return []
        findings: list[str] = []
        current: list[str] = []
        saw_marker = False
        for line in lines:
            if line in {"-", "•"} or line.startswith(("- ", "• ")):
                saw_marker = True
                if current:
                    findings.append(" ".join(current).strip())
                current = [line.lstrip("-• ").strip()] if len(line) > 1 else []
            elif saw_marker:
                current.append(line)
            else:
                current.append(line)
        if current:
            findings.append(" ".join(current).strip())
        return [item for item in findings if item]

    def _split_recommendations(self, lines: list[str]) -> tuple[str, list[str]]:
        text = self._paragraph(lines)
        if not text:
            return "", []
        marker = re.search(
            r"\b(?:I would suggest|I recommend|Recommend|Recommendation|Surgery\b|"
            r"Please (?:follow|let))",
            text,
            re.IGNORECASE,
        )
        if not marker:
            return text, []
        conclusion = text[:marker.start()].strip()
        recommendation_text = text[marker.start():].strip()
        recommendations = [part.strip() for part in re.split(r"(?<=[.!?])\s+", recommendation_text)
                           if part.strip()]
        return conclusion, recommendations

    @staticmethod
    def _paragraph(lines: list[str]) -> str:
        return " ".join(lines).strip()

    @staticmethod
    def _report_title(lines: list[str]) -> str:
        return next((line for line in lines if "radiology report" in line.lower()
                     or "teleneurology report" in line.lower()), "Radiology Report")
