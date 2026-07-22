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
        "mrn / patient id": "MRN / Patient ID",
        "exam type": "Exam type",
        "accession no": "Accession number",
        "exam date": "Exam date",
        "ordering md": "Ordering clinician",
        "report finalized": "Report finalized",
        "time": "Finalized time",
    }
    FOOTER_MARKERS = (
        "is a trading name of",
        "registered office",
        "all copyrights in the name",
        "company number",
        "confidential medical record",
        "security stamp / signature",
    )

    def overview(self, text: str) -> dict[str, Any]:
        lines = self._clean_lines(text)
        metadata: dict[str, str] = {}
        labels = "|".join(re.escape(label) for label in sorted(
            self.FIELD_LABELS, key=len, reverse=True
        ))
        field_pattern = re.compile(rf"(?P<label>{labels})\s*:", re.IGNORECASE)
        for line in lines:
            matches = list(field_pattern.finditer(line))
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
                value = line[match.end():end].strip(" |,-")
                display_label = self.FIELD_LABELS.get(match.group("label").lower())
                if display_label and value:
                    metadata[display_label] = value
        self._recover_split_identifiers(lines, metadata)

        history = self._section(
            lines,
            ("relevant clinical history", "clinical indication & technique"),
            ("report", "main findings", "findings", "conclusion", "detailed radiographic findings"),
        )
        findings_lines = self._section(
            lines,
            ("main findings", "imaging findings", "findings", "detailed radiographic findings"),
            ("conclusion", "impression", "recommendations", "impression & diagnostic summary"),
        )
        findings = self._bullets(findings_lines)
        conclusion_lines = self._section(
            lines,
            ("conclusion & recommendations", "conclusion and recommendations",
             "conclusion", "impression", "impression & diagnostic summary"),
            ("best regards", "signed", "image 1", "figure 1", "confidential medical record"),
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

    @staticmethod
    def _recover_split_identifiers(lines: list[str], metadata: dict[str, str]) -> None:
        """Recover identifiers whose label and value were extracted in adjacent columns."""
        joined_header = " ".join(lines[:25])
        patient_identifiers = re.findall(
            r"\b[A-Za-z0-9-]{2,20}\s*\|\s*[A-Za-z0-9-]{4,30}\b", joined_header
        )
        identifier = next(
            (candidate for candidate in patient_identifiers
             if re.search(r"\d", candidate.split("|", 1)[1])),
            None,
        )
        if identifier:
            metadata.setdefault("MRN / Patient ID", identifier)
            if metadata.get("Patient name", "").endswith(identifier):
                metadata["Patient name"] = metadata["Patient name"][:-len(identifier)].strip()

        has_accession_label = any(
            re.fullmatch(r"(?:accession|accession\s+no|no)\s*:??", line, re.IGNORECASE)
            for line in lines[:25]
        )
        exam_type = metadata.get("Exam type", "")
        trailing_identifier = re.search(
            r"(?P<identifier>[A-Za-z]{1,12}(?:-[A-Za-z0-9]+)+)\s*$", exam_type
        )
        if has_accession_label and trailing_identifier:
            identifier = trailing_identifier.group("identifier")
            metadata.setdefault("Accession number", identifier)
            metadata["Exam type"] = exam_type[:trailing_identifier.start()].strip(" |,-")

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
            if re.fullmatch(r"(?:page\s*)?\d+\s+of\s+\d+", line, re.IGNORECASE):
                continue
            if not line or any(marker in line.lower() for marker in self.FOOTER_MARKERS):
                continue
            if output and line == output[-1]:
                continue
            output.append(line)
        return output

    @staticmethod
    def _is_heading(line: str, headings: tuple[str, ...]) -> bool:
        normalized = re.sub(r"^\d+[.)]\s*", "", line.lower()).strip(" :&-")
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
            if line.endswith(":") and len(line) < 80:
                if current:
                    findings.append(" ".join(current).strip())
                current = [line]
                saw_marker = True
            elif line in {"-", "•"} or line.startswith(("- ", "• ")):
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
        text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE)
        text = re.split(r"\bDr\.\s+[A-Z]", text, maxsplit=1)[0].strip()
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
        if any("teleneurology report" in line.lower() for line in lines):
            return "Teleneurology report"
        return "Radiology report"
