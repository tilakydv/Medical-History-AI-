import re
from typing import Any


class DocumentStructureService:
    """Build a general readable structure for any extracted clinical document."""

    PATHOLOGY_TERMS = ("pathology", "specimen", "gross description", "microscopic")
    RADIOLOGY_TERMS = ("radiology", "imaging findings", "main findings", "mri", "ct scan")
    METADATA_LABELS = {
        "case", "case no", "case number", "date", "document date", "report date",
        "patient", "patient name", "date of birth", "dob", "age", "sex", "gender",
        "mrn", "medical record number", "physician", "attending physician",
        "consultant", "medical consultant", "hospital", "facility", "department",
    }
    NON_SECTION_LABELS = METADATA_LABELS | {"print name", "signature", "page"}
    ENUMERATED_HEADING = re.compile(
        r"^(?P<marker>(?:\d{1,3}|[IVXLCDM]{1,7}|[A-Z]))[.)]\s+"
        r"(?P<title>[^:]{2,100}):\s*(?P<content>.*)$"
    )
    STANDALONE_ENUMERATOR = re.compile(r"^(?:\d{1,3}|[IVXLCDM]{1,7}|[A-Z])[.)]$")

    def overview(self, text: str, laboratory_value_count: int = 0) -> dict[str, Any]:
        lines = self._lines(text)
        metadata = self._metadata(lines)
        sections, unsectioned = self._sections(lines)
        lowered = text.lower()
        detected: list[str] = []
        if laboratory_value_count:
            detected.append("laboratory")
        if sum(term in lowered for term in self.PATHOLOGY_TERMS) >= 2:
            detected.append("pathology")
        if sum(term in lowered for term in self.RADIOLOGY_TERMS) >= 2:
            detected.append("radiology")
        return {
            "title": lines[0] if lines else "Extracted Clinical Report",
            "metadata": metadata,
            "sections": sections,
            "unsectioned_text": " ".join(unsectioned).strip(),
            "detected_content": detected,
            "statistics": {
                "characters": len(text),
                "lines": len(lines),
                "structured_sections": len(sections),
                "laboratory_values": laboratory_value_count,
            },
        }

    def _metadata(self, lines: list[str]) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for index, line in enumerate(lines):
            if ":" not in line:
                continue
            label, value = (part.strip() for part in line.split(":", 1))
            normalized_label = re.sub(r"\s+", " ", label.lower().rstrip("."))
            if normalized_label not in self.METADATA_LABELS:
                continue
            if not value and index + 1 < len(lines):
                candidate = lines[index + 1]
                # Metadata values are often uppercase identifiers (for example MD-42),
                # so do not reject them merely because they resemble a heading.
                if not (self.ENUMERATED_HEADING.match(candidate)
                        or self.STANDALONE_ENUMERATOR.fullmatch(candidate)):
                    value = candidate
            if value and len(value) <= 200:
                metadata[label.title()] = value
        return metadata

    def _sections(self, lines: list[str]) -> tuple[dict[str, str], list[str]]:
        sections: dict[str, str] = {}
        unsectioned: list[str] = []
        heading: str | None = None
        content: list[str] = []
        enumerated_mode = any(self.ENUMERATED_HEADING.match(line) for line in lines)
        signature_footer = False
        for line in lines[1:]:
            if signature_footer:
                continue
            if heading and self._is_page_artifact(line):
                continue
            if heading and content and self._starts_signature_footer(line):
                signature_footer = True
                continue
            parsed_heading = self._heading(line)
            if parsed_heading and enumerated_mode and not self.ENUMERATED_HEADING.match(line):
                parsed_heading = None
            if parsed_heading:
                if heading and content:
                    sections[heading] = " ".join(content).strip()
                elif content:
                    unsectioned.extend(content)
                heading, initial_content = parsed_heading
                content = [initial_content] if initial_content else []
            elif heading:
                content.append(line)
            else:
                unsectioned.append(line)
        if heading and content:
            sections[heading] = " ".join(content).strip()
        elif content:
            unsectioned.extend(content)
        return sections, unsectioned

    @staticmethod
    def _is_page_artifact(line: str) -> bool:
        date_value = (
            r"(?:[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{1,4}[-/]\d{1,2}[-/]\d{1,4})"
        )
        return bool(
            re.fullmatch(r"(?:page\s*)?\d+\s*(?:of\s*\d+)?", line, re.IGNORECASE)
            or re.match(
                r"^(?:case|report|document|record)\s*(?:no|number|id)?[.:#-]?\s+\S.{0,60}$",
                line,
                re.IGNORECASE,
            )
            or re.fullmatch(rf"(?:document|report)?\s*date\s*[:.-]?\s*{date_value}",
                            line, re.IGNORECASE)
        )

    @staticmethod
    def _starts_signature_footer(line: str) -> bool:
        return bool(
            re.fullmatch(
                r"(?:M\.?D\.?|D\.?O\.?|R\.?N\.?|N\.?P\.?|P\.?A\.?(?:-C)?|"
                r"M\.?B\.?B\.?S\.?|Ph\.?D\.?|D\.?V\.?M\.?)",
                line,
                re.IGNORECASE,
            )
            or re.fullmatch(r"_+", line)
            or re.fullmatch(
                r"(?:print(?:ed)?\s+name|signature|signed|electronically signed)(?:\s+by)?[: ]*",
                line,
                re.IGNORECASE,
            )
        )

    @classmethod
    def _heading(cls, line: str) -> tuple[str, str] | None:
        stripped = line.strip()
        enumerated = cls.ENUMERATED_HEADING.match(stripped)
        if enumerated:
            marker = enumerated.group("marker")
            title = enumerated.group("title").strip().title()
            return f"{marker}. {title}", enumerated.group("content").strip()

        words = stripped.rstrip(":").split()
        if not 1 <= len(words) <= 10 or len(stripped) > 90:
            return None
        normalized = re.sub(r"\s+", " ", stripped.rstrip(":").lower().rstrip("."))
        if normalized in cls.NON_SECTION_LABELS:
            return None
        if re.match(r"^[A-Z][a-z]+\s+\d{1,2},\s+\d{4}:$", stripped):
            return None
        letters = [char for char in stripped if char.isalpha()]
        if stripped.endswith(":") or (letters and all(char.isupper() for char in letters)):
            return stripped.rstrip(":").strip().title(), ""
        return None

    @classmethod
    def _is_heading(cls, line: str) -> bool:
        return cls._heading(line) is not None

    @staticmethod
    def _lines(text: str) -> list[str]:
        raw_lines = [re.sub(r"\s+", " ", line).strip()
                     for line in text.replace("\r", "").splitlines() if line.strip()]
        lines: list[str] = []
        index = 0
        while index < len(raw_lines):
            line = raw_lines[index]
            if DocumentStructureService.STANDALONE_ENUMERATOR.fullmatch(line) and index + 1 < len(raw_lines):
                line = f"{line} {raw_lines[index + 1]}"
                index += 1
            lines.append(line)
            index += 1
        return lines
