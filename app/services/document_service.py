import re
from typing import Any


class DocumentStructureService:
    """Build a general readable structure for any extracted clinical document."""

    PATHOLOGY_TERMS = ("pathology", "specimen", "gross description", "microscopic")
    RADIOLOGY_TERMS = ("radiology", "imaging findings", "main findings", "mri", "ct scan")
    METADATA_LABELS = {
        "case", "case no", "case number", "date", "document date", "report date",
        "patient", "patient name", "date of birth", "dob", "age", "sex", "gender",
        "age / sex", "mrn", "mrn / patient", "patient id", "mrn / patient id", "id",
        "medical record number", "physician", "attending physician", "attending md",
        "md", "encounter date",
        "consultant", "medical consultant", "hospital", "facility", "department",
    }
    NON_SECTION_LABELS = METADATA_LABELS | {"print name", "signature", "page"}
    METADATA_DISPLAY = {
        "id": "Patient ID", "patient id": "Patient ID",
        "mrn / patient": "MRN / Patient ID", "mrn / patient id": "MRN / Patient ID",
        "md": "Attending clinician", "attending md": "Attending clinician",
    }
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
            if ":" in line:
                label, value = (part.strip() for part in line.split(":", 1))
            else:
                label, value = line.strip(), ""
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
            # PDF layout extraction may join the following label onto the value.
            # Stop at another recognized metadata label instead of merging fields.
            labels = "|".join(re.escape(item) for item in sorted(
                self.METADATA_LABELS, key=len, reverse=True
            ))
            value = re.split(rf"\b(?:{labels})\s*:", value,
                             maxsplit=1, flags=re.IGNORECASE)[0].strip()
            if value and len(value) <= 200:
                metadata[self.METADATA_DISPLAY.get(normalized_label, label.title())] = value
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
                    if not self._metadata_only(content):
                        sections[heading] = self._clean_section_text(" ".join(content))
                elif content:
                    unsectioned.extend(content)
                heading, initial_content = parsed_heading
                content = [initial_content] if initial_content else []
            elif heading:
                content.append(line)
            else:
                unsectioned.append(line)
        if heading and content:
            if not self._metadata_only(content):
                sections[heading] = self._clean_section_text(" ".join(content))
        elif content:
            unsectioned.extend(content)
        return sections, unsectioned

    @staticmethod
    def _clean_section_text(text: str) -> str:
        """Remove page furniture that PDF extraction joins to clinical sentences."""
        text = re.sub(
            r"(?:\s+\d+[.)])?\s+Page\s+\d+\s+of\s+\d+\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(?:Dr\.?\s+)?[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,3},?\s+"
            r"(?:M\.?D\.?|D\.?O\.?|M\.?B\.?B\.?S\.?)\b.*?\b"
            r"(?:medical\s+registration|registration|licen[cs]e)\s+"
            r"(?:no|number)\s*[:#].*$",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", text).strip()

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

    @classmethod
    def _metadata_only(cls, lines: list[str]) -> bool:
        """Identify a report-header block composed of labels and short identifiers."""
        if not lines:
            return False
        joined = " ".join(lines)
        labels_found = sum(
            bool(re.search(rf"\b{re.escape(label)}\b", joined, re.IGNORECASE))
            for label in cls.METADATA_LABELS if len(label) > 2
        )
        clinical_cues = re.search(
            r"\b(?:symptom|diagnos|finding|history|examination|treatment|medication|"
            r"assessment|plan|procedure|result)\w*\b", joined, re.IGNORECASE,
        )
        return labels_found >= 2 and clinical_cues is None

    @staticmethod
    def _lines(text: str) -> list[str]:
        raw_lines: list[str] = []
        for source_line in text.replace("\r", "").splitlines():
            line = re.sub(r"\(cid:\d+\)", " ", source_line, flags=re.IGNORECASE)
            line = re.sub(
                r"(?:\s+\d+[.)]\s*[\u2022\u25a0\u25a1\u25aa\u25ab\u25e6]+)+\s*$",
                "", line,
            )
            line = re.sub(r"[\u2022\u25a0\u25a1\u25aa\u25ab\u25e6]", "", line)
            line = re.sub(
                r"\bCONFIDENTIAL\s+(?:MEDICAL\s+)?RECORD\b.*$", "", line,
                flags=re.IGNORECASE,
            )
            line = re.sub(r"(?:\s*\d+[.)])?\s*(?:[nN]\s*){2,}$", "", line)
            line = re.sub(r"[\uE000-\uF8FF]", "", line)
            line = re.sub(r"\s+", " ", line).strip(" \t•■")
            if line:
                raw_lines.append(line)
        combined_lines: list[str] = []
        raw_index = 0
        while raw_index < len(raw_lines):
            line = raw_lines[raw_index]
            if raw_index + 1 < len(raw_lines):
                combined = re.sub(r"\s+", " ", f"{line} {raw_lines[raw_index + 1]}").lower()
                if combined in DocumentStructureService.METADATA_LABELS:
                    line = f"{line} {raw_lines[raw_index + 1]}"
                    raw_index += 1
            combined_lines.append(line)
            raw_index += 1

        lines: list[str] = []
        index = 0
        while index < len(combined_lines):
            line = combined_lines[index]
            if (DocumentStructureService.STANDALONE_ENUMERATOR.fullmatch(line)
                    and index + 1 < len(combined_lines)):
                line = f"{line} {combined_lines[index + 1]}"
                index += 1
            lines.append(line)
            index += 1
        return lines
