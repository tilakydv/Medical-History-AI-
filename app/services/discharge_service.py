import re
from typing import Any


class DischargeService:
    """Structure diagnoses, course, medications, and follow-up from discharge reports."""

    def overview(self, text: str) -> dict[str, Any]:
        lines = self._lines(text)
        details = self._details(lines)
        sections: dict[str, str] = {}
        current: str | None = None
        content: list[str] = []
        for line in lines:
            heading = re.match(r"^\d+[.)]\s+(.+?)(?::)?$", line)
            if heading and len(line) < 100:
                if current and content:
                    sections[current] = " ".join(content).strip()
                current = heading.group(1).rstrip(":").title()
                content = []
            elif current and re.match(r"^Dr\.\s", line, re.IGNORECASE):
                break
            elif current and not self._is_footer(line):
                content.append(line)
        if current and content:
            sections[current] = " ".join(content).strip()
        medications = self._medications(text)
        result = {
            "title": "Discharge Summary", "details": details, "sections": sections,
            "medications": medications,
            "disclaimer": "Verify diagnoses, medicines, and follow-up instructions against the original discharge summary.",
        }
        result["download_text"] = self.as_text(result)
        return result

    @staticmethod
    def as_text(overview: dict[str, Any]) -> str:
        parts = [overview["title"]]
        if overview["details"]:
            parts.append("ENCOUNTER DETAILS\n" + "\n".join(
                f"{key}: {value}" for key, value in overview["details"].items()))
        parts.extend(f"{heading.upper()}\n{content}"
                     for heading, content in overview["sections"].items()
                     if heading != "Discharge Medications & Outpatient Regimen")
        if overview.get("medications"):
            medication_lines = []
            for medication in overview["medications"]:
                medication_lines.append("\n".join(
                    f"{key.title()}: {value}" for key, value in medication.items()))
            parts.append("DISCHARGE MEDICATIONS & OUTPATIENT REGIMEN\n" +
                         "\n\n".join(medication_lines))
        parts.append(overview["disclaimer"])
        return "\n\n".join(parts)

    @staticmethod
    def _details(lines: list[str]) -> dict[str, str]:
        wanted = {"patient name", "mrn / patient id", "admission date", "discharge date",
                  "attending md", "service"}
        output: dict[str, str] = {}
        for line in lines[:35]:
            if ":" in line:
                label, value = [part.strip() for part in line.split(":", 1)]
                if label.lower() in wanted and value:
                    output[label.title()] = value
        joined = " ".join(lines[:35])
        dates = re.findall(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
            joined,
        )
        if dates:
            output.setdefault("Admission Date", dates[0])
        if len(dates) > 1:
            output.setdefault("Discharge Date", dates[1])
        return output

    @staticmethod
    def _medications(text: str) -> list[dict[str, str]]:
        """Read visually separated medication-table columns without drug-specific rules."""
        output: list[dict[str, str]] = []
        in_medications = False
        flattened_rows: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            normalized = re.sub(r"^\d+[.)]\s*", "", line).lower()
            if "discharge medication" in normalized or "medications & outpatient" in normalized:
                in_medications = True
                continue
            if in_medications and re.match(r"^\d+[.)]\s+", line):
                break
            if not in_medications:
                continue
            if all(term in normalized for term in ("medication", "dosage", "route", "instructions")):
                continue
            if line:
                flattened_rows.append(re.sub(r"\s+", " ", line))
            columns = [re.sub(r"\s+", " ", value).strip()
                       for value in re.split(r"\s{2,}", line) if value.strip()]
            if len(columns) >= 4 and not all(
                    heading in columns[index].lower() for index, heading in enumerate(
                        ("medication", "dosage", "route", "duration"))):
                output.append({
                    "name": columns[0], "dose": columns[1],
                    "route / frequency": columns[2],
                    "instructions": " ".join(columns[3:]),
                })
        if output:
            return output

        # OCR may collapse the visual column spacing. In that case, identify
        # general medication/dose boundaries and common instruction verbs;
        # no drug names, doses, or frequencies are enumerated here.
        fallback = re.compile(
            r"(?P<name>[A-Za-z][A-Za-z0-9 +\-]+?)\s+"
            r"(?P<dose>\d+(?:\.\d+)?\s*(?:mcg|mg|g|mL|units?)\s+\S+)\s+"
            r"(?P<route_frequency>.+?)\s+"
            r"(?P<instructions>(?:Complete|Continue|Take|Use|Apply|As needed|For|Until|"
            r"Do not|Stop)\b.+?)(?=(?:\s+[A-Za-z][A-Za-z0-9 +\-]+?\s+"
            r"\d+(?:\.\d+)?\s*(?:mcg|mg|g|mL|units?)\s+\S+\s+)|$)",
            re.IGNORECASE,
        )
        joined = " ".join(flattened_rows)
        for match in fallback.finditer(joined):
            output.append({
                "name": match.group("name").strip(),
                "dose": match.group("dose").strip(),
                "route / frequency": match.group("route_frequency").strip(),
                "instructions": match.group("instructions").strip(),
            })
        return output

    @staticmethod
    def _is_footer(line: str) -> bool:
        lower = line.lower()
        return "confidential medical record" in lower or re.fullmatch(r"page \d+ of \d+", lower) is not None

    @classmethod
    def _lines(cls, text: str) -> list[str]:
        output: list[str] = []
        for raw in text.splitlines():
            line = re.sub(r"(?:\s*[•◦]\s*)+", " ", raw)
            line = re.sub(r"\s+", " ", line).strip()
            if line and not cls._is_footer(line):
                output.append(line)
        return output
