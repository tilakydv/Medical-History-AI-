import re
from typing import Any


class PrescriptionService:
    """Extract medication orders without mixing repeated directions and quantities."""

    LABELS = ("sig / directions", "directions", "quantity", "refills")

    def overview(self, text: str) -> dict[str, Any]:
        lines = self._lines(text)
        details = self._details(lines)
        medications: list[dict[str, str]] = []
        precautions: list[str] = []
        index = 0
        while index < len(lines):
            match = re.match(r"^(\d+)\.\s+(.+)$", lines[index])
            if match and "page " not in lines[index].lower():
                medication: dict[str, str] = {"name": match.group(2).strip()}
                pending: list[str] = []
                index += 1
                while index < len(lines):
                    if re.match(r"^\d+\.\s+", lines[index]) or self._is_special(lines[index]):
                        break
                    label = self._label(lines[index])
                    if label:
                        key, value = label
                        index += 1
                        continuation: list[str] = [*pending]
                        pending = []
                        if value:
                            continuation.append(value)
                        while index < len(lines) and not self._label(lines[index]) \
                                and not re.match(r"^\d+\.\s+", lines[index]) \
                                and not self._is_special(lines[index]):
                            continuation.append(lines[index])
                            index += 1
                        medication[key] = " ".join(continuation).strip()
                        continue
                    pending.append(lines[index])
                    index += 1
                medications.append(medication)
                continue
            if self._is_special(lines[index]):
                index += 1
                while index < len(lines) and not self._is_footer(lines[index]):
                    point = lines[index].lstrip("•- ").strip()
                    if point:
                        precautions.append(point)
                    index += 1
                break
            index += 1

        result = {
            "title": "Medical Prescription",
            "details": details,
            "medications": medications,
            "precautions": precautions,
            "disclaimer": "Verify medication names, doses, and directions against the original prescription.",
        }
        result["download_text"] = self.as_text(result)
        return result

    def as_text(self, overview: dict[str, Any]) -> str:
        parts = [overview["title"]]
        if overview["details"]:
            parts.append("PRESCRIPTION DETAILS\n" + "\n".join(
                f"{key}: {value}" for key, value in overview["details"].items()))
        for number, medication in enumerate(overview["medications"], 1):
            parts.append(f"MEDICATION {number}\n" + "\n".join(
                f"{key.replace('_', ' ').title()}: {value}" for key, value in medication.items()))
        if overview["precautions"]:
            parts.append("SPECIAL INSTRUCTIONS AND PRECAUTIONS\n" + "\n".join(
                f"- {point}" for point in overview["precautions"]))
        parts.append(overview["disclaimer"])
        return "\n\n".join(parts)

    @staticmethod
    def _details(lines: list[str]) -> dict[str, str]:
        wanted = {
            "patient name": "Patient name", "mrn / patient id": "MRN / Patient ID",
            "age / sex": "Age / sex", "prescription date": "Prescription date",
            "prescriber": "Prescriber", "license no": "License number",
        }
        output: dict[str, str] = {}
        labels = "|".join(re.escape(label) for label in sorted(wanted, key=len, reverse=True))
        field_pattern = re.compile(rf"(?P<label>{labels})\s*:", re.IGNORECASE)
        for line in lines[:35]:
            matches = list(field_pattern.finditer(line))
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
                value = line[match.end():end].strip(" |,-")
                if value:
                    output[wanted[match.group("label").lower()]] = value
        joined = " ".join(lines[:35])
        date = re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b", joined)
        if date:
            output["Prescription date"] = date.group(0)
        mrn_candidates = re.findall(
            r"\b[A-Za-z0-9-]{2,20}\s*\|\s*[A-Za-z0-9-]{4,30}\b", joined)
        mrn = next((candidate for candidate in mrn_candidates
                    if re.search(r"\|\s*[A-Za-z0-9-]*\d", candidate)), None)
        if mrn:
            output["MRN / Patient ID"] = mrn
        # Column-based PDFs can append the value from a neighboring field to
        # the preceding value. Remove exact duplicate suffixes once the field
        # has also been recovered independently.
        for label, value in list(output.items()):
            for other_label, other_value in output.items():
                if label == other_label or not other_value or value == other_value:
                    continue
                if value.endswith(other_value):
                    shortened = value[:-len(other_value)].strip(" |,-")
                    if shortened:
                        output[label] = shortened
                        value = shortened
        return output

    @classmethod
    def _label(cls, line: str) -> tuple[str, str] | None:
        if ":" not in line:
            return None
        label, value = [part.strip() for part in line.split(":", 1)]
        if label.lower() not in cls.LABELS:
            return None
        key = "directions" if "direction" in label.lower() or label.lower() == "sig" else label.lower()
        return key, value

    @staticmethod
    def _is_special(line: str) -> bool:
        return "special instructions" in line.lower() or "precautions" in line.lower()

    @staticmethod
    def _is_footer(line: str) -> bool:
        lower = line.lower()
        return "confidential medical record" in lower or lower.startswith("dr.")

    @classmethod
    def _lines(cls, text: str) -> list[str]:
        return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()
                if line.strip() and not re.fullmatch(r"page \d+ of \d+", line.strip(), re.I)]
