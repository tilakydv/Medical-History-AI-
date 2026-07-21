import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import LaboratoryReport, LaboratoryValue


@dataclass
class ParsedLab:
    test_name: str
    value_numeric: float | None
    value_text: str | None
    unit: str | None
    reference_low: float | None
    reference_high: float | None
    flag: str | None


class LaboratoryService:
    # Conservative line parser. Uncertain prose is deliberately ignored.
    LINE = re.compile(
        r"^\s*(?P<name>[A-Za-z][A-Za-z0-9 /()_%.'-]{1,70}?)\s+"
        r"(?P<value>[<>]?\s*-?\d+(?:[.,]\d+)?|positive|negative|reactive|non-reactive)"
        r"(?:\s+(?P<unit>[A-Za-zµμ%/\^0-9.-]+))?"
        r"(?:\s+(?P<low>-?\d+(?:[.,]\d+)?)\s*[-–]\s*(?P<high>-?\d+(?:[.,]\d+)?))?"
        r"(?:\s+(?P<flag>H|L|HIGH|LOW|ABNORMAL))?\s*$", re.IGNORECASE)
    RANGE = re.compile(r"^\s*(?:(?P<op>[<>])\s*(?P<limit>\d+(?:[.,]\d+)?)|"
                       r"(?P<low>-?\d+(?:[.,]\d+)?)\s*[-–]\s*"
                       r"(?P<high>-?\d+(?:[.,]\d+)?))\s*$")
    VALUE = re.compile(r"^[<>]?\s*-?\d+(?:[.,]\d+)?$")
    SKIP_NAME = re.compile(
        r"method|serum|plasma|blood|urine|calculated|colorim|assay|cytometry|"
        r"electrode|enzym|visual|manual|test name|result|panel|examination|profile|"
        r"biochemistry|haematology|pathology|barcode|sample|date|page",
        re.IGNORECASE,
    )
    REFERENCE_LABELS = {"low", "high", "very high", "normal", "optimal", "near optimal",
                        "desirable", "borderline high", "borderline low"}

    def parse(self, text: str) -> list[ParsedLab]:
        values: list[ParsedLab] = []
        for raw in text.splitlines():
            match = self.LINE.match(raw)
            if not match:
                continue
            data = match.groupdict()
            raw_value = data["value"].replace(" ", "")
            numeric = self._number(raw_value.lstrip("<>"))
            low, high = self._number(data["low"]), self._number(data["high"])
            flag = data["flag"].upper() if data["flag"] else self._flag(numeric, low, high)
            values.append(ParsedLab(data["name"].strip(), numeric,
                                    None if numeric is not None else raw_value,
                                    data["unit"], low, high, flag))
        seen = {(item.test_name.lower(), item.value_numeric, item.unit) for item in values}
        for item in self._parse_vertical(text):
            key = (item.test_name.lower(), item.value_numeric, item.unit)
            if key not in seen:
                values.append(item)
                seen.add(key)
        return values

    def _parse_vertical(self, text: str) -> list[ParsedLab]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        output: list[ParsedLab] = []
        for index, line in enumerate(lines):
            reference = self.RANGE.match(line)
            if not reference or index < 3:
                continue
            value_index = index - 2
            if not self.VALUE.match(lines[value_index]):
                # Units occasionally wrap; tolerate one extra line between value and range.
                value_index = index - 3
                if value_index < 1 or not self.VALUE.match(lines[value_index]):
                    continue
            value = self._number(lines[value_index].replace(" ", "").lstrip("<>"))
            if value is None:
                continue
            unit_parts = lines[value_index + 1:index]
            unit = " ".join(unit_parts)[:40] or None
            if any(part.lower() in self.REFERENCE_LABELS for part in unit_parts):
                continue
            name = self._find_vertical_name(lines, value_index)
            if not name:
                continue
            low, high = self._reference_bounds(reference)
            output.append(ParsedLab(name, value, None, unit, low, high,
                                    self._flag(value, low, high)))
        return output

    def _find_vertical_name(self, lines: list[str], value_index: int) -> str | None:
        for candidate in reversed(lines[max(0, value_index - 5):value_index]):
            if (not self.SKIP_NAME.search(candidate) and not self.VALUE.match(candidate)
                    and not self.RANGE.match(candidate) and len(candidate) > 1
                    and candidate.lower() not in self.REFERENCE_LABELS):
                return candidate[:150]
        return None

    def _reference_bounds(self, match: re.Match[str]) -> tuple[float | None, float | None]:
        low, high = self._number(match.group("low")), self._number(match.group("high"))
        limit = self._number(match.group("limit"))
        if match.group("op") == "<":
            high = limit
        elif match.group("op") == ">":
            low = limit
        return low, high

    def overview(self, text: str) -> dict[str, Any]:
        values = self.parse(text)
        abnormal = [item for item in values if item.flag in {"H", "L"}]
        normal = [item for item in values if item.flag == "N"]
        lower = [item for item in abnormal if item.flag == "L"]
        higher = [item for item in abnormal if item.flag == "H"]
        summary_lines = [
            "### Automated laboratory overview",
            f"{len(values)} numeric results were extracted; {len(abnormal)} are outside the "
            "reference intervals printed by the laboratory.",
        ]
        if lower:
            summary_lines.append("**Below range:** " + ", ".join(
                f"{item.test_name} {item.value_numeric:g} {item.unit or ''}".strip()
                for item in lower
            ))
        if higher:
            summary_lines.append("**Above range:** " + ", ".join(
                f"{item.test_name} {item.value_numeric:g} {item.unit or ''}".strip()
                for item in higher
            ))
        summary_lines.append(
            "This is an extraction and range comparison, not a diagnosis. A clinician should "
            "interpret the findings with symptoms, history, and the original report."
        )
        return {
            "title": "Laboratory report overview",
            "summary_markdown": "\n\n".join(summary_lines),
            "disclaimer": "Automated extraction for clinical review; verify values against the source report.",
            "counts": {"values_extracted": len(values), "outside_reference_range": len(abnormal),
                       "within_reference_range": len(normal)},
            "key_findings": [self._display(item) for item in abnormal],
            "within_reference_range": [self._display(item) for item in normal],
        }

    @staticmethod
    def _display(item: ParsedLab) -> dict[str, Any]:
        reference = None
        if item.reference_low is not None and item.reference_high is not None:
            reference = f"{item.reference_low:g}–{item.reference_high:g}"
        elif item.reference_low is not None:
            reference = f">{item.reference_low:g}"
        elif item.reference_high is not None:
            reference = f"<{item.reference_high:g}"
        return {"test": item.test_name, "value": item.value_numeric,
                "unit": item.unit, "reference": reference, "flag": item.flag}

    def persist(self, db: Session, report_id: str, patient_id: str, text: str,
                observed_at: datetime | None = None) -> LaboratoryReport:
        old = db.scalar(select(LaboratoryReport).where(LaboratoryReport.report_id == report_id))
        if old:
            return old
        lab_report = LaboratoryReport(report_id=report_id, patient_id=patient_id,
                                      collected_at=observed_at)
        db.add(lab_report)
        db.flush()
        for item in self.parse(text):
            db.add(LaboratoryValue(lab_report_id=lab_report.id, patient_id=patient_id,
                                   observed_at=observed_at, **item.__dict__))
        db.commit()
        db.refresh(lab_report)
        return lab_report

    def trends(self, db: Session, patient_id: str) -> dict[str, list[dict[str, Any]]]:
        rows = db.scalars(select(LaboratoryValue).where(LaboratoryValue.patient_id == patient_id)
                          .order_by(LaboratoryValue.test_name, LaboratoryValue.observed_at)).all()
        output: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            output.setdefault(row.test_name, []).append({
                "value": row.value_numeric if row.value_numeric is not None else row.value_text,
                "unit": row.unit, "flag": row.flag,
                "observed_at": row.observed_at.isoformat() if row.observed_at else None,
            })
        return output

    @staticmethod
    def _number(value: str | None) -> float | None:
        if not value:
            return None
        try:
            normalized = value
            if "," in value and "." not in value and len(value.rsplit(",", 1)[1]) == 3:
                normalized = value.replace(",", "")
            else:
                normalized = value.replace(",", ".")
            return float(normalized)
        except ValueError:
            return None

    @staticmethod
    def _flag(value: float | None, low: float | None, high: float | None) -> str | None:
        if value is None:
            return None
        if low is not None and value < low:
            return "L"
        if high is not None and value > high:
            return "H"
        return "N" if low is not None or high is not None else None
