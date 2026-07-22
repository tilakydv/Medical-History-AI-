import re
from datetime import datetime
from typing import Any


class TimelineService:
    """Create normalized, chronological table rows from structured report text."""

    DATE_PATTERN = re.compile(
        r"\b(?:(?:January|February|March|April|May|June|July|August|September|October|"
        r"November|December)\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        re.IGNORECASE,
    )
    CLINICAL_CUES = re.compile(
        r"\b(?:present|admit|discharg|symptom|diagnos|finding|result|scan|test|"
        r"treat|medication|surgery|procedure|ruptur|death|died|expired|harm|"
        r"recommend|follow-up|abnormal|normal|hemoglobin|hematocrit)\w*\b",
        re.IGNORECASE,
    )
    CRITICAL_CUES = re.compile(
        r"\b(?:aneurysm|ruptur|death|died|expired|surgery|operation|procedure|"
        r"diagnos|discharg|medication|hemorrhage|emergency)\w*\b",
        re.IGNORECASE,
    )
    OUTCOME_CUES = re.compile(
        r"\b(?:death|died|expired|outcome|demise)\w*\b", re.IGNORECASE
    )

    def build(self, content: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        sources = list(content.get("sections", {}).items())
        if content.get("unsectioned_text") and not sources:
            sources.append(("Report content", content["unsectioned_text"]))

        def add_event(date_text: str, segment: str, heading: str) -> None:
            segment = re.sub(r"^\s*\d+[.)]\s*", "", segment.strip())
            segment = re.sub(
                r"\s+(?:respectfully submitted|electronically signed|signed by)\b.*$",
                "", segment, flags=re.IGNORECASE,
            )
            for point in self._event_points(segment):
                identity = (self._date_key(date_text).isoformat(), point)
                if identity in seen:
                    continue
                seen.add(identity)
                rows.append({
                    "date": self._date_key(date_text).strftime("%Y-%m-%d"),
                    "event": point,
                    "_sort": self._date_key(date_text),
                    "_order": len(rows),
                })

        for heading, text in sources:
            date_matches = list(self.DATE_PATTERN.finditer(text))
            if len(date_matches) >= 2 and re.search(r"(?:^|\s)\d+[.)]\s", text):
                for item in re.split(r"(?=(?:^|\s)\d+[.)]\s)", text):
                    match = self.DATE_PATTERN.search(item)
                    if match:
                        add_event(match.group(0), item, heading)
                continue

            sentences: list[str] = []
            normalized_text = re.sub(r"\s+", " ", text)
            for sentence in re.split(r"(?<=[.!?])\s+", normalized_text):
                sentence = sentence.strip()
                if not sentence:
                    continue
                matches = list(self.DATE_PATTERN.finditer(sentence))
                if len(matches) <= 1:
                    sentences.append(sentence)
                    continue

                # PDF extraction often flattens a dated list onto one line, for
                # example "Oct 4: notes Feb 9: notes Feb 11: records". Treat
                # every date as a new event instead of assigning the whole line
                # to its first date.
                prefix = sentence[:matches[0].start()].strip()
                for index, match in enumerate(matches):
                    end = matches[index + 1].start() if index + 1 < len(matches) else len(sentence)
                    part = sentence[match.start():end].strip(" ,;-")
                    if index == 0 and prefix:
                        part = f"{prefix} {part}"
                    if part:
                        sentences.append(part)
            active_date: str | None = None
            active_sentences: list[str] = []
            for sentence in sentences:
                match = self.DATE_PATTERN.search(sentence)
                reference_only = bool(match and re.search(
                    r"(?:compared\s+(?:with|to)|since|previously|prior\s+to).{0,60}$",
                    sentence[max(0, match.start() - 90):match.start()], re.IGNORECASE,
                ))
                if match and not reference_only:
                    if active_date and active_sentences:
                        add_event(active_date, " ".join(active_sentences), heading)
                    active_date, active_sentences = match.group(0), [sentence]
                elif active_date:
                    active_sentences.append(sentence)
            if active_date and active_sentences:
                add_event(active_date, " ".join(active_sentences), heading)

        document_date = content.get("metadata", {}).get("Date")
        if document_date and self.DATE_PATTERN.fullmatch(document_date.strip()):
            add_event(document_date, "Report dated on this date.", "Document details")

        rows.sort(key=lambda row: (row["_sort"], row["_order"]))
        grouped: dict[str, list[str]] = {}
        for row in rows:
            grouped.setdefault(row["date"], []).append(row["event"])
        return [{"date": date, "important_points": points}
                for date, points in grouped.items()]

    def _event_points(self, segment: str, maximum: int = 5) -> list[str]:
        sentences = [sentence.strip(" -\t") for sentence in
                     re.split(r"(?<=[.!?;])\s+", re.sub(r"\s+", " ", segment))
                     if sentence.strip(" -\t")]
        if len(sentences) <= maximum:
            return sentences
        chosen = {0}
        ranked = sorted(range(1, len(sentences)), key=lambda index: (
            bool(self.OUTCOME_CUES.search(sentences[index])),
            bool(self.CRITICAL_CUES.search(sentences[index])),
            bool(self.CLINICAL_CUES.search(sentences[index])),
            bool(re.search(r"\d", sentences[index])), -index,
        ), reverse=True)
        chosen.update(ranked[:maximum - 1])
        return [sentences[index] for index in sorted(chosen)]

    @staticmethod
    def _date_key(value: str) -> datetime:
        for pattern in ("%B %d, %Y", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y"):
            try:
                return datetime.strptime(value.title(), pattern)
            except ValueError:
                continue
        return datetime.max
