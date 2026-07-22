import re


def dated_entries(text: str) -> list[str]:
    numeric_dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
    numbered_items = re.split(r"(?=(?:^|\s)\d+[.)]\s)", text)
    cleaned_items = [
        re.sub(
            r"\s+(?:respectfully submitted|electronically signed|signed by)\b.*$", "",
            re.sub(r"^\s*\d+[.)]\s*", "", item).strip(), flags=re.IGNORECASE,
        ).strip()
        for item in numbered_items if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", item)
    ]
    if len(numeric_dates) >= 2 and len(cleaned_items) >= 2:
        return cleaned_items
    date_start = (
        r"(?:(?:January|February|March|April|May|June|July|August|September|October|"
        r"November|December)\s+\d{1,2},\s+\d{4}|\d{1,4}[-/]\d{1,2}[-/]\d{1,4})\s*:"
    )
    starts = list(re.finditer(date_start, text, re.IGNORECASE))
    if len(starts) < 2:
        return []
    return [
        text[match.start(): starts[index + 1].start() if index + 1 < len(starts) else len(text)].strip()
        for index, match in enumerate(starts)
    ]


def section_summary_points(text: str, maximum: int = 4) -> list[str]:
    """Create concise extractive bullets without adding facts to the report."""
    clauses = [clause.strip(" -\t") for clause in
               re.split(r"(?<=[.!?;])\s+", re.sub(r"\s+", " ", text).strip())
               if clause.strip(" -\t")]
    if len(clauses) <= maximum:
        return clauses
    clinical_cues = re.compile(
        r"\b(?:assessment|diagnos|finding|result|conclusion|recommend|treat|plan|"
        r"harm|death|failed|failure|abnormal|normal|improved|worsen|follow-up|"
        r"medication|allerg|procedure|surgery)\w*\b", re.IGNORECASE,
    )
    selected = {0}
    ranked = sorted(range(1, len(clauses)), key=lambda index: (
        bool(clinical_cues.search(clauses[index])), bool(re.search(r"\d", clauses[index])), -index,
    ), reverse=True)
    selected.update(ranked[:maximum - 1])
    return [clauses[index] for index in sorted(selected)]
