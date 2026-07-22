import html
import io
import re

import fitz

def _safe_text(text: str) -> str:
    """Normalize punctuation supported consistently by the built-in PDF font."""
    return (text.replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2013", "-").replace("\u2014", "-")
            .replace("\u00a0", " "))


def _wrap_line(text: str, width: float, font_size: float) -> list[str]:
    if not text:
        return [""]
    if re.fullmatch(r"[=\-~_]{3,}", text.strip()):
        symbol = text.strip()[0]
        count = max(3, int(width / fitz.get_text_length(symbol, "helv", font_size)))
        return [symbol * count]

    prefix = "- " if text.startswith("- ") else ""
    words = text[len(prefix):].split()
    output: list[str] = []
    current = prefix
    continuation_prefix = "  " if prefix else ""
    for word in words:
        candidate = f"{current} {word}".strip() if current.strip() else f"{current}{word}"
        if fitz.get_text_length(candidate, "helv", font_size) <= width:
            current = candidate
            continue
        if current.strip():
            output.append(current)
            current = continuation_prefix
        # Split unusually long tokens instead of allowing horizontal clipping.
        while fitz.get_text_length(f"{current}{word}", "helv", font_size) > width:
            available = max(1, int(len(word) * width /
                                   fitz.get_text_length(word, "helv", font_size)))
            output.append(f"{current}{word[:available]}")
            word = word[available:]
            current = continuation_prefix
        current = f"{current}{word}"
    if current.strip():
        output.append(current)
    return output or [""]


def generate_pdf_bytes(title: str, content: str) -> bytes:
    """Render free-form or Markdown text with the shared structured PDF design."""
    try:
        from frontend.structured_pdf import generate_typed_report_pdf
    except ModuleNotFoundError:
        from structured_pdf import generate_typed_report_pdf

    details, sections = _structure_download_text(content)
    return generate_typed_report_pdf(
        title,
        "Generated from stored patient data",
        details=details,
        sections=sections,
        disclaimer="Verify generated details against the original stored reports.",
    )


def _structure_download_text(content: str) -> tuple[dict[str, str], dict[str, object]]:
    """Convert arbitrary report text into generic details and section blocks."""
    clean = _safe_text(str(content or "")).replace("â€¢", "-").replace("•", "-")
    lines = [re.sub(r"\s+", " ", line).strip() for line in clean.splitlines()]
    details: dict[str, str] = {}
    sections: dict[str, object] = {}
    current_heading = "Report content"
    current_lines: list[str] = []

    def unique_heading(value: str) -> str:
        heading = value.strip().strip("#*: ") or "Report content"
        candidate = heading
        number = 2
        while candidate in sections:
            candidate = f"{heading} ({number})"
            number += 1
        return candidate

    def flush() -> None:
        nonlocal current_lines
        meaningful = [item for item in current_lines if item]
        if not meaningful:
            current_lines = []
            return
        bullet_items = [item[2:].strip() for item in meaningful if item.startswith("- ")]
        non_bullets = [item for item in meaningful if not item.startswith("- ")]
        if bullet_items and not non_bullets:
            sections[unique_heading(current_heading)] = bullet_items
        else:
            sections[unique_heading(current_heading)] = "\n".join(meaningful)
        current_lines = []

    for line in lines:
        if not line or re.fullmatch(r"[=~_-]{3,}", line):
            continue
        markdown_heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        bold_label = re.match(r"^\*\*(.+?)\s*:\*\*\s*(.*)$", line)
        plain_detail = re.match(r"^([A-Za-z][A-Za-z0-9 /()_-]{1,45})\s*:\s*(.+)$", line)
        is_upper_heading = (
            len(line) <= 90 and any(char.isalpha() for char in line)
            and all(not char.isalpha() or char.isupper() for char in line)
        )
        if markdown_heading or is_upper_heading:
            flush()
            current_heading = (markdown_heading.group(1) if markdown_heading else line).strip()
            continue
        if bold_label:
            if current_heading != "Report content":
                value = bold_label.group(2).strip()
                current_lines.append(
                    f"- {bold_label.group(1).strip()}: {value}".rstrip(": ")
                )
                continue
            flush()
            current_heading = bold_label.group(1).strip()
            if bold_label.group(2).strip():
                current_lines.append(bold_label.group(2).strip())
            continue
        if plain_detail and not current_lines and current_heading == "Report content":
            label, value = plain_detail.groups()
            details[label.strip()] = value.strip()
            continue
        if re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", line):
            point = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line).strip()
            current_lines.append(f"- {point}")
        elif line.startswith(">"):
            current_lines.append(line.lstrip("> "))
        else:
            current_lines.append(line.replace("**", ""))
    flush()
    if not sections:
        sections["Report content"] = "No content available."
    return details, sections


def _html_text(value: object) -> str:
    return html.escape(_safe_text(str(value)))


def generate_structured_report_pdf(title: str, filename: str,
                                   content: dict[str, object]) -> bytes:
    """Generate a polished structured report with real tables and bullet lists."""
    try:
        from frontend.structured_pdf import generate_structured_report_pdf as renderer
    except ModuleNotFoundError:
        from structured_pdf import generate_structured_report_pdf as renderer
    return renderer(title, filename, content)

    # Legacy renderer retained below temporarily for backward-compatible source history.
    statistics = content.get("statistics", {}) if isinstance(content, dict) else {}
    metadata = content.get("metadata", {}) if isinstance(content, dict) else {}
    timeline = content.get("timeline", []) if isinstance(content, dict) else []

    parts = [
        '<div class="cover">',
        '<h1>Structured Clinical Report</h1>',
        f'<p class="subtitle">{_html_text(filename)}</p>',
        '</div>',
        '<h2>Report overview</h2>',
        '<table class="overview"><tr>',
        f'<td><b>{_html_text(statistics.get("lines", 0))}</b><br><span>Extracted lines</span></td>',
        f'<td><b>{_html_text(statistics.get("structured_sections", 0))}</b><br><span>Sections</span></td>',
        f'<td><b>{_html_text(statistics.get("laboratory_values", 0))}</b><br><span>Clinical values</span></td>',
        '</tr></table>',
    ]

    if metadata:
        parts.extend([
            '<h2>Document details</h2>',
            '<table class="details"><colgroup><col width="28%"><col width="72%"></colgroup>',
        ])
        for label, value in metadata.items():
            parts.append(
                f'<tr><th>{_html_text(label)}</th><td>{_html_text(value)}</td></tr>'
            )
        parts.append('</table>')

    if timeline:
        parts.extend([
            '<h2>Chronological timeline</h2>',
            '<p class="note">Important dated events, ordered from earliest to latest.</p>',
            '<table class="timeline timeline-header"><colgroup><col width="20%"><col width="80%">'
            '</colgroup><thead><tr><th>Date</th><th>Important points</th></tr></thead></table>',
        ])
        for date_row in timeline:
            points = date_row.get("important_points") or ["No details recorded."]
            parts.append(
                '<table class="timeline timeline-group"><colgroup><col width="20%">'
                '<col width="80%"></colgroup><tbody>'
            )
            for index, point in enumerate(points):
                parts.append('<tr>')
                if index == 0:
                    parts.append(
                        f'<td class="date" rowspan="{len(points)}">'
                        f'{_html_text(date_row.get("date", ""))}</td>'
                    )
                parts.append(f'<td class="point">&#8226;&nbsp; {_html_text(point)}</td></tr>')
            parts.append('</tbody></table>')

    parts.append(
        '<div class="disclaimer"><b>Clinical review notice:</b> These structured details '
        'must be verified against the original uploaded report.</div>'
    )
    css = """
        body { font-family: sans-serif; font-size: 9.5pt; color: #202936; line-height: 1.35; }
        h1 { font-size: 20pt; color: #17365d; margin: 0 0 4pt 0; font-weight: bold; }
        .subtitle { font-size: 10pt; color: #5f6b7a; margin: 0 0 14pt 0; }
        h2 { font-size: 14pt; color: #17365d; margin: 17pt 0 7pt 0; font-weight: bold;
             border-bottom: 1px solid #4f81bd; padding-bottom: 3pt; }
        h3 { font-size: 11.5pt; color: #244a73; margin: 10pt 0 4pt 0; font-weight: bold; }
        p { margin: 2pt 0 5pt 0; } .note { color: #5f6b7a; font-size: 8.5pt; }
        table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt 0; }
        th { background: #17365d; color: white; font-weight: bold; text-align: left; }
        th, td { border: 0.6px solid #aeb8c4; padding: 6pt; vertical-align: top; }
        .overview td { width: 33%; text-align: center; background: #eef3f8; }
        .overview b { font-size: 14pt; color: #17365d; }
        .overview span { font-size: 8pt; color: #5f6b7a; }
        .details th { background: #e7eef6; color: #17365d; }
        .timeline { margin: 0; table-layout: fixed; }
        .timeline-header { page-break-after: avoid; }
        .timeline-group { page-break-inside: avoid; }
        .timeline-group td { border-top: 0; }
        .date { font-weight: bold; background: #eef3f8; }
        ul { margin: 2pt 0 7pt 14pt; padding: 0; }
        li { margin: 0 0 4pt 0; } .section { margin-bottom: 6pt; }
        .section-start { page-break-before: always; }
        .disclaimer { margin-top: 16pt; padding: 8pt; background: #fff4d6;
                      border: 0.7px solid #d7b65d; font-size: 8.5pt; }
    """

    story = fitz.Story(html="".join(parts), user_css=css)
    page_width, page_height = 595.0, 842.0
    content_rect = fitz.Rect(42, 48, page_width - 42, page_height - 48)
    mediabox = fitz.Rect(0, 0, page_width, page_height)
    intermediate = io.BytesIO()
    writer = fitz.DocumentWriter(intermediate)

    def rect_function(_rect_number: int, _filled: fitz.Rect) -> tuple[fitz.Rect, fitz.Rect, None]:
        return mediabox, content_rect, None

    story.write(writer, rect_function)
    writer.close()
    document = fitz.open(stream=intermediate.getvalue(), filetype="pdf")

    total = document.page_count
    for index, page in enumerate(document, start=1):
        if index > 1:
            page.insert_text(fitz.Point(42, 30), _safe_text(title), fontsize=7.5,
                             fontname="helv", color=(0.35, 0.4, 0.48))
        page.draw_line(fitz.Point(42, page_height - 35),
                       fitz.Point(page_width - 42, page_height - 35),
                       color=(0.65, 0.7, 0.76), width=0.5)
        page.insert_text(fitz.Point(42, page_height - 22),
                         f"MedBrief AI Clinical System | Page {index} of {total}",
                         fontsize=7.5, fontname="helv", color=(0.4, 0.45, 0.52))
    output = document.tobytes(garbage=4, deflate=True)
    document.close()
    return output
