import html
import io
import re

import fitz

try:
    from frontend.content_utils import dated_entries, section_summary_points
except ModuleNotFoundError:
    from content_utils import dated_entries, section_summary_points


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
    """Generate a readable, wrapped, multi-page clinical PDF."""
    document = fitz.open()
    page_width, page_height = 595.0, 842.0
    left_margin, right_margin = 48.0, 48.0
    top_margin, bottom_margin = 42.0, 48.0
    body_width = page_width - left_margin - right_margin
    body_font_size = 9.5
    line_height = 13.0
    body_start = top_margin + 42.0
    body_end = page_height - bottom_margin - 18.0

    logical_lines = (_safe_text(content).splitlines() or ["No content available."])
    rendered_lines: list[str] = []
    for logical_line in logical_lines:
        rendered_lines.extend(_wrap_line(logical_line.rstrip(), body_width, body_font_size))

    page = None
    y = body_start

    def new_page() -> fitz.Page:
        nonlocal y
        created = document.new_page(width=page_width, height=page_height)
        safe_title = _safe_text(title).upper()
        created.insert_textbox(
            fitz.Rect(left_margin, top_margin, page_width - right_margin, top_margin + 26),
            safe_title,
            fontsize=11,
            fontname="helv",
            color=(0.08, 0.18, 0.32),
        )
        created.draw_line(
            fitz.Point(left_margin, top_margin + 28),
            fitz.Point(page_width - right_margin, top_margin + 28),
            color=(0.1, 0.4, 0.8),
            width=1.2,
        )
        y = body_start
        return created

    for line in rendered_lines:
        if page is None or y + line_height > body_end:
            page = new_page()
        page.insert_text(
            fitz.Point(left_margin, y),
            line,
            fontsize=body_font_size,
            fontname="helv",
            color=(0.08, 0.08, 0.08),
        )
        y += line_height

    total_pages = document.page_count
    for index, pdf_page in enumerate(document, start=1):
        footer = f"MedBrief AI Clinical System | Page {index} of {total_pages}"
        pdf_page.insert_text(
            fitz.Point(left_margin, page_height - bottom_margin + 18),
            footer,
            fontsize=7.5,
            fontname="helv",
            color=(0.45, 0.45, 0.45),
        )

    output = document.tobytes(garbage=4, deflate=True)
    document.close()
    return output


def _html_text(value: object) -> str:
    return html.escape(_safe_text(str(value)))


def generate_structured_report_pdf(title: str, filename: str,
                                   content: dict[str, object]) -> bytes:
    """Generate a polished structured report with real tables and bullet lists."""
    statistics = content.get("statistics", {}) if isinstance(content, dict) else {}
    metadata = content.get("metadata", {}) if isinstance(content, dict) else {}
    timeline = content.get("timeline", []) if isinstance(content, dict) else []
    sections = content.get("sections", {}) if isinstance(content, dict) else {}

    parts = [
        '<div class="cover">',
        '<h1>Summarized Clinical Report</h1>',
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

    if sections:
        parts.append('<h2 class="section-start">Section summaries</h2>')
        for heading, section_text in sections.items():
            parts.append(f'<div class="section"><h3>{_html_text(heading)}</h3><ul>')
            points = dated_entries(section_text) or section_summary_points(section_text)
            for point in points:
                parts.append(f'<li>{_html_text(point)}</li>')
            parts.append('</ul></div>')
    elif content.get("unsectioned_text"):
        parts.extend([
            '<h2>Report summary</h2>', '<ul>',
            *[f'<li>{_html_text(point)}</li>' for point in
              section_summary_points(str(content["unsectioned_text"]))],
            '</ul>',
        ])

    parts.append(
        '<div class="disclaimer"><b>Clinical review notice:</b> This extractive summary '
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
