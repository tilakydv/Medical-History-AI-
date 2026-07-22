import html
import io
import re
import unicodedata
from typing import Any

import fitz
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (CondPageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)


NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#4F81BD")
LIGHT_BLUE = colors.HexColor("#EAF0F7")
GRID = colors.HexColor("#B6C0CB")
TEXT = colors.HexColor("#202936")
MUTED = colors.HexColor("#5F6B7A")
WARNING = colors.HexColor("#FFF4D6")


def _escape(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("×", "x").replace("µ", "u").replace("μ", "u")
    text = "".join(character for character in text
                   if unicodedata.category(character) not in {"Cc", "Co"}
                   and character != "\ufffd")
    return html.escape(text)


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.footer_notice = kwargs.pop("footer_notice", "")
        super().__init__(*args, **kwargs)
        self._saved_pages: list[dict[str, Any]] = []

    def showPage(self) -> None:  # noqa: N802 - ReportLab API
        self._saved_pages.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total = len(self._saved_pages)
        for state in self._saved_pages:
            self.__dict__.update(state)
            self.setStrokeColor(GRID)
            self.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
            self.setFont("Helvetica", 7.5)
            self.setFillColor(MUTED)
            if self.footer_notice:
                self.setFont("Helvetica", 6.5)
                self.drawCentredString(
                    A4[0] / 2, 14.3 * mm,
                    "Clinical review notice: Verify extracted details against the original report.",
                )
            self.drawString(18 * mm, 9.5 * mm, "MedBrief AI Clinical System")
            self.drawRightString(A4[0] - 18 * mm, 9.5 * mm,
                                 f"Page {self._pageNumber} of {total}")
            super().showPage()
        super().save()


def generate_structured_report_pdf(title: str, filename: str,
                                   content: dict[str, Any]) -> bytes:
    """Create a styled multi-page report with a true row-spanned timeline table."""
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=20 * mm, title=title, author="MedBrief AI",
    )
    sample = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("ReportTitle", parent=sample["Title"], fontName="Helvetica-Bold",
                                fontSize=20, leading=24, textColor=NAVY, alignment=0,
                                spaceAfter=3),
        "subtitle": ParagraphStyle("Subtitle", parent=sample["Normal"], fontSize=9.5,
                                   leading=12, textColor=MUTED, spaceAfter=12),
        "h2": ParagraphStyle("H2", parent=sample["Heading2"], fontName="Helvetica-Bold",
                             fontSize=14, leading=17, textColor=NAVY, spaceBefore=12,
                             spaceAfter=7, borderColor=BLUE, borderWidth=0,
                             borderPadding=(0, 0, 3, 0)),
        "h3": ParagraphStyle("H3", parent=sample["Heading3"], fontName="Helvetica-Bold",
                             fontSize=11.5, leading=14, textColor=colors.HexColor("#244A73"),
                             spaceBefore=7, spaceAfter=4),
        "body": ParagraphStyle("Body", parent=sample["BodyText"], fontSize=9.2,
                               leading=12.3, textColor=TEXT),
        "small": ParagraphStyle("Small", parent=sample["BodyText"], fontSize=8.2,
                                leading=10.5, textColor=MUTED),
        "cell": ParagraphStyle("Cell", parent=sample["BodyText"], fontSize=8.6,
                               leading=11.2, textColor=TEXT),
        "cell_bold": ParagraphStyle("CellBold", parent=sample["BodyText"],
                                    fontName="Helvetica-Bold", fontSize=8.7,
                                    leading=11.2, textColor=TEXT),
        "table_header": ParagraphStyle("TableHeader", parent=sample["BodyText"],
                                       fontName="Helvetica-Bold", fontSize=8.7,
                                       leading=11.2, textColor=colors.white),
        "metric": ParagraphStyle("Metric", parent=sample["BodyText"], fontName="Helvetica-Bold",
                                 fontSize=14, leading=17, textColor=NAVY, alignment=TA_CENTER),
        "metric_label": ParagraphStyle("MetricLabel", parent=sample["BodyText"], fontSize=7.5,
                                       leading=9, textColor=MUTED, alignment=TA_CENTER),
    }
    flow: list[Any] = [
        Paragraph("Structured Clinical Report", styles["title"]),
        Paragraph(_escape(filename), styles["subtitle"]),
        Paragraph("Report overview", styles["h2"]),
    ]

    stats = content.get("statistics", {})
    overview = Table([[
        [Paragraph(_escape(stats.get("lines", 0)), styles["metric"]),
         Paragraph("Extracted lines", styles["metric_label"])],
        [Paragraph(_escape(stats.get("structured_sections", 0)), styles["metric"]),
         Paragraph("Sections", styles["metric_label"])],
        [Paragraph(_escape(stats.get("laboratory_values", 0)), styles["metric"]),
         Paragraph("Clinical values", styles["metric_label"])],
    ]], colWidths=[document.width / 3] * 3)
    overview.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE), ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, GRID), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    flow.extend([overview, Spacer(1, 4)])

    metadata = content.get("metadata", {})
    if metadata:
        flow.append(Paragraph("Document details", styles["h2"]))
        detail_rows = [[Paragraph(f"<b>{_escape(label)}</b>", styles["cell"]),
                        Paragraph(_escape(value), styles["cell"])]
                       for label, value in metadata.items()]
        details = Table(detail_rows, colWidths=[document.width * 0.27, document.width * 0.73])
        details.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
            ("BOX", (0, 0), (-1, -1), 0.6, GRID), ("INNERGRID", (0, 0), (-1, -1), 0.45, GRID),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        flow.append(details)

    timeline = content.get("timeline", [])
    if timeline:
        flow.extend([
            Paragraph("Chronological timeline", styles["h2"]),
            Paragraph("Important dated events, ordered from earliest to latest.", styles["small"]),
            Spacer(1, 4),
        ])
        timeline_rows: list[list[Any]] = [[
            Paragraph("Date", styles["table_header"]),
            Paragraph("Important points", styles["table_header"]),
        ]]
        spans: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
        for date_row in timeline:
            start = len(timeline_rows)
            points = date_row.get("important_points") or ["No details recorded."]
            for index, point in enumerate(points):
                timeline_rows.append([
                    Paragraph(_escape(date_row["date"]), styles["cell_bold"]) if index == 0 else "",
                    Paragraph(f"&bull;&nbsp; {_escape(point)}", styles["cell"]),
                ])
            if len(points) > 1:
                spans.append(("SPAN", (0, start), (0, len(timeline_rows) - 1)))
        timeline_table = Table(timeline_rows, colWidths=[document.width * 0.18,
                                                         document.width * 0.82],
                               repeatRows=1, splitByRow=1)
        timeline_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 1), (0, -1), LIGHT_BLUE),
            ("BOX", (0, 0), (-1, -1), 0.7, GRID), ("INNERGRID", (0, 0), (-1, -1), 0.45, GRID),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6), *spans,
        ]))
        flow.append(timeline_table)

    sections = content.get("sections", {})
    if sections:
        flow.append(Paragraph("Clinical report details", styles["h2"]))
        for heading, section_text in sections.items():
            if not section_text:
                continue
            sentences = re.split(r"(?<=[.!?])\s+", str(section_text).strip())
            chunks: list[str] = []
            current = ""
            for sentence in sentences:
                candidates = ([sentence[index:index + 700]
                               for index in range(0, len(sentence), 700)]
                              if len(sentence) > 700 else [sentence])
                for candidate in candidates:
                    if current and len(current) + len(candidate) + 1 > 700:
                        chunks.append(current)
                        current = candidate
                    else:
                        current = f"{current} {candidate}".strip()
            if current:
                chunks.append(current)
            section_rows = [
                [Paragraph(_escape(heading), styles["cell_bold"]) if index == 0 else "",
                 Paragraph(_escape(chunk), styles["body"])]
                for index, chunk in enumerate(chunks)
            ]
            section_table = Table(
                section_rows,
                colWidths=[document.width * 0.25, document.width * 0.75],
                splitByRow=1,
                splitInRow=1,
            )
            section_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            flow.extend([section_table, Spacer(1, 5)])

    document.build(
        flow,
        canvasmaker=lambda *args, **kwargs: NumberedCanvas(
            *args, footer_notice="clinical", **kwargs
        ),
    )
    return output.getvalue()


def generate_typed_report_pdf(
        title: str,
        filename: str,
        details: dict[str, Any] | None = None,
        sections: dict[str, Any] | None = None,
        tables: list[dict[str, Any]] | None = None,
        disclaimer: str = "",
) -> bytes:
    """Render any structured report using the shared clinical-report design system."""
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=20 * mm, title=title, author="MedBrief AI",
    )
    sample = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TypedTitle", parent=sample["Title"], fontName="Helvetica-Bold",
        fontSize=20, leading=24, textColor=NAVY, alignment=0, spaceAfter=3,
    )
    subtitle = ParagraphStyle(
        "TypedSubtitle", parent=sample["Normal"], fontSize=9.5, leading=12,
        textColor=MUTED, spaceAfter=12,
    )
    h2 = ParagraphStyle(
        "TypedH2", parent=sample["Heading2"], fontName="Helvetica-Bold",
        fontSize=14, leading=17, textColor=NAVY, spaceBefore=12, spaceAfter=7,
    )
    body = ParagraphStyle(
        "TypedBody", parent=sample["BodyText"], fontSize=9, leading=12,
        textColor=TEXT,
    )
    label = ParagraphStyle(
        "TypedLabel", parent=body, fontName="Helvetica-Bold", textColor=TEXT,
    )
    header = ParagraphStyle(
        "TypedHeader", parent=body, fontName="Helvetica-Bold", textColor=colors.white,
    )

    def paragraph(value: Any, style: ParagraphStyle = body) -> Paragraph:
        if isinstance(value, (list, tuple)):
            rendered = "<br/>".join(f"&bull;&nbsp; {_escape(item)}" for item in value if item)
        else:
            lines = [line.strip() for line in str(value).splitlines() if line.strip()]
            rendered = "<br/>".join(
                f"&bull;&nbsp; {_escape(line[2:])}" if line.startswith("- ")
                else _escape(line)
                for line in lines
            )
        return Paragraph(rendered or "Not recorded", style)

    def style_table(table: Table, header_rows: int = 0) -> None:
        commands: list[tuple[Any, ...]] = [
            ("BOX", (0, 0), (-1, -1), 0.6, GRID),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
        if header_rows:
            commands.append(("BACKGROUND", (0, 0), (-1, header_rows - 1), NAVY))
        table.setStyle(TableStyle(commands))

    flow: list[Any] = [
        Paragraph(_escape(title), title_style),
        Paragraph(_escape(filename), subtitle),
    ]
    if details:
        flow.append(Paragraph("Report details", h2))
        rows = [[paragraph(key, label), paragraph(value)] for key, value in details.items()]
        detail_table = Table(rows, colWidths=[document.width * 0.28, document.width * 0.72])
        style_table(detail_table)
        detail_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE)]))
        flow.append(detail_table)

    for heading, content in (sections or {}).items():
        if content in (None, "", [], {}):
            continue
        flow.append(Paragraph(_escape(heading), h2))
        if isinstance(content, (list, tuple)):
            section_rows = [[paragraph([item])] for item in content if item]
        else:
            text = str(content)
            paragraphs = [item.strip() for item in text.splitlines() if item.strip()]
            section_rows = [[paragraph(item)] for item in (paragraphs or [text])]
        section_table = Table(
            section_rows,
            colWidths=[document.width],
            splitByRow=1,
            splitInRow=1,
        )
        style_table(section_table)
        flow.append(section_table)

    for table_spec in tables or []:
        columns = table_spec.get("columns") or []
        rows = table_spec.get("rows") or []
        if not columns or not rows:
            continue
        flow.extend([
            CondPageBreak(55 * mm),
            Paragraph(_escape(table_spec.get("title", "Details")), h2),
        ])
        table_rows = [[paragraph(column.get("label", column.get("key", "")), header)
                       for column in columns]]
        for row in rows:
            table_rows.append([paragraph(row.get(column.get("key", ""), ""))
                               for column in columns])
        widths = table_spec.get("widths")
        col_widths = ([document.width * float(width) for width in widths]
                      if widths and len(widths) == len(columns) else
                      [document.width / len(columns)] * len(columns))
        data_table = Table(table_rows, colWidths=col_widths, repeatRows=1, splitByRow=1)
        style_table(data_table, header_rows=1)
        flow.append(data_table)

    document.build(
        flow,
        canvasmaker=lambda *args, **kwargs: NumberedCanvas(
            *args, footer_notice=disclaimer, **kwargs
        ),
    )
    return output.getvalue()


def generate_laboratory_report_pdf(filename: str, overview: dict[str, Any]) -> bytes:
    """Create a complete laboratory table instead of a prose-only lab download."""
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=14 * mm, leftMargin=14 * mm,
        topMargin=12 * mm, bottomMargin=17 * mm,
        title=f"Laboratory Overview - {filename}", author="MedBrief AI",
    )
    sample = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "LabTitle", parent=sample["Title"], fontName="Helvetica-Bold",
        fontSize=19, leading=23, textColor=NAVY, alignment=0, spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "LabSubtitle", parent=sample["Normal"], fontSize=8, leading=9.5,
        textColor=MUTED, spaceAfter=7,
    )
    cell = ParagraphStyle(
        "LabCell", parent=sample["BodyText"], fontSize=7.2, leading=8.7,
        textColor=TEXT,
    )
    header = ParagraphStyle(
        "LabHeader", parent=cell, fontName="Helvetica-Bold", textColor=colors.white,
    )
    rows = overview.get("key_findings", []) + overview.get("within_reference_range", [])
    counts = overview.get("counts", {})
    findings = overview.get("key_findings", [])
    below = [item for item in findings if item.get("flag") == "L"]
    above = [item for item in findings if item.get("flag") == "H"]
    summary_lines = [
        f"{counts.get('values_extracted', len(rows))} numeric results were extracted; "
        f"{counts.get('outside_reference_range', len(findings))} are outside the printed "
        "reference intervals."
    ]
    if below:
        summary_lines.append("<b>Below range:</b> " + ", ".join(
            f"{_escape(item.get('test', ''))} {_escape(item.get('value', ''))} "
            f"{_escape(item.get('unit') or '')}".strip() for item in below
        ))
    if above:
        summary_lines.append("<b>Above range:</b> " + ", ".join(
            f"{_escape(item.get('test', ''))} {_escape(item.get('value', ''))} "
            f"{_escape(item.get('unit') or '')}".strip() for item in above
        ))
    flow: list[Any] = [
        Paragraph("Laboratory Results Overview", title_style),
        Paragraph(_escape(filename), subtitle),
        *[Paragraph(line, subtitle) for line in summary_lines],
    ]
    table_rows: list[list[Any]] = [[Paragraph(label, header) for label in
                                    ("Test", "Value", "Unit", "Reference", "Flag")]]
    for row in rows:
        table_rows.append([
            Paragraph(_escape(row.get("test", "")), cell),
            Paragraph(_escape(row.get("value", "")), cell),
            Paragraph(_escape(row.get("unit") or ""), cell),
            Paragraph(_escape(row.get("reference") or ""), cell),
            Paragraph(_escape(row.get("flag") or ""), cell),
        ])
    table = Table(
        table_rows,
        colWidths=[document.width * 0.38, document.width * 0.12,
                   document.width * 0.18, document.width * 0.20,
                   document.width * 0.12],
        repeatRows=1,
    )
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    for index, row in enumerate(rows, start=1):
        if row.get("flag") in {"H", "L"}:
            commands.append(("BACKGROUND", (0, index), (-1, index), WARNING))
    table.setStyle(TableStyle(commands))
    flow.extend([table, Spacer(1, 10), Paragraph(
        "Verify all extracted results and reference intervals against the original laboratory report.",
        subtitle,
    )])
    document.build(flow, canvasmaker=NumberedCanvas)
    return output.getvalue()


def generate_grounded_summary_pdf(patient_name: str, summary: dict[str, Any]) -> bytes:
    """Create a concise structured summary PDF from the response fields, not raw excerpts."""
    details = {
        "Patient name": patient_name,
        "Patient ID": summary.get("patient_id") or "Not recorded",
        "Overview": summary.get("patient_overview") or "Not recorded",
    }
    sections: dict[str, Any] = {
        "Demographic conflicts requiring verification": summary.get("demographic_conflicts") or [],
        "Medical history": summary.get("medical_history") or [],
        "Diagnoses and clinical impressions": summary.get("diagnoses") or [],
        "Current conditions and important findings": summary.get("current_conditions") or [],
        "Laboratory observations": summary.get("laboratory_observations") or [],
        "Documented medications by report date and status": summary.get("current_medications") or [],
        "Allergies": summary.get("allergies") or [],
        "MRI findings": ([summary["mri_findings_summary"]]
                         if summary.get("mri_findings_summary") else []),
        "Documented follow-up and precautions": summary.get("recommended_follow_up") or [],
    }
    sections["Extraction and verification warnings"] = (
        summary.get("extraction_warnings") or []
    )
    timeline_rows = [{
        "date": row.get("date", "Not recorded"),
        "points": "\n".join(f"- {point}" for point in row.get("important_points", [])),
    } for row in (summary.get("chronological_timeline") or [])]
    report_rows = [{
        "type": report_type,
        "points": "\n".join(f"- {point}" for point in points[:2]),
    } for report_type, points in (summary.get("report_type_findings") or {}).items()
    if points]
    tables = []
    if timeline_rows:
        tables.append({
            "title": "Chronological timeline",
            "columns": [
                {"key": "date", "label": "Date"},
                {"key": "points", "label": "Important events"},
            ],
            "rows": timeline_rows,
            "widths": [0.20, 0.80],
        })
    if report_rows:
        tables.append({
            "title": "Report-type overview",
            "columns": [
                {"key": "type", "label": "Report type"},
                {"key": "points", "label": "Key findings"},
            ],
            "rows": report_rows,
            "widths": [0.20, 0.80],
        })
    return generate_typed_report_pdf(
        "Grounded Patient Summary",
        patient_name,
        details=details,
        sections=sections,
        tables=tables,
        disclaimer="Summary is grounded in stored reports; verify against the originals.",
    )


def generate_full_patient_record_pdf(patient: dict[str, Any]) -> bytes:
    """Create a full record as a sequence of properly structured report PDFs."""
    from app.services.discharge_service import DischargeService
    from app.services.document_service import DocumentStructureService
    from app.services.lab_service import LaboratoryService
    from app.services.pathology_service import PathologyService
    from app.services.prescription_service import PrescriptionService
    from app.services.radiology_service import RadiologyService

    reports = patient.get("reports", [])
    inventory = [{
        "type": str(report.get("report_type", "Report")).title(),
        "filename": report.get("original_filename") or "Not recorded",
        "status": str(report.get("status") or "Not recorded").title(),
        "date": report.get("report_date") or report.get("created_at") or "Not recorded",
    } for report in reports]
    components = [generate_typed_report_pdf(
        "Full Patient Record",
        patient.get("name") or "Patient",
        details={
            "Patient name": patient.get("name") or "Not recorded",
            "External ID": patient.get("external_id") or "Not recorded",
            "Date of birth": patient.get("date_of_birth") or "Not recorded",
            "Sex": patient.get("sex") or "Not recorded",
            "Patient ID": patient.get("id") or "Not recorded",
            "Registered": patient.get("created_at") or "Not recorded",
        },
        tables=[{
            "title": "Stored report inventory",
            "columns": [
                {"key": "type", "label": "Type"},
                {"key": "filename", "label": "File"},
                {"key": "status", "label": "Status"},
                {"key": "date", "label": "Report / upload date"},
            ],
            "rows": inventory,
            "widths": [0.16, 0.36, 0.14, 0.34],
        }] if inventory else [],
        disclaimer="Full record assembled from stored patient reports.",
    )]

    for report in reports:
        text = report.get("extracted_text") or ""
        if not text:
            continue
        filename = report.get("original_filename") or "Stored report"
        report_type = str(report.get("report_type") or "clinical").lower()
        if report_type in {"laboratory", "lab"}:
            components.append(generate_laboratory_report_pdf(
                filename, LaboratoryService().overview(text),
            ))
        elif report_type == "prescription":
            data = PrescriptionService().overview(text)
            components.append(generate_typed_report_pdf(
                "Medical Prescription", filename, details=data["details"],
                sections={"Special instructions and precautions": data["precautions"]},
                tables=[{
                    "title": "Medication orders",
                    "columns": [
                        {"key": "name", "label": "Medication"},
                        {"key": "directions", "label": "Directions"},
                        {"key": "quantity", "label": "Quantity"},
                        {"key": "refills", "label": "Refills"},
                    ],
                    "rows": data["medications"], "widths": [0.24, 0.52, 0.14, 0.10],
                }], disclaimer=data["disclaimer"],
            ))
        elif report_type == "radiology":
            data = RadiologyService().overview(text)
            components.append(generate_typed_report_pdf(
                data["title"], filename, details=data["metadata"], sections={
                    "Relevant clinical history": data["clinical_history"],
                    "Main imaging findings": data["findings"],
                    "Conclusion": data["conclusion"],
                    "Documented recommendations": data["recommendations"],
                    "Image references": data["image_references"],
                }, disclaimer=data["disclaimer"],
            ))
        elif report_type == "pathology":
            data = PathologyService().overview(text)
            components.append(generate_typed_report_pdf(
                data["title"], filename, details=data["details"],
                sections=data["sections"], disclaimer=data["disclaimer"],
            ))
        elif report_type == "discharge":
            data = DischargeService().overview(text)
            components.append(generate_typed_report_pdf(
                data["title"], filename, details=data["details"],
                sections={key: value for key, value in data["sections"].items()
                          if key != "Discharge Medications & Outpatient Regimen"},
                tables=[{
                    "title": "Discharge medications and outpatient regimen",
                    "columns": [
                        {"key": "name", "label": "Medication"},
                        {"key": "dose", "label": "Dose"},
                        {"key": "route / frequency", "label": "Route / frequency"},
                        {"key": "instructions", "label": "Instructions"},
                    ],
                    "rows": data.get("medications", []),
                    "widths": [0.20, 0.16, 0.24, 0.40],
                }], disclaimer=data["disclaimer"],
            ))
        else:
            data = DocumentStructureService().overview(text)
            components.append(generate_structured_report_pdf(
                "Structured Clinical Report", filename, data,
            ))

    combined = fitz.open()
    for component in components:
        source = fitz.open(stream=component, filetype="pdf")
        combined.insert_pdf(source)
        source.close()
    total = combined.page_count
    for number, page in enumerate(combined, 1):
        height, width = page.rect.height, page.rect.width
        footer_area = fitz.Rect(0, height - 52, width, height)
        page.add_redact_annot(footer_area, fill=(1, 1, 1))
        page.apply_redactions()
        page.draw_line(fitz.Point(51, height - 40), fitz.Point(width - 51, height - 40),
                       color=(0.71, 0.75, 0.80), width=0.6)
        page.insert_text(fitz.Point(51, height - 24), "MedBrief AI Clinical System",
                         fontsize=7.2, fontname="helv", color=(0.37, 0.42, 0.48))
        page.insert_text(fitz.Point(width - 103, height - 24), f"Page {number} of {total}",
                         fontsize=7.2, fontname="helv", color=(0.37, 0.42, 0.48))
    output = combined.tobytes(garbage=4, deflate=True)
    combined.close()
    return output
