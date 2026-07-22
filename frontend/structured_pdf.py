import html
import io
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (CondPageBreak, ListFlowable, ListItem, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

try:
    from frontend.content_utils import dated_entries, section_summary_points
except ModuleNotFoundError:
    from content_utils import dated_entries, section_summary_points


NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#4F81BD")
LIGHT_BLUE = colors.HexColor("#EAF0F7")
GRID = colors.HexColor("#B6C0CB")
TEXT = colors.HexColor("#202936")
MUTED = colors.HexColor("#5F6B7A")
WARNING = colors.HexColor("#FFF4D6")


def _escape(value: object) -> str:
    return html.escape(str(value).replace("\u2013", "-").replace("\u2014", "-"))


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
        Paragraph("Summarized Clinical Report", styles["title"]),
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
        flow.extend([PageBreak(), Paragraph("Section summaries", styles["h2"])])
        for heading, text in sections.items():
            flow.extend([CondPageBreak(28 * mm), Paragraph(_escape(heading), styles["h3"])])
            points = dated_entries(text) or section_summary_points(text)
            flow.append(ListFlowable(
                [ListItem(Paragraph(_escape(point), styles["body"]), leftIndent=8)
                 for point in points], bulletType="bullet", start="circle",
                leftIndent=14, bulletFontName="Helvetica", bulletFontSize=7,
                spaceAfter=5,
            ))

    notice = Table([[Paragraph(
        "<b>Clinical review notice:</b> This extractive summary must be verified against "
        "the original uploaded report.", styles["small"]
    )]], colWidths=[document.width])
    notice.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WARNING), ("BOX", (0, 0), (-1, -1), 0.6,
                                                              colors.HexColor("#D7B65D")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    flow.extend([Spacer(1, 10), notice])
    document.build(flow, canvasmaker=NumberedCanvas)
    return output.getvalue()
