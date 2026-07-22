import html
from typing import Any


def timeline_table_html(rows: list[dict[str, Any]]) -> str:
    """Render one merged date cell with one adjacent row per important point."""
    body: list[str] = []
    for date_row in rows:
        points = date_row.get("important_points") or ["No details recorded."]
        for index, point in enumerate(points):
            cells: list[str] = ["<tr>"]
            if index == 0:
                cells.append(
                    f'<td class="timeline-date" rowspan="{len(points)}">'
                    f'{html.escape(str(date_row.get("date", "")))}</td>'
                )
            cells.append(
                '<td class="timeline-point"><span class="timeline-bullet">•</span> '
                f'{html.escape(str(point))}</td></tr>'
            )
            body.append("".join(cells))

    return """
<style>
.timeline-wrap { overflow-x: auto; margin: 0.5rem 0 1rem; }
.timeline-table { width: 100%; border-collapse: separate; border-spacing: 0;
  border: 1px solid rgba(128,128,128,.35); border-radius: 10px; overflow: hidden; }
.timeline-table th { text-align: left; padding: 12px; background: rgba(128,128,128,.12);
  border-bottom: 1px solid rgba(128,128,128,.35); }
.timeline-table td { padding: 11px 12px; border-bottom: 1px solid rgba(128,128,128,.25); }
.timeline-table tr:last-child td { border-bottom: none; }
.timeline-date { width: 18%; min-width: 130px; vertical-align: top; font-weight: 600;
  border-right: 1px solid rgba(128,128,128,.35); background: rgba(128,128,128,.04); }
.timeline-point { width: 82%; line-height: 1.45; }
.timeline-bullet { font-weight: 700; margin-right: 4px; }
</style>
<div class="timeline-wrap">
  <table class="timeline-table">
    <thead><tr><th>Date</th><th>Important points</th></tr></thead>
    <tbody>""" + "".join(body) + """</tbody>
  </table>
</div>"""
