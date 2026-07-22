from frontend.table_utils import timeline_table_html


def test_timeline_table_merges_date_cell_and_uses_one_row_per_point():
    rendered = timeline_table_html([{
        "date": "2006-03-21",
        "important_points": ["Patient returned with pain.", "Emergency surgery performed."],
    }])

    assert 'rowspan="2">2006-03-21</td>' in rendered
    assert rendered.count('class="timeline-point"') == 2
    assert "Patient returned with pain." in rendered
    assert "Emergency surgery performed." in rendered


def test_timeline_table_escapes_report_text():
    rendered = timeline_table_html([{
        "date": "2026-01-01",
        "important_points": ["<script>alert('unsafe')</script>"],
    }])

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
