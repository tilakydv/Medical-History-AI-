import fitz

from frontend.pdf_utils import generate_pdf_bytes
from frontend.structured_pdf import generate_structured_report_pdf


def test_generated_pdf_wraps_and_preserves_complete_long_content():
    final_words = "FINAL CLINICAL INFORMATION MUST REMAIN PRESENT"
    paragraph = ("A long extracted clinical sentence containing history, findings, and documented care. " * 30)
    pdf = generate_pdf_bytes("Structured Clinical Details", paragraph + final_words)

    document = fitz.open(stream=pdf, filetype="pdf")
    extracted = " ".join(page.get_text() for page in document)
    page_count = document.page_count
    document.close()

    assert final_words in extracted.replace("\n", " ")
    assert page_count >= 1
    assert "Page 1 of" in extracted


def test_structured_pdf_contains_table_and_bullet_content():
    content = {
        "statistics": {"lines": 12, "structured_sections": 1, "laboratory_values": 0},
        "metadata": {"Date": "March 21, 2006"},
        "timeline": [{
            "date": "2006-03-21",
            "important_points": ["Patient presented with pain.", "Emergency surgery performed."],
        }],
        "sections": {"Assessment": "A ruptured aneurysm was documented. Treatment was started."},
        "unsectioned_text": "",
    }

    pdf = generate_structured_report_pdf("Structured report", "clinical.pdf", content)
    document = fitz.open(stream=pdf, filetype="pdf")
    text = " ".join(page.get_text() for page in document)
    document.close()

    assert "Chronological timeline" in text
    assert "2006-03-21" in text
    assert "Patient presented with pain" in text
    assert "Section summaries" in text
