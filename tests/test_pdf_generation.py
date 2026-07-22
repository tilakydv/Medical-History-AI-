import fitz

from frontend.pdf_utils import generate_pdf_bytes
from frontend.structured_pdf import (generate_laboratory_report_pdf,
                                     generate_full_patient_record_pdf,
                                     generate_grounded_summary_pdf,
                                     generate_structured_report_pdf,
                                     generate_typed_report_pdf)


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


def test_generic_download_pdf_formats_markdown_as_headings_and_bullets():
    pdf = generate_pdf_bytes(
        "Grounded Summary - Example Patient",
        "### Patient Summary Brief\n\n**Overview:** Stable follow-up.\n\n"
        "### Important findings\n\n- First documented point\n- Second documented point",
    )

    document = fitz.open(stream=pdf, filetype="pdf")
    text = " ".join(page.get_text() for page in document)
    document.close()

    assert "Patient Summary Brief" in text
    assert "Overview" in text
    assert "First documented point" in text
    assert "###" not in text
    assert "**" not in text
    assert "Clinical review notice" in text


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
    assert "Section summaries" not in text
    assert "Structured Clinical Report" in text
    assert "Clinical report details" in text
    assert "A ruptured aneurysm was documented" in text


def test_laboratory_pdf_contains_complete_result_table():
    overview = {
        "summary_markdown": "2 values extracted; 1 is outside range.",
        "key_findings": [{"test": "CRP", "value": 28.4, "unit": "mg/L",
                          "reference": "0-5", "flag": "H"}],
        "within_reference_range": [{"test": "Glucose", "value": 92,
                                    "unit": "mg/dL", "reference": "70-99", "flag": "N"}],
    }

    pdf = generate_laboratory_report_pdf("laboratory.pdf", overview)
    document = fitz.open(stream=pdf, filetype="pdf")
    text = " ".join(page.get_text() for page in document)
    document.close()

    assert "CRP" in text
    assert "Glucose" in text
    assert "Reference" in text


def test_typed_report_pdf_uses_shared_tables_and_keeps_structured_content():
    pdf = generate_typed_report_pdf(
        "Medical Prescription",
        "prescription.pdf",
        details={"Patient": "Example Patient", "Date": "October 12, 2025"},
        sections={"Precautions": ["Take with food.", "Seek help for a severe reaction."]},
        tables=[{
            "title": "Medication orders",
            "columns": [
                {"key": "name", "label": "Medication"},
                {"key": "directions", "label": "Directions"},
            ],
            "rows": [{"name": "Example medicine", "directions": "Once daily"}],
            "widths": [0.35, 0.65],
        }],
        disclaimer="Verify against the original report.",
    )
    document = fitz.open(stream=pdf, filetype="pdf")
    text = " ".join(page.get_text() for page in document)
    document.close()

    assert "Report details" in text
    assert "Medication orders" in text
    assert "Example medicine" in text
    assert "Clinical review notice" in text


def test_grounded_summary_pdf_uses_structured_response_fields_only():
    pdf = generate_grounded_summary_pdf("Example Patient", {
        "patient_id": "patient-1",
        "patient_overview": "Summary compiled from two stored reports.",
        "medical_history": ["Hypertension documented."],
        "diagnoses": ["Pneumonia documented."],
        "current_conditions": ["Productive cough."],
        "laboratory_observations": ["CRP: 28.4 mg/L (High)"],
        "current_medications": ["Example medicine: once daily"],
        "allergies": ["Penicillin allergy documented."],
        "recommended_follow_up": ["Review in 48 hours."],
        "narrative_summary_markdown": "### Raw source\n- BROKEN HEADER\n- ID",
    })
    document = fitz.open(stream=pdf, filetype="pdf")
    text = " ".join(page.get_text() for page in document)
    document.close()

    assert "Diagnoses and clinical impressions" in text
    assert "CRP: 28.4 mg/L" in text
    assert "BROKEN HEADER" not in text


def test_full_record_pdf_groups_reports_and_has_global_page_numbers():
    pdf = generate_full_patient_record_pdf({
        "id": "patient-1", "name": "Example Patient", "external_id": "E-1",
        "date_of_birth": "1990-01-01", "sex": "Female", "created_at": "2025-01-01",
        "reports": [{
            "report_type": "prescription", "original_filename": "rx.pdf",
            "status": "extracted", "created_at": "2025-01-02",
            "extracted_text": (
                "Patient Name: Example Patient\n1. Example medicine 10 mg Tablet\n"
                "Directions: Take once daily\nQuantity: 5 Tablets\nRefills: 0"
            ),
        }],
    })
    document = fitz.open(stream=pdf, filetype="pdf")
    text = " ".join(page.get_text() for page in document)
    page_count = document.page_count
    document.close()

    assert "Stored report inventory" in text
    assert "Medication orders" in text
    assert "Example medicine" in text
    assert f"Page {page_count} of {page_count}" in text
