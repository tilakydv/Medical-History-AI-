from app.services.document_service import DocumentStructureService


def test_general_document_structure_preserves_mixed_content():
    text = """CLINICAL FOLLOW-UP REPORT
Patient Name: Example Patient
ASSESSMENT:
Patient reports improved symptoms.
LABORATORY RESULTS:
Glucose 108 mg/dL 70-99 H
PLAN:
Repeat testing at the next documented visit."""

    overview = DocumentStructureService().overview(text, laboratory_value_count=1)

    assert overview["metadata"]["Patient Name"] == "Example Patient"
    assert "improved symptoms" in overview["sections"]["Assessment"]
    assert "Glucose 108" in overview["sections"]["Laboratory Results"]
    assert "Repeat testing" in overview["sections"]["Plan"]
    assert overview["detected_content"] == ["laboratory"]


def test_numbered_consultant_sections_do_not_turn_prose_or_dates_into_metadata():
    text = """Medical Consultant Report and Summary
Case No:
MD-42
Date:
July 25, 2009
1.
Detailed (Chronological) Analysis: A 59 year old woman was evaluated. At 10:10 AM, CPR continued.
2.
Proposed Standard(s) of Care: A complete assessment was required.
9.
Records Reviewed:
October 4, 2006: Physician Office Notes
February 11, 2007: Death Certificate
10. Additional Documents and Information Necessary: Reference text.
11. Investigational Questions for Physician: None"""

    overview = DocumentStructureService().overview(text)

    assert overview["metadata"] == {"Case No": "MD-42", "Date": "July 25, 2009"}
    assert "At 10:10 AM" in overview["sections"]["1. Detailed (Chronological) Analysis"]
    assert "October 4, 2006: Physician Office Notes" in overview["sections"]["9. Records Reviewed"]
    assert overview["sections"]["11. Investigational Questions For Physician"] == "None"


def test_page_headers_and_signature_footer_are_removed_from_numbered_sections():
    text = """Medical Consultant Report
1. Actual Harm Identified: Clinical harm text.
Case No. MD-42
Date July 25, 2009
Page 2
2. Investigational Questions for Physician: None
MD
July 25, 2009
______________________________
Print Name
MD
______________________________
Signature"""

    sections = DocumentStructureService().overview(text)["sections"]

    assert sections["1. Actual Harm Identified"] == "Clinical harm text."
    assert sections["2. Investigational Questions For Physician"] == "None"


def test_general_enumerators_and_clinician_signature_are_supported():
    text = """DISCHARGE REPORT
I.
Assessment: Symptoms improved.
II) Plan: Follow-up is documented.
Electronically signed by:
Example Clinician
DO"""

    sections = DocumentStructureService().overview(text)["sections"]

    assert sections["I. Assessment"] == "Symptoms improved."
    assert sections["II. Plan"] == "Follow-up is documented."


def test_clinical_structure_removes_header_fragments_and_embedded_footer_artifacts():
    text = """CLINICAL EVALUATION REPORT
Patient Name: Example Patient
MRN / Patient
Id
003 | abc123
Age / Sex
42 Yrs / Male Encounter Date: October 12, 2025 Attending
Md
Example Clinician Department: Internal Medicine
1. HISTORY: Patient presented with fever. (cid:127) CONFIDENTIAL MEDICAL RECORD - PATIENT ID: 003
2. PLAN: Order imaging. 1. ◦◦◦ CONFIDENTIAL MEDICAL RECORD - PATIENT ID: 003"""

    overview = DocumentStructureService().overview(text)

    assert overview["metadata"]["Patient Name"] == "Example Patient"
    assert "Id" not in overview["sections"]
    assert "Age / Sex" not in overview["sections"]
    assert "Md" not in overview["sections"]
    assert "CONFIDENTIAL" not in " ".join(overview["sections"].values())
    assert "(cid:" not in " ".join(overview["sections"].values())
    assert overview["sections"]["1. History"] == "Patient presented with fever."
