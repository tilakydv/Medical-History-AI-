from app.services.discharge_service import DischargeService
from app.services.pathology_service import PathologyService
from app.services.prescription_service import PrescriptionService
from app.services.radiology_service import RadiologyService


def test_prescription_keeps_each_medication_with_its_own_directions():
    text = """MEDICAL PRESCRIPTION
Patient Name: Test Patient
1. Azithromycin 500 mg Oral Tablet
Take 500 mg on day 1 and 250 mg on days 2 through 5.
Sig / Directions:
Take with food.
Quantity: 5 Tablets
Refills: 0
2. Benzonatate 100 mg Oral Capsule
Swallow one capsule three times daily as needed.
Sig / Directions:
Do not chew.
Quantity: 15 Capsules
Refills: 0
SPECIAL INSTRUCTIONS & PRECAUTIONS
• Complete the antibiotic course.
CONFIDENTIAL MEDICAL RECORD"""

    result = PrescriptionService().overview(text)

    assert len(result["medications"]) == 2
    assert "500 mg on day 1" in result["medications"][0]["directions"]
    assert "Do not chew" in result["medications"][1]["directions"]
    assert result["medications"][0]["quantity"] == "5 Tablets"


def test_prescription_does_not_mistake_address_for_patient_id():
    text = """1200 Medical Center Parkway, Suite 100 | Phone: (555) 019-2810
Patient Name: Test Patient 003 | 97fc05c6
Prescription Date: October 12, 2025
1. Medicine 5 mg Tablet
Directions: Take once daily.
Quantity: 5 Tablets
Refills: 0"""

    result = PrescriptionService().overview(text)

    assert result["details"]["MRN / Patient ID"] == "003 | 97fc05c6"


def test_prescription_splits_multiple_metadata_fields_on_one_line():
    text = """MEDICAL PRESCRIPTION
Age / Sex: 42 Yrs / Male Prescription Date: October 12, 2025
Prescriber: Example Clinician License No: LIC-123
1. Medicine 5 mg Tablet
Directions: Take once daily.
Quantity: 5 Tablets
Refills: 0"""

    details = PrescriptionService().overview(text)["details"]

    assert details["Age / sex"] == "42 Yrs / Male"
    assert details["Prescription date"] == "October 12, 2025"
    assert details["Prescriber"] == "Example Clinician"
    assert details["License number"] == "LIC-123"


def test_prescription_removes_neighboring_date_and_id_from_other_fields():
    text = """MEDICAL PRESCRIPTION
Patient Name: Example Patient 003 | abc123
Age / Sex: 42 Yrs / Male October 12, 2025
Prescription Date: October 12, 2025
1. Medicine 5 mg Tablet
Directions: Take once daily.
Quantity: 5 Tablets
Refills: 0"""

    details = PrescriptionService().overview(text)["details"]

    assert details["Patient name"] == "Example Patient"
    assert details["Age / sex"] == "42 Yrs / Male"


def test_pathology_recognises_numbered_histopathology_sections():
    text = """HISTOPATHOLOGY REPORT
1. CLINICAL INDICATION & SPECIMEN ORIGIN
Specimen Submitted: Punch biopsy of skin.
2. MACROSCOPIC / GROSS DESCRIPTION
Single skin punch measuring 0.6 cm.
3. MICROSCOPIC EXAMINATION & HISTOLOGICAL ANALYSIS
No cytologic atypia identified.
4. IMMUNOHISTOCHEMISTRY (IHC) & SPECIAL STAINS
S100 positive. Ki-67 below 1%.
5. PATHOLOGICAL SUMMARY & FINAL DIAGNOSIS
BENIGN COMPOUND MELANOCYTIC NEVUS. NO MALIGNANCY.
CONFIDENTIAL MEDICAL RECORD"""

    result = PathologyService().overview(text)

    assert len(result["sections"]) == 5
    assert "BENIGN COMPOUND" in result["sections"]["Pathological summary and final diagnosis"]
    assert "CONFIDENTIAL" not in result["download_text"]


def test_pathology_preserves_source_bullets_for_structured_rendering():
    result = PathologyService().overview(
        "PATHOLOGY REPORT\nDIAGNOSIS\n"
        "� First documented finding.\n� Second documented finding."
    )

    assert result["sections"]["Diagnosis"] == (
        "- First documented finding.\n- Second documented finding."
    )


def test_radiology_recognises_numbered_sections_and_removes_signature():
    text = """RADIOLOGY REPORT
Exam Type: Chest Radiograph
1. CLINICAL INDICATION & TECHNIQUE
Clinical History: Fever and cough.
Technique: PA and lateral radiographs.
2. DETAILED RADIOGRAPHIC FINDINGS
Lungs & Airways:
Right lower lobe consolidation with air bronchograms.
Pleura & Fluid Collections:
No pleural effusion.
3. IMPRESSION & DIAGNOSTIC SUMMARY
IMPRESSION: Right lower lobe pneumonia.
RECOMMENDATION: Follow-up imaging in 4-6 weeks.
Dr. Example Radiologist
CONFIDENTIAL MEDICAL RECORD"""

    result = RadiologyService().overview(text)

    assert len(result["findings"]) == 2
    assert "pneumonia" in result["conclusion"].lower()
    assert result["recommendations"] == ["RECOMMENDATION: Follow-up imaging in 4-6 weeks."]
    assert "Radiologist" not in RadiologyService().as_text(result)
    assert "Page 1 of 2" not in RadiologyService().as_text(result)


def test_radiology_splits_multiple_metadata_fields_on_one_line():
    result = RadiologyService().overview(
        "RADIOLOGY REPORT\n"
        "Exam Type: Chest Radiograph Accession No: RAD-123\n"
        "Exam Date: October 12, 2025 | 09:45 AM Ordering MD: Example Clinician\n"
        "FINDINGS\nNo acute abnormality.\nIMPRESSION\nNormal study."
    )

    assert result["metadata"]["Exam type"] == "Chest Radiograph"
    assert result["metadata"]["Accession number"] == "RAD-123"
    assert result["metadata"]["Exam date"] == "October 12, 2025 | 09:45 AM"
    assert result["metadata"]["Ordering clinician"] == "Example Clinician"


def test_radiology_recovers_identifiers_from_visually_split_columns():
    result = RadiologyService().overview(
        "RADIOLOGY REPORT\nEast Wing | Phone\nMRN / Patient\n"
        "Patient Name: Example Patient 003 | abc123\nID:\n"
        "Accession\nExam Type: Chest Radiograph RAD-2025-44120\nNo:\n"
        "FINDINGS\nNo acute abnormality.\nIMPRESSION\nNormal study."
    )

    assert result["metadata"]["Patient name"] == "Example Patient"
    assert result["metadata"]["MRN / Patient ID"] == "003 | abc123"
    assert result["metadata"]["Exam type"] == "Chest Radiograph"
    assert result["metadata"]["Accession number"] == "RAD-2025-44120"


def test_discharge_summary_keeps_diagnoses_medications_and_follow_up():
    text = """DISCHARGE SUMMARY
Admission Date: October 15, 2025
Discharge Date: October 18, 2025
1. DIAGNOSES & CLINICAL IDENTIFIERS
Final Discharge Diagnosis: Resolving bacterial pneumonia.
2. DETAILED SUMMARY OF INPATIENT HOSPITAL COURSE
Treated with ceftriaxone and azithromycin and oxygen was discontinued on day 2.
3. DISCHARGE MEDICATIONS & OUTPATIENT REGIMEN
Levofloxacin 500 mg once daily for four days.
4. DISCHARGE INSTRUCTIONS & FOLLOW-UP PLAN
Repeat chest X-ray in four weeks."""

    result = DischargeService().overview(text)

    assert len(result["sections"]) == 4
    assert "Levofloxacin" in result["sections"]["Discharge Medications & Outpatient Regimen"]
    assert "four weeks" in result["sections"]["Discharge Instructions & Follow-Up Plan"]
