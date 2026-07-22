from medbrief_clinical_intel.modules.summarizer import PatientSummarizer
from medbrief_clinical_intel.schemas.input_models import DocumentRecord, PatientRecordInput


def test_grounded_summary_extracts_categories_without_raw_report_headers():
    record = PatientRecordInput(
        patient_id="patient-1",
        patient_name="Example Patient",
        age=42,
        gender="Male",
        documents=[
            DocumentRecord(
                document_id="clinical-1",
                document_type="clinical",
                extracted_text=(
                    "HOSPITAL HEADER\n1. PAST MEDICAL HISTORY: Essential hypertension.\n"
                    "2. DIAGNOSTIC ASSESSMENT: Community-acquired pneumonia.\n"
                    "3. FOLLOW-UP PLAN: Clinical review in 48 hours."
                ),
            ),
            DocumentRecord(
                document_id="lab-1",
                document_type="laboratory",
                extracted_text="CRP 28.4 0-5 mg/L HIGH\nHemoglobin 14.8 13.5-17.5 g/dL Normal",
            ),
            DocumentRecord(
                document_id="rx-1",
                document_type="prescription",
                extracted_text=(
                    "1. Example medicine 10 mg Tablet\nDirections: Take once daily\n"
                    "Quantity: 5 Tablets\nRefills: 0"
                ),
            ),
        ],
    )

    summary = PatientSummarizer()._create_grounded_summary(record)

    assert any("hypertension" in item.lower() for item in summary.medical_history)
    assert any("pneumonia" in item.lower() for item in summary.diagnoses)
    assert any("CRP" in item and "High" in item for item in summary.laboratory_observations)
    assert any("Example medicine" in item for item in summary.current_medications)
    assert "HOSPITAL HEADER" not in summary.narrative_summary_markdown
    assert "Extracted source information" not in summary.narrative_summary_markdown


def test_grounded_summary_flags_demographic_conflicts_and_groups_reports():
    record = PatientRecordInput(
        patient_id="patient-2", patient_name="Example Patient", age=20, gender="Male",
        documents=[DocumentRecord(
            document_id="clinical-2", document_type="clinical", date="2025-10-12",
            extracted_text=(
                "Age / Sex: 42 Yrs / Male\nEncounter Date: October 12, 2025\n"
                "1. DIAGNOSTIC ASSESSMENT: Community-acquired pneumonia.\n"
                "2. FOLLOW-UP PLAN: Clinical review in 48 hours."
            ),
        )],
    )

    summary = PatientSummarizer()._create_grounded_summary(record)

    assert summary.demographic_conflicts == [
        "Registered age is 20; Clinical records age 42."
    ]
    assert "Clinical" in summary.report_type_findings
    assert summary.chronological_timeline
    assert summary.chronological_timeline[0]["date"] == "2025-10-12"


def test_grounded_summary_keeps_different_doses_and_removes_exact_duplicate_orders():
    record = PatientRecordInput(
        patient_id="patient-3", patient_name="Example Patient", age=42, gender="Male",
        documents=[
            DocumentRecord(
                document_id="rx-1", document_type="prescription", date="2025-10-12",
                extracted_text=(
                    "1. Acetaminophen 500 mg Tablet\nDirections: Take every 6 hours as needed\n"
                    "2. Acetaminophen 650 mg Tablet\nDirections: Take every 6 hours as needed"
                ),
            ),
            DocumentRecord(
                document_id="rx-2", document_type="prescription", date="2025-10-12",
                extracted_text=(
                    "1. Acetaminophen 500 mg Tablet\nDirections: Take every 6 hours as needed"
                ),
            ),
        ],
    )

    summary = PatientSummarizer()._create_grounded_summary(record)

    assert len(summary.current_medications) == 2
    assert any("500 mg" in item and "Prescribed - 2025-10-12" in item
               for item in summary.current_medications)
    assert any("650 mg" in item for item in summary.current_medications)


def test_positive_allergy_supersedes_confusing_nkda_statement():
    record = PatientRecordInput(
        patient_id="patient-4", patient_name="Example Patient",
        documents=[DocumentRecord(
            document_id="clinical-4", document_type="clinical",
            extracted_text=(
                "Past Medical History: No known drug allergies (NKDA) to non-penicillin classes.\n"
                "Allergies: Penicillin (urticarial rash in childhood)."
            ),
        )],
    )

    summary = PatientSummarizer()._create_grounded_summary(record)

    assert summary.allergies == ["Penicillin (urticarial rash in childhood)."]
    assert all("NKDA" not in item for item in summary.medical_history)
