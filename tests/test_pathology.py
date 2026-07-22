from app.services.pathology_service import PathologyService


def test_pathology_report_is_structured_as_narrative_sections():
    text = """SAMPLE PATHOLOGY REPORT
Specimen and patient identification
PATIENT:
Patient Name
DATE OF BIRTH:
01/01/1960
MEDICAL RECORD NUMBER:
Patient's hospital number
GENDER:
SPECIMEN COLLECTION DATE/TIME: 01/01/2019 12:00h
SPECIMEN RECEIPT DATE/TIME: 01/01/2019 13:00h
CASE NUMBER: Pathology case number
SPECIMEN LABEL(S):
Specimen site and medical procedure (such as Appendix, appendectomy)
PATIENT HISTORY:
Information the clinical team think the Pathologist should know about the patient.
PRE-OPERATIVE DIAGNOSIS:
The current clinical diagnosis relating to the specimen.
DIAGNOSIS:
The final diagnosis made by the Pathologist based on examination of the tissue specimen.
NOTE:
Additional information helpful to the clinical team.
SPECIAL TEST RESULTS:
Results from additional testing can be included here.
GROSS DESCRIPTION:
Description of the specimen when received by pathology."""

    overview = PathologyService().overview(text)

    assert overview["title"] == "SAMPLE PATHOLOGY REPORT"
    assert overview["details"]["Patient"] == "Patient Name"
    assert overview["details"]["Date of birth"] == "01/01/1960"
    assert overview["details"]["Case number"] == "Pathology case number"
    assert "Specimen site" in overview["sections"]["Specimen"]
    assert "current clinical diagnosis" in overview["sections"]["Pre-operative diagnosis"]
    assert "final diagnosis" in overview["sections"]["Diagnosis"]
    assert "additional testing" in overview["sections"]["Special test results"]
