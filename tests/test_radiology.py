from app.services.radiology_service import RadiologyService


def test_radiology_report_is_structured_without_repeated_footer():
    text = """Vet Oracle Teleneurology is a trading name of CVS (UK) Limited.
Teleneurology report
Date: 06/11/2019
Referring veterinary surgeon: Raimundo Tamagnini
Hospital: The Cat Vet
Patient name and surname: Munchkin Malost
Species (canine/feline): Feline
Breed: Himalayan
Age: 7
Sex: FN
Body areas scanned and charged: Brain, cervical spine
Relevant clinical history, clinical findings and diagnostic test results:
Seizures since July 2019. Frequency increasing.
Report
Thank you for submitting this MR study.
Main findings
-
Dilation of the left lateral ventricle (image 1)
-
Severe syringomyelia extending the whole length of the cervical spinal cord
and particularly marked between C2 and C5 (image 4)
Conclusion & recommendations
The findings represent a feline form of craniosynostosis. The prognosis is guarded.
I would suggest oral prednisolone in combination with oral gabapentin.
Surgery would be the next step in case of failure of medical management.
Best regards
Image 1 - Trv T2W thalamus
Vet Oracle Teleneurology is a trading name of CVS (UK) Limited."""

    overview = RadiologyService().overview(text)

    assert overview["title"] == "Teleneurology report"
    assert overview["metadata"]["Patient name"] == "Munchkin Malost"
    assert overview["metadata"]["Body areas scanned"] == "Brain, cervical spine"
    assert overview["clinical_history"] == "Seizures since July 2019. Frequency increasing."
    assert overview["findings"] == [
        "Dilation of the left lateral ventricle (image 1)",
        "Severe syringomyelia extending the whole length of the cervical spinal cord "
        "and particularly marked between C2 and C5 (image 4)",
    ]
    assert "craniosynostosis" in overview["conclusion"]
    assert len(overview["recommendations"]) == 2
    assert "trading name" not in RadiologyService().as_text(overview)
