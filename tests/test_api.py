import fitz


def patient(client):
    response = client.post("/patients", json={"name": "Test Patient", "external_id": "P-001"})
    assert response.status_code == 201
    return response.json()


def pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Hemoglobin 11.2 g/dL 12.0-16.0 L\nGlucose 108 mg/dL 70-99 H")
    content = document.tobytes()
    document.close()
    return content


def test_report_upload_extract_and_lab_contract(client):
    person = patient(client)
    response = client.post("/upload-report", data={"patient_id": person["id"],
                           "report_type": "laboratory"},
                           files={"file": ("lab.pdf", pdf_bytes(), "application/pdf")})
    assert response.status_code == 200
    uploaded = response.json()
    report_id = uploaded["resource_id"]

    extracted = client.post("/extract-report", params={"report_id": report_id})
    assert extracted.status_code == 200
    assert "Hemoglobin" in extracted.json()["extracted_text"]

    contract = client.get(f"/integration/reports/{report_id}")
    assert contract.status_code == 200
    values = contract.json()["laboratory_values"]
    assert values[0]["name"] == "Hemoglobin"
    assert values[0]["flag"] == "L"


def test_duplicate_upload_is_idempotent(client):
    person = patient(client)
    kwargs = {"data": {"patient_id": person["id"]},
              "files": {"file": ("report.pdf", pdf_bytes(), "application/pdf")}}
    first = client.post("/upload-report", **kwargs)
    second = client.post("/upload-report", **kwargs)
    assert first.status_code == second.status_code == 200
    assert second.json()["duplicate"] is True
    assert first.json()["resource_id"] == second.json()["resource_id"]


def test_rejects_extension_content_mismatch(client):
    person = patient(client)
    response = client.post("/upload-report", data={"patient_id": person["id"]},
                           files={"file": ("fake.pdf", b"not a pdf", "application/pdf")})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_file"


def test_missing_mri_model_returns_service_unavailable(client):
    person = patient(client)
    # Minimal structurally loadable NIfTI is generated in the test itself.
    import tempfile
    from pathlib import Path
    import nibabel as nib
    import numpy as np
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "brain.nii"
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4)), target)
        content = target.read_bytes()
    uploaded = client.post("/upload-mri", data={"patient_id": person["id"]},
                           files={"file": ("brain.nii", content, "application/octet-stream")})
    assert uploaded.status_code == 200
    response = client.post("/analyze-mri", params={"mri_id": uploaded.json()["resource_id"]})
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_clinical_intelligence_uses_persisted_ocr_record(client):
    person = patient(client)
    uploaded = client.post(
        "/upload-report",
        data={"patient_id": person["id"], "report_type": "clinical"},
        files={"file": ("history.pdf", pdf_bytes(), "application/pdf")},
    ).json()
    client.post("/extract-report", params={"report_id": uploaded["resource_id"]})

    record = client.get(f"/clinical-intel/{person['id']}/record")
    assert record.status_code == 200
    assert record.json()["documents"][0]["document_id"] == uploaded["resource_id"]

    summary = client.post(f"/clinical-intel/{person['id']}/summary")
    assert summary.status_code == 200
    assert summary.json()["patient_id"] == person["id"]
    assert summary.json()["narrative_summary_markdown"]


def test_clinical_chat_uses_patient_context(client):
    person = patient(client)
    response = client.post(
        f"/clinical-intel/{person['id']}/chat/query",
        json={"query": "What is documented for this patient?", "chat_history": []},
    )
    assert response.status_code == 200
    assert response.json()["patient_id"] == person["id"]
    assert response.json()["is_grounded"] is True
