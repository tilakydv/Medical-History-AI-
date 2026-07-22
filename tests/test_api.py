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


def test_mri_reference_image_can_be_stored_but_is_not_analysis_ready(client):
    import io

    from PIL import Image

    person = patient(client)
    content = io.BytesIO()
    Image.new("L", (16, 12), color=80).save(content, format="PNG")

    uploaded = client.post(
        "/upload-mri",
        data={"patient_id": person["id"]},
        files={"file": ("slice.png", content.getvalue(), "image/png")},
    )

    assert uploaded.status_code == 200
    scan = client.get(f"/mri/{uploaded.json()['resource_id']}")
    assert scan.status_code == 200
    assert scan.json()["metadata_json"]["format"] == "reference_image"
    assert scan.json()["metadata_json"]["analysis_supported"] is False


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
    markdown = summary.json()["narrative_summary_markdown"]
    assert "Hemoglobin 11.2" in markdown
    assert "Grounded Patient Summary" in markdown
    assert "Brain Tumor" not in markdown

    reports = client.get(f"/patients/{person['id']}/reports")
    assert reports.status_code == 200
    assert reports.json()[0]["original_filename"] == "history.pdf"
    assert reports.json()[0]["status"] == "extracted"


def test_clinical_chat_uses_patient_context(client):
    person = patient(client)
    response = client.post(
        f"/clinical-intel/{person['id']}/chat/query",
        json={"query": "What is documented for this patient?", "chat_history": []},
    )
    assert response.status_code == 200
    assert response.json()["patient_id"] == person["id"]
    assert response.json()["is_grounded"] is True


def test_delete_patient(client):
    person = patient(client)
    person_id = person["id"]
    uploaded = client.post(
        "/upload-report",
        data={"patient_id": person_id, "report_type": "laboratory"},
        files={"file": ("lab.pdf", pdf_bytes(), "application/pdf")},
    )
    assert uploaded.status_code == 200

    del_resp = client.delete(f"/patients/{person_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["id"] == person_id

    get_resp = client.get(f"/patient/{person_id}")
    assert get_resp.status_code == 404


def test_delete_one_report_keeps_patient_and_other_reports(client):
    person = patient(client)
    first = client.post(
        "/upload-report",
        data={"patient_id": person["id"], "report_type": "laboratory"},
        files={"file": ("first-lab.pdf", pdf_bytes(), "application/pdf")},
    ).json()
    client.post("/extract-report", params={"report_id": first["resource_id"]})

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "A separate clinical report for Test Patient")
    second_content = document.tobytes()
    document.close()
    second = client.post(
        "/upload-report",
        data={"patient_id": person["id"], "report_type": "clinical"},
        files={"file": ("second.pdf", second_content, "application/pdf")},
    ).json()

    deleted = client.delete(f"/reports/{first['resource_id']}")

    assert deleted.status_code == 200
    assert client.get(f"/reports/{first['resource_id']}").status_code == 404
    assert client.get(f"/patient/{person['id']}").status_code == 200
    reports = client.get(f"/patients/{person['id']}/reports").json()
    assert [report["id"] for report in reports] == [second["resource_id"]]


def test_update_patient(client):
    person = patient(client)
    person_id = person["id"]
    patch_resp = client.patch(f"/patients/{person_id}", json={"name": "Updated Name", "external_id": "P-999"})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["name"] == "Updated Name"
    assert patch_resp.json()["external_id"] == "P-999"

    get_resp = client.get(f"/patient/{person_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Updated Name"


def test_rejects_mismatched_patient_document(client):
    person = patient(client)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Patient Name: Alex Smith\nHemoglobin 14.2 g/dL 12.0-16.0")
    content = document.tobytes()
    document.close()

    uploaded = client.post(
        "/upload-report",
        data={"patient_id": person["id"], "report_type": "laboratory"},
        files={"file": ("mismatched.pdf", content, "application/pdf")},
    )
    assert uploaded.status_code == 200
    report_id = uploaded.json()["resource_id"]

    extracted = client.post("/extract-report", params={"report_id": report_id})
    assert extracted.status_code == 422
    assert "does not match registered patient" in extracted.json()["message"]


def test_clinical_chat_ollama_provider(client):
    from unittest.mock import patch, MagicMock
    from medbrief_clinical_intel.config import default_config
    
    person = patient(client)
    
    # Temporarily set config to ollama
    original_provider = default_config.llm_provider
    original_url = default_config.ollama_api_url
    original_model = default_config.ollama_model
    
    default_config.llm_provider = "ollama"
    default_config.ollama_api_url = "http://localhost:11434"
    default_config.ollama_model = "test-model"
    
    from app.api.clinical_routes import adapter
    adapter.cache_clear()
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "content": "This is a mock Ollama response about the patient."
        }
    }
    
    with patch("httpx.post", return_value=mock_response) as mock_post:
        response = client.post(
            f"/clinical-intel/{person['id']}/chat/query",
            json={"query": "Give me info", "chat_history": []},
        )
        assert response.status_code == 200
        assert response.json()["answer"] == "This is a mock Ollama response about the patient."
        
        # Verify httpx.post was called with the correct parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:11434/api/chat"
        assert kwargs["json"]["model"] == "test-model"
        
    # Reset config
    default_config.llm_provider = original_provider
    default_config.ollama_api_url = original_url
    default_config.ollama_model = original_model
    adapter.cache_clear()


def test_clinical_chatbot_only_llm_routing(client):
    from unittest.mock import patch, MagicMock
    from medbrief_clinical_intel.config import default_config
    from app.api.clinical_routes import adapter
    
    person = patient(client)
    
    # Temporarily set config to gemini with key
    original_provider = default_config.llm_provider
    original_gemini_key = default_config.gemini_api_key
    original_chatbot_only = default_config.chatbot_only_llm
    
    default_config.llm_provider = "gemini"
    default_config.gemini_api_key = "fake-key"
    default_config.chatbot_only_llm = True
    
    adapter.cache_clear()
    
    # Test Chatbot Query: should trigger Gemini API call
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Gemini chatbot reply"}]
                }
            }
        ]
    }
    
    with patch("httpx.post", return_value=mock_response) as mock_post:
        response = client.post(
            f"/clinical-intel/{person['id']}/chat/query",
            json={"query": "Chat query", "chat_history": []},
        )
        assert response.status_code == 200
        assert response.json()["answer"] == "Gemini chatbot reply"
        mock_post.assert_called_once()
        
    # Test Summarizer: should bypass Gemini and use mock/local summary because chatbot_only_llm is True
    adapter.cache_clear()
    with patch("httpx.post") as mock_post_sum:
        sum_response = client.post(f"/clinical-intel/{person['id']}/summary")
        assert sum_response.status_code == 200
        mock_post_sum.assert_not_called()
        
    # Reset config
    default_config.llm_provider = original_provider
    default_config.gemini_api_key = original_gemini_key
    default_config.chatbot_only_llm = original_chatbot_only
    adapter.cache_clear()

