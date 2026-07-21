# MedBrief AI Backend

Production-oriented FastAPI and Streamlit application for document OCR, structured laboratory
data, brain MRI segmentation, persistence, and the integrated partner-owned Qwen clinical
intelligence package.

Clinical summarization, chatbot, timeline generation, medication/allergy intelligence,
contradiction and missing-information detection, and multilingual processing are vendored from
the partner's `Doctor summarization` package. The backend only adapts stored normalized records
to that package; ownership and module boundaries remain separate.

## Capabilities

- PDF and image reports with PyMuPDF embedded-text extraction and PaddleOCR fallback
- Image cleanup, normalized text, page-level provenance, and structured JSON
- Conservative lab value extraction, reference flags, persistence, and historical series
- DICOM/NIfTI validation, NIfTI preprocessing, configured nnU-Net inference, mask metrics,
  tumor localization, physical volume, confidence sidecars, and PNG overlays
- PostgreSQL-ready SQLAlchemy schema (SQLite is the development default)
- Content validation, upload limits, SHA-256 duplicate detection, domain errors, and logging
- Versioned clean-data contracts for downstream LLM modules

MRI output is algorithmic decision support, not a diagnosis. Deploy only a validated nnU-Net
model, preserve model/version provenance, and require qualified clinical review.

## Folder structure

```text
app/
  api/          HTTP routes and dependencies
  core/         configuration, errors, logging
  db/           SQLAlchemy base and sessions
  models/       relational persistence models
  schemas/      request/response contracts
  services/     OCR, lab, MRI, upload and integration logic
  utils/        file validation/storage helpers
tests/          unit and API integration tests
frontend/       Streamlit end-to-end test interface
medbrief_clinical_intel/  partner-owned Qwen clinical intelligence package
```

## Local installation

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env  # use cp on macOS/Linux
uvicorn app.main:app --reload
```

In a second terminal, start the Streamlit interface:

```bash
streamlit run frontend/streamlit_app.py
```

Open `http://localhost:8501`. The UI expects the API at `http://127.0.0.1:8000`; override this
with `MEDBRIEF_API_URL` when the API is hosted elsewhere.

Install scanned-document OCR only where needed:

```bash
pip install -e ".[ocr]"
```

For PostgreSQL, install `.[postgres]` and set `DATABASE_URL` to a
`postgresql+psycopg://...` URL. `docker compose up --build` starts both services.

## MRI deployment

Base dependencies can validate DICOM/NIfTI and inspect metadata. Inference requires:

1. A clinically validated, task-appropriate nnU-Net model and weights.
2. `NNUNET_COMMAND` set to the deployment's inference wrapper executable.
3. `MRI_MODEL_NAME` set to the configured dataset/model identifier.
4. Optional MONAI/PyTorch dependencies via `pip install -e ".[mri]"` for extending the
   preprocessing pipeline.

The wrapper is called as:

```text
<NNUNET_COMMAND> -i <preprocessed.nii.gz> -o <segmentation.nii.gz> -d <MRI_MODEL_NAME>
```

It must output a label mask. It may put `{"confidence": 0.93}` in a JSON sidecar alongside
the mask. Confidence is returned as `null` when the model does not provide calibrated
probability output. DICOM archive upload is supported, but production conversion must be
wired to a series-aware converter (for example dcm2niix) before inference; the service does
not silently combine arbitrary DICOM instances.

## API workflow

Interactive OpenAPI documentation is at `http://localhost:8000/docs`.

1. `POST /patients` creates a patient.
2. `POST /upload-report` accepts multipart `patient_id`, `report_type`, and `file`.
3. `POST /extract-report?report_id=...` runs extraction and lab parsing when applicable.
4. `POST /upload-mri` accepts multipart `patient_id` and `file`.
5. `POST /analyze-mri?mri_id=...` runs the configured inference adapter.

Read APIs:

- `GET /patient/{id}` — patient with report/MRI references
- `GET /reports/{id}` — OCR report result
- `GET /mri/{id}` — MRI metadata and result
- `GET /labs/{id}` — structured laboratory report
- `GET /patients/{id}/lab-trends` — graph-ready series grouped by test
- `GET /integration/reports/{id}` — normalized downstream LLM contract
- `GET /integration/mri/{id}` — normalized downstream LLM contract
- `GET /clinical-intel/{patient_id}/record` — exact normalized input assembled from the database
- `POST /clinical-intel/{patient_id}/summary`
- `POST /clinical-intel/{patient_id}/timeline`
- `POST /clinical-intel/{patient_id}/medications-allergies`
- `POST /clinical-intel/{patient_id}/contradictions`
- `POST /clinical-intel/{patient_id}/missing-info`
- `POST /clinical-intel/{patient_id}/full-analysis`
- `POST /clinical-intel/{patient_id}/chat/query`

`USE_MOCK_LLM=true` is the development default, so every UI feature can be tested without
downloading the 14B model. Set it to `false` only on a machine provisioned for Qwen and install
the partner module's Hugging Face/PyTorch dependencies.

## Data model

`Patient` owns uploads, reports, lab history and MRI scans. An `Upload` records immutable file
identity and status. A `Report` may own one `LaboratoryReport`, which owns normalized
`LaboratoryValue` rows indexed by patient, test, and observation time. An `MRIScan` may own one
`MRIResult`. `TimelineEvent` is included only as a persistence/interface contract for the other
developer; no timeline generation exists here.

For production schema evolution, replace the development `create_all` startup convenience
with Alembic migrations, use managed secrets, malware scanning, encrypted object storage,
authorization, audit trails, retention policies, and healthcare-jurisdiction controls.

## Testing

```bash
pytest
ruff check app tests
```

Tests cover the report lifecycle, lab parsing, duplicate uploads, content mismatch rejection,
and safe behavior when MRI model configuration is absent.
