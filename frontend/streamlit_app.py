import os
from datetime import date
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("MEDBRIEF_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = httpx.Timeout(180.0, connect=5.0)

st.set_page_config(page_title="MedBrief AI", page_icon="🩺", layout="wide")
st.title("MedBrief AI")
st.caption("OCR, laboratory, MRI, and clinical-intelligence integration workspace")


def request(method: str, path: str, **kwargs: Any) -> Any:
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
        if response.is_error:
            detail = response.json()
            message = detail.get("message") or detail.get("detail") or response.text
            st.error(f"Request failed ({response.status_code}): {message}")
            return None
        content_type = response.headers.get("content-type", "")
        return response.json() if "json" in content_type else response.content
    except httpx.RequestError as exc:
        st.error(f"Cannot reach the backend at {API_URL}: {exc}")
        return None


def patients() -> list[dict[str, Any]]:
    return request("GET", "/patients") or []


with st.sidebar:
    st.header("Connection")
    st.code(API_URL)
    health = request("GET", "/health")
    st.success("Backend connected") if health else st.warning("Start the FastAPI backend")
    st.divider()
    people = patients() if health else []
    options = {f"{p['name']} · {p['id'][:8]}": p["id"] for p in people}
    selected_label = st.selectbox("Active patient", list(options), index=0) if options else None
    patient_id = options.get(selected_label) if selected_label else None

tabs = st.tabs(["Patients", "Reports & Labs", "Brain MRI", "Clinical AI", "Chat"])

with tabs[0]:
    st.subheader("Create a patient")
    with st.form("create_patient", clear_on_submit=True):
        col1, col2 = st.columns(2)
        name = col1.text_input("Name")
        external_id = col2.text_input("External ID")
        dob = col1.date_input("Date of birth", value=None, max_value=date.today())
        sex = col2.selectbox("Sex", ["", "Female", "Male", "Other", "Unknown"])
        submitted = st.form_submit_button("Create patient", type="primary")
    if submitted:
        payload = {"name": name, "external_id": external_id or None,
                   "date_of_birth": dob.isoformat() if dob else None, "sex": sex or None}
        created = request("POST", "/patients", json=payload)
        if created:
            st.success(f"Created {created['name']}. Select the patient in the sidebar.")
            st.json(created)
            st.rerun()
    if patient_id:
        st.subheader("Current patient record")
        detail = request("GET", f"/patient/{patient_id}")
        if detail:
            st.json(detail)

with tabs[1]:
    st.subheader("Upload and extract a clinical report")
    if not patient_id:
        st.info("Create and select a patient first.")
    else:
        report_type = st.selectbox("Report type", ["clinical", "laboratory", "pathology",
                                                    "discharge", "radiology", "prescription"])
        report_file = st.file_uploader("PDF or image", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
        if st.button("Upload report", disabled=report_file is None):
            result = request("POST", "/upload-report", data={"patient_id": patient_id,
                             "report_type": report_type}, files={"file": (report_file.name,
                             report_file.getvalue(), report_file.type)})
            if result:
                st.session_state["report_id"] = result["resource_id"]
                st.success("Report uploaded" + (" (duplicate reused)" if result["duplicate"] else ""))
                st.json(result)
        report_id = st.text_input("Report ID", value=st.session_state.get("report_id", ""))
        if st.button("Extract report", disabled=not report_id):
            extracted = request("POST", "/extract-report", params={"report_id": report_id})
            if extracted:
                st.success("Extraction complete")
                overview = request("GET", f"/reports/{report_id}/lab-overview")
                if overview:
                    st.markdown(overview["summary_markdown"])
                    counts = overview["counts"]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Values found", counts["values_extracted"])
                    col2.metric("Outside range", counts["outside_reference_range"])
                    col3.metric("Within range", counts["within_reference_range"])
                    st.subheader("Key findings requiring review")
                    if overview["key_findings"]:
                        st.dataframe(overview["key_findings"], use_container_width=True,
                                     hide_index=True)
                    else:
                        st.info("No numeric results outside the supplied reference ranges were found.")
                    st.caption(overview["disclaimer"])
                    with st.expander("Values within the supplied reference ranges"):
                        st.dataframe(overview["within_reference_range"],
                                     use_container_width=True, hide_index=True)
                st.info("For a narrative patient summary, open the Clinical AI tab and run Summary.")
                with st.expander("Technical OCR output"):
                    st.text_area("Extracted text", extracted.get("extracted_text", ""), height=260)
                    st.json(extracted.get("structured_data"))
        if st.button("Show laboratory trends"):
            trends = request("GET", f"/patients/{patient_id}/lab-trends")
            if trends:
                st.json(trends)

with tabs[2]:
    st.subheader("Upload and analyze brain MRI")
    st.warning("MRI output is decision support and requires qualified clinical review.")
    if not patient_id:
        st.info("Create and select a patient first.")
    else:
        mri_file = st.file_uploader("DICOM, DICOM ZIP, or NIfTI", type=["dcm", "zip", "nii", "gz"])
        if st.button("Upload MRI", disabled=mri_file is None):
            result = request("POST", "/upload-mri", data={"patient_id": patient_id},
                             files={"file": (mri_file.name, mri_file.getvalue(),
                                             mri_file.type or "application/octet-stream")})
            if result:
                st.session_state["mri_id"] = result["resource_id"]
                st.success("MRI uploaded and validated")
                st.json(result)
        mri_id = st.text_input("MRI ID", value=st.session_state.get("mri_id", ""))
        if st.button("Run nnU-Net analysis", disabled=not mri_id):
            analyzed = request("POST", "/analyze-mri", params={"mri_id": mri_id})
            if analyzed:
                st.success("MRI analysis complete")
                st.json(analyzed)
                overlay = request("GET", f"/mri/{mri_id}/overlay")
                if overlay:
                    st.image(overlay, caption="Segmentation overlay")

with tabs[3]:
    st.subheader("Clinical intelligence")
    st.caption("Partner-owned Qwen modules consume the normalized patient record built by this backend.")
    if not patient_id:
        st.info("Select a patient with extracted reports first.")
    else:
        operation = st.selectbox("Analysis", ["full-analysis", "summary", "timeline",
                                               "medications-allergies", "contradictions", "missing-info"])
        if st.button("Run clinical analysis", type="primary"):
            result = request("POST", f"/clinical-intel/{patient_id}/{operation}")
            if result:
                if operation == "summary" and result.get("narrative_summary_markdown"):
                    st.markdown(result["narrative_summary_markdown"])
                st.json(result)
        with st.expander("Normalized input sent to the clinical modules"):
            if st.button("Load normalized record"):
                record = request("GET", f"/clinical-intel/{patient_id}/record")
                if record:
                    st.json(record)

with tabs[4]:
    st.subheader("Grounded patient-record chat")
    if not patient_id:
        st.info("Select a patient with extracted reports first.")
    else:
        history_key = f"chat_{patient_id}"
        history = st.session_state.setdefault(history_key, [])
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if query := st.chat_input("Ask about this patient's uploaded records"):
            history.append({"role": "user", "content": query})
            payload_history = history[:-1]
            result = request("POST", f"/clinical-intel/{patient_id}/chat/query",
                             json={"query": query, "chat_history": payload_history})
            if result:
                history.append({"role": "assistant", "content": result["answer"]})
                st.rerun()
