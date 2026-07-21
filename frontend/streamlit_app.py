import csv
import io
import json
import os
from datetime import date
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("MEDBRIEF_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = httpx.Timeout(180.0, connect=5.0)

st.set_page_config(page_title="MedBrief AI", page_icon="⚕", layout="wide")
st.title("MedBrief Patient Workspace")
st.caption("Register patients, store reports, extract clinical information, and download results.")


def request(method: str, path: str, **kwargs: Any) -> Any:
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
        if response.is_error:
            try:
                detail = response.json()
                message = detail.get("message") or detail.get("detail") or response.text
            except ValueError:
                message = response.text
            st.error(f"Request failed ({response.status_code}): {message}")
            return None
        content_type = response.headers.get("content-type", "")
        return response.json() if "json" in content_type else response.content
    except httpx.RequestError as exc:
        st.error(f"Cannot reach the backend at {API_URL}: {exc}")
        return None


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    output = io.StringIO()
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def patient_selector(people: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    if not people:
        st.info("Register a patient in the Patients tab first.")
        return None
    labels = {
        f"{person['name']} | {person.get('external_id') or 'No external ID'} | {person['id'][:8]}": person
        for person in people
    }
    label = st.selectbox("Select patient", list(labels), key=key)
    return labels[label]


def report_selector(patient_id: str, key: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    reports = request("GET", f"/patients/{patient_id}/reports") or []
    if not reports:
        st.info("No reports stored for this patient yet.")
        return None, reports
    labels = {
        f"{item['original_filename']} | {item['report_type']} | {item['status']} | {item['id'][:8]}": item
        for item in reports
    }
    label = st.selectbox("Select stored report", list(labels), key=key)
    return labels[label], reports


health = request("GET", "/health")
with st.sidebar:
    st.header("System")
    st.code(API_URL)
    if health:
        st.success("Database and backend connected")
    else:
        st.warning("Start the FastAPI backend")

people = request("GET", "/patients") if health else []
people = people or []
tabs = st.tabs(["Patients", "Reports & Labs", "Brain MRI", "Grounded Summary"])

with tabs[0]:
    st.subheader("Register a patient")
    with st.form("create_patient", clear_on_submit=True):
        col1, col2 = st.columns(2)
        name = col1.text_input("Full name")
        external_id = col2.text_input("Hospital / external ID")
        dob = col1.date_input("Date of birth", value=None, max_value=date.today())
        sex = col2.selectbox("Sex", ["", "Female", "Male", "Other", "Unknown"])
        submitted = st.form_submit_button("Register patient", type="primary")
    if submitted:
        if not name.strip():
            st.error("Patient name is required.")
        else:
            created = request("POST", "/patients", json={
                "name": name.strip(),
                "external_id": external_id.strip() or None,
                "date_of_birth": dob.isoformat() if dob else None,
                "sex": sex or None,
            })
            if created:
                st.success(f"Registered {created['name']} in the patient database.")
                st.rerun()

    st.subheader("Registered patients")
    if people:
        st.dataframe(
            [{
                "Name": p["name"], "External ID": p.get("external_id"),
                "Date of birth": p.get("date_of_birth"), "Sex": p.get("sex"),
                "Patient ID": p["id"], "Registered": p["created_at"],
            } for p in people],
            use_container_width=True,
            hide_index=True,
        )
        selected = patient_selector(people, "patients_patient")
        if selected:
            detail = request("GET", f"/patient/{selected['id']}")
            if detail:
                col1, col2, col3 = st.columns(3)
                col1.metric("Stored reports", len(detail.get("reports", [])))
                col2.metric("Stored MRI scans", len(detail.get("mri_scans", [])))
                col3.metric("External ID", detail.get("external_id") or "Not recorded")
                st.download_button(
                    "Download patient record (JSON)", json_bytes(detail),
                    file_name=f"{selected['name']}-patient-record.json", mime="application/json",
                )
    else:
        st.info("No patients have been registered.")

with tabs[1]:
    st.subheader("Reports and laboratory results")
    patient = patient_selector(people, "reports_patient")
    if patient:
        st.caption(f"All uploads and extracted results below belong to **{patient['name']}**.")
        with st.expander("Upload a new report", expanded=True):
            report_type = st.selectbox(
                "Report type",
                ["clinical", "laboratory", "pathology", "discharge", "radiology", "prescription"],
                key="report_type",
            )
            report_file = st.file_uploader(
                "Upload PDF or image", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
                key="report_upload",
            )
            if st.button("Upload report", disabled=report_file is None, type="primary"):
                uploaded = request(
                    "POST", "/upload-report",
                    data={"patient_id": patient["id"], "report_type": report_type},
                    files={"file": (report_file.name, report_file.getvalue(), report_file.type)},
                )
                if uploaded:
                    st.session_state["selected_report_id"] = uploaded["resource_id"]
                    st.success("Report stored in the selected patient's record.")
                    st.rerun()

        selected_report, reports = report_selector(patient["id"], "stored_report")
        if reports:
            st.dataframe(
                [{
                    "File": r["original_filename"], "Type": r["report_type"],
                    "Status": r["status"], "Report date": r.get("report_date"),
                    "Uploaded": r["created_at"], "Report ID": r["id"],
                } for r in reports],
                use_container_width=True,
                hide_index=True,
            )
        if selected_report:
            if selected_report["status"] != "extracted":
                if st.button("Extract selected report", type="primary"):
                    extracted = request("POST", "/extract-report",
                                        params={"report_id": selected_report["id"]})
                    if extracted:
                        st.success("Report extraction completed and saved.")
                        st.rerun()
            else:
                report = request("GET", f"/reports/{selected_report['id']}")
                if report:
                    st.subheader("Extracted report information")
                    st.text_area("Extracted text", report.get("extracted_text", ""), height=260)
                    col1, col2 = st.columns(2)
                    col1.download_button(
                        "Download extracted text", (report.get("extracted_text") or "").encode("utf-8"),
                        file_name=f"{selected_report['original_filename']}-extracted.txt",
                        mime="text/plain",
                    )
                    col2.download_button(
                        "Download report data (JSON)", json_bytes(report),
                        file_name=f"{selected_report['original_filename']}-extracted.json",
                        mime="application/json",
                    )
                    overview = request("GET", f"/reports/{selected_report['id']}/lab-overview")
                    if overview:
                        st.subheader("Laboratory overview")
                        st.markdown(overview["summary_markdown"])
                        counts = overview["counts"]
                        a, b, c = st.columns(3)
                        a.metric("Values found", counts["values_extracted"])
                        b.metric("Outside range", counts["outside_reference_range"])
                        c.metric("Within range", counts["within_reference_range"])
                        rows = overview["key_findings"] + overview["within_reference_range"]
                        if rows:
                            st.dataframe(rows, use_container_width=True, hide_index=True)
                        d1, d2 = st.columns(2)
                        d1.download_button(
                            "Download lab overview (JSON)", json_bytes(overview),
                            file_name=f"{selected_report['original_filename']}-labs.json",
                            mime="application/json",
                        )
                        d2.download_button(
                            "Download lab values (CSV)", csv_bytes(rows),
                            file_name=f"{selected_report['original_filename']}-labs.csv",
                            mime="text/csv",
                        )

        trends = request("GET", f"/patients/{patient['id']}/lab-trends")
        if trends and trends.get("series"):
            with st.expander("Patient laboratory history and trends"):
                st.json(trends["series"])
                flat_rows = [
                    {"test": test, **entry}
                    for test, entries in trends["series"].items() for entry in entries
                ]
                c1, c2 = st.columns(2)
                c1.download_button(
                    "Download trends (JSON)", json_bytes(trends),
                    file_name=f"{patient['name']}-lab-trends.json", mime="application/json",
                )
                c2.download_button(
                    "Download trends (CSV)", csv_bytes(flat_rows),
                    file_name=f"{patient['name']}-lab-trends.csv", mime="text/csv",
                )

with tabs[2]:
    st.subheader("Brain MRI")
    st.warning("MRI output is decision support and requires qualified clinical review.")
    patient = patient_selector(people, "mri_patient")
    if patient:
        detail = request("GET", f"/patient/{patient['id']}") or {}
        scans = detail.get("mri_scans", [])
        mri_file = st.file_uploader(
            "Upload DICOM, DICOM ZIP, or NIfTI", type=["dcm", "zip", "nii", "gz"],
            key="mri_upload",
        )
        if st.button("Upload MRI", disabled=mri_file is None, type="primary"):
            uploaded = request(
                "POST", "/upload-mri", data={"patient_id": patient["id"]},
                files={"file": (mri_file.name, mri_file.getvalue(),
                                  mri_file.type or "application/octet-stream")},
            )
            if uploaded:
                st.success("MRI stored in the selected patient's record.")
                st.rerun()
        if scans:
            choices = {f"{s['modality']} | {s['status']} | {s['id'][:8]}": s for s in scans}
            scan = choices[st.selectbox("Select stored MRI", list(choices), key="stored_mri")]
            st.json(scan)
            st.download_button(
                "Download MRI information (JSON)", json_bytes(scan),
                file_name=f"{patient['name']}-mri-{scan['id'][:8]}.json", mime="application/json",
            )
            if scan["status"] != "analyzed" and st.button("Run configured MRI analysis"):
                analyzed = request("POST", "/analyze-mri", params={"mri_id": scan["id"]})
                if analyzed:
                    st.success("MRI analysis completed.")
                    st.rerun()
            if scan.get("result") and scan["result"].get("overlay_path"):
                overlay = request("GET", f"/mri/{scan['id']}/overlay")
                if overlay:
                    st.image(overlay, caption="Segmentation overlay")
                    st.download_button(
                        "Download MRI overlay", overlay,
                        file_name=f"{patient['name']}-mri-overlay.png", mime="image/png",
                    )
        else:
            st.info("No MRI scans stored for this patient.")

with tabs[3]:
    st.subheader("Grounded patient summary")
    st.caption("Uses only patient details and text extracted from stored reports. No demo facts are added.")
    patient = patient_selector(people, "summary_patient")
    if patient:
        detail = request("GET", f"/patient/{patient['id']}") or {}
        extracted_count = sum(1 for report in detail.get("reports", [])
                              if report.get("status") == "extracted")
        st.info(f"{extracted_count} extracted report(s) are available for this summary.")
        if st.button("Generate grounded summary", type="primary", disabled=extracted_count == 0):
            result = request("POST", f"/clinical-intel/{patient['id']}/summary")
            if result:
                st.session_state[f"summary_{patient['id']}"] = result
        summary = st.session_state.get(f"summary_{patient['id']}")
        if summary:
            st.markdown(summary["narrative_summary_markdown"])
            with st.expander("Structured summary data"):
                st.json(summary)
            col1, col2 = st.columns(2)
            col1.download_button(
                "Download summary (Markdown)", summary["narrative_summary_markdown"].encode("utf-8"),
                file_name=f"{patient['name']}-grounded-summary.md", mime="text/markdown",
            )
            col2.download_button(
                "Download summary data (JSON)", json_bytes(summary),
                file_name=f"{patient['name']}-grounded-summary.json", mime="application/json",
            )
        record = request("GET", f"/clinical-intel/{patient['id']}/record")
        if record:
            with st.expander("Complete normalized patient information"):
                st.json(record)
                st.download_button(
                    "Download normalized record (JSON)", json_bytes(record),
                    file_name=f"{patient['name']}-normalized-record.json", mime="application/json",
                )
