import csv
import io
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
    report_types = sorted({item["report_type"] for item in reports})
    selected_type = st.selectbox(
        "Filter reports by type", ["all", *report_types], key=f"{key}_type",
    )
    visible_reports = (
        reports if selected_type == "all"
        else [item for item in reports if item["report_type"] == selected_type]
    )
    labels = {
        f"{item['original_filename']} | {item['report_type']} | {item['status']} | {item['id'][:8]}": item
        for item in visible_reports
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
                    "Download patient record (CSV)", csv_bytes([{
                        "patient_id": detail["id"], "name": detail["name"],
                        "external_id": detail.get("external_id"),
                        "date_of_birth": detail.get("date_of_birth"),
                        "sex": detail.get("sex"), "registered": detail["created_at"],
                        "stored_reports": len(detail.get("reports", [])),
                        "stored_mri_scans": len(detail.get("mri_scans", [])),
                    }]),
                    file_name=f"{selected['name']}-patient-record.csv", mime="text/csv",
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
                    if st.button("Re-extract and refresh stored lab values"):
                        refreshed = request("POST", "/extract-report",
                                            params={"report_id": selected_report["id"]})
                        if refreshed:
                            st.success("Extraction and stored laboratory values refreshed.")
                            st.rerun()
                    st.text_area("Extracted text", report.get("extracted_text", ""), height=260)
                    col1, col2 = st.columns(2)
                    col1.download_button(
                        "Download extracted text", (report.get("extracted_text") or "").encode("utf-8"),
                        file_name=f"{selected_report['original_filename']}-extracted.txt",
                        mime="text/plain",
                    )
                    col2.download_button(
                        "Download report details (CSV)", csv_bytes([{
                            "report_id": report["id"], "patient_id": report["patient_id"],
                            "filename": selected_report["original_filename"],
                            "report_type": report["report_type"], "status": report["status"],
                            "extraction_method": report.get("extraction_method"),
                            "report_date": report.get("report_date"),
                        }]),
                        file_name=f"{selected_report['original_filename']}-details.csv",
                        mime="text/csv",
                    )
                    report_type = selected_report["report_type"].lower()
                    if report_type in {"lab", "laboratory", "pathology"}:
                        overview = request(
                            "GET", f"/reports/{selected_report['id']}/lab-overview"
                        )
                    else:
                        overview = None
                    if overview and report_type in {"lab", "laboratory", "pathology"}:
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
                        st.download_button(
                            "Download lab values (CSV)", csv_bytes(rows),
                            file_name=f"{selected_report['original_filename']}-labs.csv",
                            mime="text/csv",
                        )
                    elif report_type == "radiology":
                        radiology = request(
                            "GET", f"/reports/{selected_report['id']}/radiology-overview"
                        )
                        st.subheader("Radiology report")
                        st.info(
                            "This is the written radiology interpretation. Upload the actual "
                            "DICOM or NIfTI scan in the Brain MRI tab for imaging analysis."
                        )
                        if radiology:
                            if radiology["metadata"]:
                                st.markdown("#### Report and patient details")
                                st.dataframe(
                                    [{"Detail": key, "Information": value}
                                     for key, value in radiology["metadata"].items()],
                                    use_container_width=True, hide_index=True,
                                )
                            if radiology["clinical_history"]:
                                st.markdown("#### Relevant clinical history")
                                st.write(radiology["clinical_history"])
                            st.markdown("#### Main imaging findings")
                            if radiology["findings"]:
                                for number, finding in enumerate(radiology["findings"], start=1):
                                    st.markdown(f"{number}. {finding}")
                            else:
                                st.caption("No dedicated findings section was identified.")
                            if radiology["conclusion"]:
                                st.markdown("#### Conclusion")
                                st.write(radiology["conclusion"])
                            if radiology["recommendations"]:
                                st.markdown("#### Documented recommendations")
                                for recommendation in radiology["recommendations"]:
                                    st.markdown(f"- {recommendation}")
                            if radiology["image_references"]:
                                with st.expander("Image references"):
                                    for reference in radiology["image_references"]:
                                        st.write(reference)
                            st.caption(radiology["disclaimer"])
                            col1, col2 = st.columns(2)
                            col1.download_button(
                                "Download structured radiology report",
                                radiology["download_text"].encode("utf-8"),
                                file_name=(
                                    f"{selected_report['original_filename']}-structured.txt"
                                ),
                                mime="text/plain",
                            )
                            col2.download_button(
                                "Download radiology findings (CSV)",
                                csv_bytes([{"finding": item}
                                           for item in radiology["findings"]]),
                                file_name=(
                                    f"{selected_report['original_filename']}-findings.csv"
                                ),
                                mime="text/csv",
                            )
                    else:
                        st.subheader(f"{report_type.title()} report")
                        st.caption(
                            "The extracted report text is displayed above and is available "
                            "for inclusion in the grounded patient summary."
                        )

        trends = request("GET", f"/patients/{patient['id']}/lab-trends")
        if trends and trends.get("series"):
            with st.expander("Patient laboratory history and trends"):
                flat_rows = [
                    {"test": test, **entry}
                    for test, entries in trends["series"].items() for entry in entries
                ]
                st.dataframe(flat_rows, use_container_width=True, hide_index=True)
                st.download_button(
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
            "Upload MRI file", key="mri_upload",
        )
        st.caption(
            "Accepted: NIfTI (.nii/.nii.gz), DICOM (.dcm/.dicom), DICOM ZIP, and "
            "reference images (PNG, JPEG, TIFF). Only 3D NIfTI is currently analysis-ready."
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
            m1, m2, m3 = st.columns(3)
            m1.metric("Modality", scan["modality"])
            m2.metric("Status", scan["status"].title())
            m3.metric("Study date", scan.get("study_date") or "Not recorded")
            result = scan.get("result")
            if result:
                st.subheader("MRI analysis result")
                st.dataframe([{
                    "Model": result.get("model_name"),
                    "Model version": result.get("model_version"),
                    "Tumor volume (mm³)": result.get("tumor_volume_mm3"),
                    "Confidence": result.get("confidence_score"),
                    "Localization": str(result.get("localization") or "Not recorded"),
                    "Findings": str(result.get("findings") or "Not recorded"),
                }], use_container_width=True, hide_index=True)
            st.download_button(
                "Download MRI information (CSV)", csv_bytes([{
                    "mri_id": scan["id"], "patient_id": scan["patient_id"],
                    "modality": scan["modality"], "format": scan["format"],
                    "study_date": scan.get("study_date"), "status": scan["status"],
                    "tumor_volume_mm3": result.get("tumor_volume_mm3") if result else None,
                    "confidence": result.get("confidence_score") if result else None,
                }]),
                file_name=f"{patient['name']}-mri-{scan['id'][:8]}.csv", mime="text/csv",
            )
            analysis_supported = (scan.get("metadata_json") or {}).get("format") == "nifti"
            if scan["status"] != "analyzed" and st.button(
                "Run configured MRI analysis", disabled=not analysis_supported,
                help=None if analysis_supported else "3D analysis currently requires NIfTI input.",
            ):
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
            summary_sections = {
                "Medical history": summary.get("medical_history", []),
                "Diagnoses": summary.get("diagnoses", []),
                "Current conditions": summary.get("current_conditions", []),
                "Laboratory observations": summary.get("laboratory_observations", []),
                "Current medications": summary.get("current_medications", []),
                "Allergies": summary.get("allergies", []),
                "Documented follow-up": summary.get("recommended_follow_up", []),
            }
            with st.expander("Summary details"):
                for heading, items in summary_sections.items():
                    st.markdown(f"**{heading}**")
                    if items:
                        st.markdown("\n".join(f"- {item}" for item in items))
                    else:
                        st.caption("Not found in the extracted records.")
            st.download_button(
                "Download summary (Markdown)", summary["narrative_summary_markdown"].encode("utf-8"),
                file_name=f"{patient['name']}-grounded-summary.md", mime="text/markdown",
            )
        record = request("GET", f"/clinical-intel/{patient['id']}/record")
        if record:
            documents = record.get("documents", [])
            with st.expander("Extracted source documents"):
                st.dataframe([{
                    "Type": document.get("document_type"),
                    "Date": document.get("date") or "Not recorded",
                    "Document ID": document.get("document_id"),
                } for document in documents], use_container_width=True, hide_index=True)
                combined_text = "\n\n".join(
                    f"{document.get('document_type', 'Report')} | {document.get('date') or 'Date not recorded'}\n"
                    f"{document.get('extracted_text', '')}"
                    for document in documents
                )
                st.download_button(
                    "Download all extracted source text", combined_text.encode("utf-8"),
                    file_name=f"{patient['name']}-extracted-records.txt", mime="text/plain",
                )
