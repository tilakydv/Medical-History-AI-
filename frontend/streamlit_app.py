import csv
import io
import os
from datetime import date
from typing import Any

import httpx
import streamlit as st

try:
    from frontend.content_utils import dated_entries, section_summary_points
    from frontend.pdf_utils import generate_pdf_bytes
    from frontend.structured_pdf import generate_structured_report_pdf
    from frontend.table_utils import timeline_table_html
except ModuleNotFoundError:
    # Streamlit adds the script's own directory to sys.path when launched as
    # `streamlit run frontend/streamlit_app.py`.
    from content_utils import dated_entries, section_summary_points
    from pdf_utils import generate_pdf_bytes
    from structured_pdf import generate_structured_report_pdf
    from table_utils import timeline_table_html

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


def format_patient_record_text(detail: dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        "MEDBRIEF CLINICAL PATIENT RECORD",
        "=" * 60,
        f"Patient Name : {detail.get('name', 'N/A')}",
        f"External ID  : {detail.get('external_id') or 'Not recorded'}",
        f"Date of Birth: {detail.get('date_of_birth') or 'Not recorded'}",
        f"Sex          : {detail.get('sex') or 'Not recorded'}",
        f"Patient ID   : {detail.get('id', 'N/A')}",
        f"Registered   : {detail.get('created_at', 'N/A')}",
        "",
        "-" * 60,
        f"STORED REPORTS ({len(detail.get('reports', []))})",
        "-" * 60,
    ]
    reports = detail.get("reports", [])
    if not reports:
        lines.append("No reports stored.")
    for r in reports:
        lines.append(f"• File: {r.get('original_filename', 'N/A')} | Type: {r.get('report_type', '')} | Status: {r.get('status', '')}")
        lines.append(f"  Uploaded: {r.get('created_at', '')}")
        if r.get("extracted_text"):
            lines.append("  Extracted Text:")
            for t_line in r["extracted_text"].splitlines():
                lines.append(f"    {t_line}")
        lines.append("")

    mri_scans = detail.get("mri_scans", [])
    lines.extend([
        "-" * 60,
        f"STORED MRI SCANS ({len(mri_scans)})",
        "-" * 60,
    ])
    if not mri_scans:
        lines.append("No MRI scans stored.")
    for scan in mri_scans:
        lines.append(f"• Modality: {scan.get('modality', 'MRI')} | Format: {scan.get('format', '')} | Status: {scan.get('status', '')}")
        if scan.get("result"):
            res = scan["result"]
            lines.append(f"  Tumor Volume: {res.get('tumor_volume_mm3', 0)} mm³")
            lines.append(f"  Confidence  : {res.get('confidence_score', 'N/A')}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_lab_trends_text(patient_name: str, trends: dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        f"LABORATORY HISTORY AND TRENDS - {patient_name.upper()}",
        "=" * 60,
    ]
    for test, entries in trends.get("series", {}).items():
        lines.extend(["", test, "-" * len(test)])
        for entry in entries:
            value = entry.get("value", "Not recorded")
            unit = entry.get("unit") or ""
            observed = entry.get("observed_at") or "Date not recorded"
            flag = entry.get("flag") or "Not flagged"
            result = f"{value} {unit}".strip()
            lines.append(f"{observed}: {result} | Flag: {flag}")
    return "\n".join(lines)


def format_mri_text(scan: dict[str, Any], patient_name: str) -> str:
    lines = [
        "=" * 60,
        f"BRAIN MRI RECORD - {patient_name.upper()}",
        "=" * 60,
        f"Modality   : {scan.get('modality', 'MRI')}",
        f"Format     : {scan.get('format', 'N/A')}",
        f"Status     : {scan.get('status', 'N/A')}",
        f"Study Date : {scan.get('study_date') or 'Not recorded'}",
        f"Created    : {scan.get('created_at', 'N/A')}",
        "",
    ]
    if scan.get("result"):
        res = scan["result"]
        lines.extend([
            "-" * 60,
            "ANALYSIS RESULTS",
            "-" * 60,
            f"Model Name  : {res.get('model_name', 'N/A')}",
            f"Tumor Volume: {res.get('tumor_volume_mm3', 0)} mm³",
            f"Confidence  : {res.get('confidence_score', 'N/A')}",
            f"Localization: {res.get('localization', {})}",
        ])
    lines.append("=" * 60)
    return "\n".join(lines)


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
    label = st.selectbox(
        "Select stored report",
        list(labels),
        index=None,
        placeholder="Choose a stored report",
        key=key,
    )
    if label is None:
        st.info("Select a stored report to load its extracted information.")
        return None, reports
    return labels[label], reports


def clear_generated_results() -> None:
    """Clear results that may become stale after navigation or a new upload."""
    for key in list(st.session_state):
        if key.startswith("summary_"):
            del st.session_state[key]


def clear_report_selection() -> None:
    clear_generated_results()
    for key in list(st.session_state):
        if key.startswith("extracted_text_"):
            del st.session_state[key]
    st.session_state.pop("stored_report", None)
    st.session_state.pop("stored_report_type", None)
    st.session_state.pop("selected_report_id", None)


def clear_mri_selection() -> None:
    clear_generated_results()
    st.session_state.pop("stored_mri", None)


def refresh_current_page() -> None:
    """Reset transient widgets while preserving database-backed patient information."""
    clear_report_selection()
    clear_mri_selection()
    st.session_state.pop("report_upload", None)
    st.session_state.pop("mri_upload", None)


health = request("GET", "/health")
with st.sidebar:
    st.header("System")
    st.code(API_URL)
    if health:
        st.success("Database and backend connected")
    else:
        st.warning("Start the FastAPI backend")
    st.divider()
    st.header("Navigation")
    page = st.radio(
        "Open page",
        ["Patients", "Reports & Labs", "Brain MRI", "Grounded Summary", "AI Chatbot"],
        key="active_page",
        on_change=refresh_current_page,
        label_visibility="collapsed",
    )
    st.caption("Pages refresh automatically when navigation, files, or stored data change.")

people = request("GET", "/patients") if health else []
people = people or []

if page == "Patients":
    st.subheader("Register a patient")
    with st.form("create_patient", clear_on_submit=True):
        col1, col2 = st.columns(2)
        name = col1.text_input("Full name")
        external_id = col2.text_input("Hospital / external ID")
        dob = col1.date_input("Date of birth", value=None, min_value=date(1900, 1, 1), max_value=date.today())
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
        st.markdown("---")
        st.subheader("Manage or Delete Patient Record")
        selected = patient_selector(people, "patients_patient")
        if selected:
            detail = request("GET", f"/patient/{selected['id']}")
            if detail:
                col1, col2, col3 = st.columns(3)
                col1.metric("Stored reports", len(detail.get("reports", [])))
                col2.metric("Stored MRI scans", len(detail.get("mri_scans", [])))
                col3.metric("External ID", detail.get("external_id") or "Not recorded")
                
                st.download_button(
                    "📄 Download Patient Record (PDF)",
                    generate_pdf_bytes(f"Patient Record - {selected['name']}", format_patient_record_text(detail)),
                    file_name=f"{selected['name']}-patient-record.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                
                with st.expander("✏️ Edit Patient Details", expanded=False):
                    with st.form(f"edit_patient_{selected['id']}"):
                        edit_col1, edit_col2 = st.columns(2)
                        new_name = edit_col1.text_input("Full name", value=selected["name"])
                        new_ext_id = edit_col2.text_input("Hospital / external ID", value=selected.get("external_id") or "")
                        current_dob = date.fromisoformat(selected["date_of_birth"]) if selected.get("date_of_birth") else None
                        new_dob = edit_col1.date_input("Date of birth", value=current_dob, min_value=date(1900, 1, 1), max_value=date.today())
                        sex_options = ["", "Female", "Male", "Other", "Unknown"]
                        current_sex_idx = sex_options.index(selected["sex"]) if selected.get("sex") in sex_options else 0
                        new_sex = edit_col2.selectbox("Sex", sex_options, index=current_sex_idx)
                        save_submitted = st.form_submit_button("Save Patient Changes", type="primary")
                    if save_submitted:
                        if not new_name.strip():
                            st.error("Patient name is required.")
                        else:
                            updated = request("PATCH", f"/patients/{selected['id']}", json={
                                "name": new_name.strip(),
                                "external_id": new_ext_id.strip() or None,
                                "date_of_birth": new_dob.isoformat() if new_dob else None,
                                "sex": new_sex or None,
                            })
                            if updated:
                                st.success(f"Updated patient details for {updated['name']}.")
                                clear_generated_results()
                                st.rerun()
                
                st.markdown("##### 🔴 Danger Zone: Delete Patient")
                st.warning(f"Delete patient **{selected['name']}** (ID: `{selected['id']}`)? This purges all reports, lab data, MRI scans, and files permanently.")
                confirm_del = st.checkbox(f"Confirm I want to permanently delete {selected['name']}", key=f"chk_del_{selected['id']}")
                if st.button(f"🗑️ Permanently Delete {selected['name']}", key=f"btn_del_{selected['id']}", type="primary", disabled=not confirm_del, use_container_width=True):
                    deleted = request("DELETE", f"/patients/{selected['id']}")
                    if deleted:
                        st.success(f"Patient {selected['name']} deleted successfully from memory.")
                        st.rerun()
    else:
        st.info("No patients have been registered.")

if page == "Reports & Labs":
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
                key="report_upload", on_change=clear_report_selection,
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
                    clear_generated_results()
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
            with st.expander("Delete selected report", expanded=False):
                st.warning(
                    f"Delete **{selected_report['original_filename']}** from "
                    f"{patient['name']}'s record? This also removes its extracted and "
                    "laboratory data."
                )
                confirm_report_delete = st.checkbox(
                    "I understand this report will be permanently deleted",
                    key=f"confirm_report_delete_{selected_report['id']}",
                )
                if st.button(
                    "Permanently delete this report",
                    key=f"delete_report_{selected_report['id']}",
                    type="primary",
                    disabled=not confirm_report_delete,
                ):
                    deleted = request("DELETE", f"/reports/{selected_report['id']}")
                    if deleted:
                        clear_generated_results()
                        st.success(deleted["message"])
                        st.rerun()
            if selected_report["status"] != "extracted":
                if st.button("Extract selected report", type="primary"):
                    extracted = request("POST", "/extract-report",
                                        params={"report_id": selected_report["id"]})
                    if extracted:
                        st.success("Report extraction completed and saved.")
                        clear_generated_results()
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
                            clear_generated_results()
                            st.rerun()
                    st.text_area(
                        "Extracted text",
                        report.get("extracted_text", ""),
                        height=260,
                        key=f"extracted_text_{selected_report['id']}",
                    )
                    col1, col2 = st.columns(2)
                    col1.download_button(
                        "📄 Download Extracted Report (PDF)",
                        generate_pdf_bytes(f"Report - {selected_report['original_filename']}", report.get("extracted_text", "")),
                        file_name=f"{selected_report['original_filename']}-extracted.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    report_type = selected_report["report_type"].lower()
                    content = request(
                        "GET", f"/reports/{selected_report['id']}/content-overview"
                    )
                    detected = set(content.get("detected_content", [])) if content else set()
                    if content:
                        st.subheader("Summarized structured report content")
                        st.caption(
                            "Point-wise extractive summary. Use the extracted text above to review "
                            "the complete source wording."
                        )
                        stats = content["statistics"]
                        s1, s2, s3 = st.columns(3)
                        s1.metric("Extracted lines", stats["lines"])
                        s2.metric("Structured sections", stats["structured_sections"])
                        s3.metric("Clinical values", stats["laboratory_values"])
                        if content["metadata"]:
                            st.markdown("#### Document details")
                            st.dataframe(
                                [{"Detail": key, "Information": value}
                                 for key, value in content["metadata"].items()],
                                use_container_width=True,
                                hide_index=True,
                            )
                        timeline = content.get("timeline", [])
                        if timeline:
                            st.markdown("#### Chronological timeline")
                            st.caption(
                                "Explicitly dated events extracted from the report and ordered "
                                "from earliest to latest."
                            )
                            st.markdown(timeline_table_html(timeline), unsafe_allow_html=True)
                        if content["sections"]:
                            st.markdown("#### Section summaries")
                            for heading, section_text in content["sections"].items():
                                with st.expander(heading):
                                    entries = dated_entries(section_text)
                                    if entries:
                                        for entry in entries:
                                            st.markdown(f"- {entry}")
                                    else:
                                        for point in section_summary_points(section_text):
                                            st.markdown(f"- {point}")
                        if content["unsectioned_text"] and not content["sections"]:
                            st.markdown("#### Report content")
                            st.write(content["unsectioned_text"])
                        st.download_button(
                            "Download Summarized Details (PDF)",
                            generate_structured_report_pdf(
                                f"Summarized Details - {selected_report['original_filename']}",
                                selected_report["original_filename"],
                                content,
                            ),
                            file_name=(
                                f"{selected_report['original_filename']}-summarized-details.pdf"
                            ),
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"structured_details_{selected_report['id']}",
                        )

                    overview = request(
                        "GET", f"/reports/{selected_report['id']}/lab-overview"
                    )
                    if overview and overview["counts"]["values_extracted"] > 0:
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
                            "📄 Download Lab Overview (PDF)",
                            generate_pdf_bytes(f"Lab Overview - {selected_report['original_filename']}", overview.get("summary_markdown", "")),
                            file_name=f"{selected_report['original_filename']}-labs-overview.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )

                    if report_type == "pathology" or "pathology" in detected:
                        pathology = request(
                            "GET", f"/reports/{selected_report['id']}/pathology-overview"
                        )
                        st.subheader("Pathology report")
                        if pathology:
                            if pathology["details"]:
                                st.markdown("#### Specimen and patient details")
                                st.dataframe(
                                    [{"Detail": key, "Information": value}
                                     for key, value in pathology["details"].items()],
                                    use_container_width=True,
                                    hide_index=True,
                                )
                            if pathology["sections"]:
                                for heading, content in pathology["sections"].items():
                                    st.markdown(f"#### {heading}")
                                    st.write(content)
                            else:
                                st.info(
                                    "No standard pathology sections were identified. "
                                    "Review the extracted source text above."
                                )
                            st.caption(pathology["disclaimer"])
                            st.download_button(
                                "Download structured pathology report (PDF)",
                                generate_pdf_bytes(
                                    f"Pathology Report - {selected_report['original_filename']}",
                                    pathology["download_text"],
                                ),
                                file_name=(
                                    f"{selected_report['original_filename']}-pathology.pdf"
                                ),
                                mime="application/pdf",
                                use_container_width=True,
                            )

                    if report_type == "radiology" or "radiology" in detected:
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
                            st.download_button(
                                "📄 Download Radiology Report (PDF)",
                                generate_pdf_bytes(f"Radiology Report - {selected_report['original_filename']}", radiology.get("download_text", "")),
                                file_name=f"{selected_report['original_filename']}-radiology.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )

                    if not detected and report_type not in {"pathology", "radiology"}:
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
                st.download_button(
                    "📄 Download Trends Report (PDF)",
                    generate_pdf_bytes(
                        f"Lab Trends - {patient['name']}",
                        format_lab_trends_text(patient["name"], trends),
                    ),
                    file_name=f"{patient['name']}-lab-trends.pdf", mime="application/pdf",
                    use_container_width=True,
                )

if page == "Brain MRI":
    st.subheader("Brain MRI")
    st.warning("MRI output is decision support and requires qualified clinical review.")
    patient = patient_selector(people, "mri_patient")
    if patient:
        detail = request("GET", f"/patient/{patient['id']}") or {}
        scans = detail.get("mri_scans", [])
        mri_file = st.file_uploader(
            "Upload MRI file", key="mri_upload", on_change=clear_mri_selection,
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
                clear_generated_results()
                st.rerun()
        if scans:
            choices = {f"{s['modality']} | {s['status']} | {s['id'][:8]}": s for s in scans}
            selected_scan = st.selectbox(
                "Select stored MRI",
                list(choices),
                index=None,
                placeholder="Choose a stored MRI",
                key="stored_mri",
            )
            if selected_scan is None:
                st.info("Select a stored MRI to load its information.")
                st.stop()
            scan = choices[selected_scan]
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
                "📄 Download MRI Report (PDF)",
                generate_pdf_bytes(f"Brain MRI Report - {patient['name']}", format_mri_text(scan, patient['name'])),
                file_name=f"{patient['name']}-mri-{scan['id'][:8]}.pdf", mime="application/pdf",
                use_container_width=True,
            )
            analysis_supported = (scan.get("metadata_json") or {}).get("format") == "nifti"
            if scan["status"] != "analyzed" and st.button(
                "Run configured MRI analysis", disabled=not analysis_supported,
                help=None if analysis_supported else "3D analysis currently requires NIfTI input.",
            ):
                analyzed = request("POST", "/analyze-mri", params={"mri_id": scan["id"]})
                if analyzed:
                    st.success("MRI analysis completed.")
                    clear_generated_results()
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

if page == "Grounded Summary":
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
            st.download_button(
                "📄 Download Grounded Summary (PDF)",
                generate_pdf_bytes(f"Grounded Summary - {patient['name']}", summary["narrative_summary_markdown"]),
                file_name=f"{patient['name']}-grounded-summary.pdf", mime="application/pdf",
                use_container_width=True,
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
                    "📄 Download Full Patient Record (PDF)",
                    generate_pdf_bytes(f"Full Patient Record - {patient['name']}", format_patient_record_text(detail)),
                    file_name=f"{patient['name']}-full-record.pdf", mime="application/pdf",
                )

if page == "AI Chatbot":
    st.subheader("🤖 AI Clinical Chatbot")
    st.caption("Ask any question about the patient's medical history, lab results, diagnoses, medications, or MRI scans.")
    patient = patient_selector(people, "chat_patient")
    if patient:
        patient_id = patient["id"]
        chat_key = f"chat_messages_{patient_id}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = [
                {"role": "assistant", "content": f"Hello! I am your AI Clinical Assistant. Ask me anything about **{patient['name']}**'s uploaded medical records or lab results."}
            ]

        st.markdown("**Quick Questions:**")
        qc1, qc2, qc3, qc4 = st.columns(4)
        quick_query = None
        if qc1.button("📄 Summarize Patient", key="q1", use_container_width=True):
            quick_query = "Summarize this patient's medical records and key findings."
        if qc2.button("💊 Medications & Allergies", key="q2", use_container_width=True):
            quick_query = "What medications and allergies are documented for this patient?"
        if qc3.button("📊 Lab Results", key="q3", use_container_width=True):
            quick_query = "What are the lab results and any abnormal lab values?"
        if qc4.button("🧠 MRI Findings", key="q4", use_container_width=True):
            quick_query = "What are the brain MRI findings and tumor status?"

        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("citations"):
                    with st.expander("📚 Source Citations"):
                        for cite in msg["citations"]:
                            st.caption(f"• **{cite.get('document_id')}**: {cite.get('excerpt')}")

        user_input = st.chat_input(f"Ask a question about {patient['name']}...")
        active_query = user_input or quick_query

        if active_query:
            st.session_state[chat_key].append({"role": "user", "content": active_query})
            with st.chat_message("user"):
                st.write(active_query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing patient records..."):
                    history_payload = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state[chat_key][:-1]
                    ]
                    response = request(
                        "POST",
                        f"/clinical-intel/{patient_id}/chat/query",
                        json={"query": active_query, "chat_history": history_payload},
                    )
                    if response:
                        answer = response.get("answer", "No response generated.")
                        citations = response.get("citations", [])
                        st.write(answer)
                        if citations:
                            with st.expander("📚 Source Citations"):
                                for cite in citations:
                                    st.caption(f"• **{cite.get('document_id')}**: {cite.get('excerpt')}")
                        st.session_state[chat_key].append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                        })
                        st.rerun()
