import re
from typing import Optional, List
from medbrief_clinical_intel.schemas.input_models import PatientRecordInput
from medbrief_clinical_intel.schemas.chatbot import ChatRequest, ChatResponse, Citation
from medbrief_clinical_intel.llm.client import QwenLLMClient
from medbrief_clinical_intel.llm.prompt_templates import SystemPrompts, ClinicalPromptBuilder


class ClinicalChatbot:
    """Module 3: AI Clinical RAG Chatbot strictly grounded in patient records."""

    def __init__(self, llm_client: Optional[QwenLLMClient] = None):
        self.llm_client = llm_client or QwenLLMClient()

    def answer_question(
        self,
        request: ChatRequest,
        patient_record: PatientRecordInput
    ) -> ChatResponse:
        """Answer clinical question strictly using patient record context."""
        if self.llm_client.is_loaded:
            context_str = self._format_context(patient_record)
            history_str = ""
            if request.chat_history:
                history_str = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in request.chat_history])

            prompt = ClinicalPromptBuilder.build_rag_chat_prompt(
                context_text=context_str,
                query=request.query,
                history_str=history_str
            )

            response_text = self.llm_client.generate(
                prompt, system_prompt=SystemPrompts.CHATBOT_RAG_SYSTEM
            )

            not_found = (
                "information not found" in response_text.lower() or
                "not documented" in response_text.lower() or
                "not present in" in response_text.lower()
            )
            citations = self._extract_citations(patient_record)

            return ChatResponse(
                patient_id=patient_record.patient_id,
                query=request.query,
                answer=response_text,
                is_grounded=True,
                not_found_in_records=not_found,
                citations=citations if not not_found else []
            )

        # High-precision grounded search & extraction engine
        answer_text, citations, not_found = self._extract_accurate_answer(request.query, patient_record)

        return ChatResponse(
            patient_id=patient_record.patient_id,
            query=request.query,
            answer=answer_text,
            is_grounded=True,
            not_found_in_records=not_found,
            citations=citations
        )

    def _extract_accurate_answer(
        self, query: str, record: PatientRecordInput
    ) -> tuple[str, List[Citation], bool]:
        """Grounded text extraction engine that searches patient records for exact answers."""
        q_raw = query.strip()
        q_lower = q_raw.lower()
        p_name = record.patient_name or f"Patient {record.patient_id}"
        docs = record.documents or []
        mri = record.mri_findings

        # 1. Greetings
        if re.search(r"\b(hello|hi|hey|greetings|good\s+morning|good\s+afternoon)\b", q_lower) and len(q_lower.split()) <= 4:
            return (
                f"Hello! I am your AI Clinical Assistant. Ask me any question about **{p_name}**'s uploaded medical records, lab results, diagnoses, or MRI scans.",
                [],
                False
            )

        # 2. Patient Demographics & Identification
        if any(term in q_lower for term in ["who is", "name", "age", "gender", "dob", "date of birth", "patient details"]):
            demo_parts = [f"**Patient Name:** {p_name}", f"**Patient ID:** `{record.patient_id}`"]
            if record.age:
                demo_parts.append(f"**Age:** {record.age}")
            if record.gender:
                demo_parts.append(f"**Sex:** {record.gender}")
            demo_parts.append(f"**Uploaded Reports:** {len(docs)}")
            demo_parts.append(f"**MRI Scans:** {'1 stored' if mri else 'None stored'}")
            return ("\n".join(demo_parts), self._extract_citations(record), False)

        # 3. Brain MRI / Imaging Specific Query
        if any(term in q_lower for term in ["mri", "brain", "scan", "tumor", "lesion", "segmentation", "imaging", "edema"]):
            if mri:
                mri_text = (
                    f"### Brain MRI Segmentation Findings ({p_name})\n\n"
                    f"• **Tumor/Lesion Detected:** {'Yes' if mri.tumor_detected else 'No'}\n"
                    f"• **Classification:** {mri.tumor_type or 'Not specified'}\n"
                    f"• **Location:** {mri.location or 'Not specified'}\n"
                    f"• **Volume:** {mri.volume_cm3 or 'N/A'} cm³ ({mri.volume_mm3 or 'N/A'} mm³)\n"
                    f"• **Peritumoral Edema:** {'Present' if mri.edema_present else 'None'}\n"
                    f"• **Midline Shift:** {'Present' if mri.midline_shift else 'None'}\n"
                    f"• **Notes:** {mri.additional_notes or 'None'}"
                )
                cite = [Citation(document_id="Brain_MRI_Segmentation", excerpt="MONAI/nnU-Net Brain MRI segmentation report")]
                return (mri_text, cite, False)

        # 4. Summarize Patient / Overall Issues / Key Findings / Problems / Report Content
        if any(term in q_lower for term in ["summarize", "summary", "issue", "issues", "problem", "diagnosis", "diagnoses", "overview", "what is wrong", "report", "in report", "their in report", "happened", "happening", "status", "condition", "history", "finding", "findings", "result", "results", "tell me"]):
            if not docs and not mri:
                return (
                    f"No medical reports or MRI scans have been uploaded for **{p_name}** yet. "
                    "Please upload a report in the **Reports & Labs** tab and click **Extract selected report** to analyze findings.",
                    [],
                    True
                )

            sections = [f"### Clinical Report Summary for {p_name}"]
            if mri:
                sections.append(
                    f"**Brain MRI:** {'Tumor detected' if mri.tumor_detected else 'No tumor detected'} "
                    f"in {mri.location or 'brain'} ({mri.tumor_type or 'Lesion'})."
                )

            if docs:
                for doc in docs:
                    summary_block = self._format_report_clinical_summary(doc.extracted_text)
                    sections.append(f"**Report ({doc.document_type}):**\n{summary_block}")

            return ("\n\n".join(sections), self._extract_citations(record), False)

        # 5. Medications & Allergies Query
        if any(term in q_lower for term in ["medication", "medications", "drug", "drugs", "prescription", "allergy", "allergies", "penicillin", "dose"]):
            med_lines = []
            citations = []
            for doc in docs:
                for line in doc.extracted_text.splitlines():
                    l_str = line.strip()
                    l_low = l_str.lower()
                    if any(k in l_low for k in ["medication", "tab", "mg", "capsule", "syrup", "daily", "bid", "tid", "qd", "allergy", "allergic", "penicillin", "rx"]):
                        med_lines.append(f"• {l_str}")
                        citations.append(Citation(document_id=doc.document_id, excerpt=l_str[:120]))

            if med_lines:
                return (
                    f"### Documented Medications & Allergies ({p_name})\n\n" + "\n".join(med_lines[:12]),
                    citations[:4],
                    False
                )
            return (
                f"No specific medications or drug allergies are documented in the uploaded records for **{p_name}**.",
                [],
                True
            )

        # 6. Specific Keyword / Document Search across all uploaded files
        stop_words = {"what", "is", "in", "their", "there", "the", "report", "file", "document", "of", "for", "and", "a", "an", "does", "have", "has", "ronak", "bhavya", "patient", "show", "tell", "me", "about", "saying", "says", "to"}
        keywords = [w for w in re.findall(r"\w+", q_lower) if w not in stop_words and len(w) >= 3]

        if docs and keywords:
            matching_lines = []
            citations = []
            for doc in docs:
                for line in doc.extracted_text.splitlines():
                    l_str = line.strip()
                    l_low = l_str.lower()
                    if any(kw in l_low for kw in keywords):
                        matching_lines.append(f"• {l_str}")
                        citations.append(Citation(document_id=doc.document_id, excerpt=l_str[:120]))

            if matching_lines:
                return (
                    f"Based on **{p_name}**'s uploaded records:\n\n" + "\n".join(matching_lines[:10]) +
                    "\n\n*All findings above are extracted directly from the stored patient records.*",
                    citations[:4],
                    False
                )

        # 7. Fallback when docs are present
        if docs:
            sections = [f"### Report Summary for {p_name}"]
            for doc in docs:
                summary_block = self._format_report_clinical_summary(doc.extracted_text)
                sections.append(f"**Extracted Clinical Content ({doc.document_type}):**\n{summary_block}")

            return ("\n\n".join(sections), self._extract_citations(record), False)

        return (
            f"No medical reports or lab files have been extracted for **{p_name}** yet. "
            "Please upload a report in the **Reports & Labs** tab and click **Extract selected report** to analyze findings.",
            [],
            True
        )

    def _format_report_clinical_summary(self, extracted_text: str) -> str:
        """Parses laboratory interpretation, recommendations, and abnormal values from extracted report text."""
        lines = [l.strip() for l in extracted_text.splitlines() if l.strip()]

        interp = []
        in_interp = False
        recom = []
        in_recom = False

        for line in lines:
            l_low = line.lower()
            if any(k in l_low for k in ["laboratory interpretation", "clinical impression", "impression:", "conclusion"]):
                in_interp = True
                in_recom = False
                continue
            elif any(k in l_low for k in ["recommendation", "plan:"]):
                in_recom = True
                in_interp = False
                continue
            elif any(k in l_low for k in ["verified by", "consultant", "signature", "dr."]) and len(line) < 60:
                in_interp = False
                in_recom = False

            if in_interp:
                interp.append(line)
            elif in_recom:
                recom.append(line)

        lab_findings = []
        for line in lines:
            l_low = line.lower()
            if any(flag in l_low for flag in ["critical high", "very high", "high", "low", "severely reduced", "critical low", "abnormal"]):
                if not any(hk in l_low for hk in ["patient name", "department", "laboratory report", "doctor", "referring physician"]):
                    lab_findings.append(line)

        formatted_sections = []
        if interp:
            formatted_sections.append(f"**📋 Clinical Interpretation & Diagnosis:**\n" + " ".join(interp))
        if recom:
            formatted_sections.append(f"**🩺 Clinical Recommendations:**\n" + " ".join(recom))
        if lab_findings:
            bullets = "\n".join([f"  • {item}" for item in lab_findings[:15]])
            formatted_sections.append(f"**⚠️ Key Laboratory Observations & Flags:**\n{bullets}")

        if formatted_sections:
            return "\n\n".join(formatted_sections)

        filtered = self._extract_clinical_findings_text(extracted_text)
        return "\n".join([f"  • {item}" for item in filtered[:12]])

    def _extract_clinical_findings_text(self, extracted_text: str) -> List[str]:
        """Strips out standard hospital letterhead noise and patient metadata lines."""
        header_keywords = [
            "department of", "diagnostic laboratories", "clinical pathology",
            "laboratory report", "patient name", "patient id", "referred by",
            "hospital", "doctor name", "sample collected", "page 1 of", "page 2 of",
            "printed on", "consultant pathologist", "signature", "end of report",
            "collection date", "referring physician", "date of birth", "age / sex",
            "external id", "hospital id", "mc-2026", "mc-", "jonathan michael",
            "carter", "harrison", "report date"
        ]
        clinical_lines = []
        for line in extracted_text.splitlines():
            l_str = line.strip()
            if not l_str or l_str.startswith("---") or len(l_str) < 3:
                continue
            l_low = l_str.lower()
            if any(hk in l_low for hk in header_keywords):
                continue
            clinical_lines.append(l_str)
        return clinical_lines

    def _format_context(self, record: PatientRecordInput) -> str:
        parts = [
            f"Patient ID: {record.patient_id}",
            f"Age: {record.age or 'Unspecified'}, Gender: {record.gender or 'Unspecified'}"
        ]

        if record.mri_findings:
            mri = record.mri_findings
            parts.append(
                f"\n--- Brain MRI Segmentation Findings ---\n"
                f"Tumor Detected: {mri.tumor_detected}\n"
                f"Type: {mri.tumor_type or 'N/A'}\n"
                f"Location: {mri.location or 'N/A'}\n"
                f"Volume: {mri.volume_cm3} cm³ ({mri.volume_mm3} mm³)\n"
                f"Edema: {mri.edema_present}, Midline Shift: {mri.midline_shift}\n"
                f"Notes: {mri.additional_notes or 'None'}"
            )

        if record.documents:
            parts.append("\n--- Extracted Uploaded Patient Documents ---")
            for doc in record.documents:
                parts.append(
                    f"\n[Document ID: {doc.document_id} | Date: {doc.date or 'N/A'} | Type: {doc.document_type}]\n"
                    f"{doc.extracted_text}"
                )

        return "\n".join(parts)

    def _extract_citations(self, record: PatientRecordInput) -> List[Citation]:
        citations = []
        if record.mri_findings:
            citations.append(Citation(document_id="Brain_MRI_Segmentation", excerpt="MONAI/nnU-Net Brain MRI findings"))
        for doc in record.documents:
            excerpt_snippet = doc.extracted_text[:150].replace("\n", " ") + "..." if len(doc.extracted_text) > 150 else doc.extracted_text
            citations.append(Citation(document_id=doc.document_id, excerpt=excerpt_snippet))
        return citations



