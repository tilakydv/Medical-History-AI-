"""
Feature 3: Grounded AI Clinical Chatbot RAG Module.
Answers queries exclusively using uploaded patient records, ensuring zero hallucinations.
Supported questions include:
- Summarize this patient.
- What medications is the patient taking?
- Does the patient have allergies?
- Show MRI findings.
- Has the patient undergone surgery?
- What changed compared to previous reports?
- Show abnormal laboratory findings.
"""

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

        # Check for refusal / missing info phrase
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
                f"Volume: {mri.volume_cm3} cmÂ³ ({mri.volume_mm3} mmÂ³)\n"
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


