"""
Clinically aligned system prompts and user prompt templates for Qwen2.5-14B-Instruct.
"""

from typing import Optional, Dict, Any, List
import json


class SystemPrompts:
    CLINICAL_INTEL_SYSTEM = (
        "You are MedBrief AI Clinical Intelligence Module, powered by Qwen2.5-14B-Instruct. "
        "You serve as an expert clinical assistant for physicians, radiologists, and specialists. "
        "CRITICAL DIRECTIVES:\n"
        "1. ACCURACY & ZERO HALLUCINATION: Rely STRICTLY AND ONLY on the provided patient records and structured MRI findings. "
        "Do NOT invent, infer, or hallucinate diagnoses, medications, dates, or laboratory results that are not documented.\n"
        "2. CLINICAL PRECISION: Use clear, concise medical terminology suitable for physician review.\n"
        "3. STRUCTURED OUTPUT: Always return valid, well-formed JSON matching the exact specified schema whenever JSON is requested."
    )

    TRANSLATION_SYSTEM = (
        "You are an expert bilingual medical translator specializing in English, Hindi, and mixed Hindi-English (Hinglish) clinical documents. "
        "Translate Hindi/Hinglish text into clear clinical English. "
        "CRITICAL REQUIREMENT: PRESERVE ALL MEDICAL TERMINOLOGY, drug names, anatomical terms, dosages, and numerical findings accurately. "
        "Do not translate medical terms into generic Hindi words if the clinical term is standard (e.g. keep 'Stent', 'MRI', 'Glioma', 'Diabetes', 'HbA1c')."
    )

    CHATBOT_RAG_SYSTEM = (
        "You are MedBrief AI Grounded Clinical Assistant. "
        "You answer questions exclusively using the uploaded patient records provided in the context.\n"
        "RULES:\n"
        "1. If the requested information is present in the context, provide a precise clinical answer with exact document citations.\n"
        "2. If the requested information IS NOT present in the provided records, explicitly state: "
        "'Information not found in uploaded patient records.' DO NOT speculate or bring in outside medical knowledge for this specific patient."
    )


class ClinicalPromptBuilder:
    """Helper to assemble prompts for each clinical module."""

    @staticmethod
    def build_summarizer_prompt(context_text: str, json_schema_str: str) -> str:
        return f"""Task: Generate a concise, doctor-friendly patient summary based strictly on the uploaded records below.

Patient Records & Structured MRI Findings:
---
{context_text}
---

Your summary must include:
1. Patient Overview
2. Medical History
3. Diagnoses
4. Current Conditions
5. Brain MRI Findings (incorporate structured findings if available)
6. Laboratory Observations
7. Current Medications
8. Allergies
9. Recommended Follow-up (ONLY if explicitly mentioned in uploaded reports)

Return your response strictly in JSON matching the following schema:
```json
{json_schema_str}
```
"""

    @staticmethod
    def build_timeline_prompt(context_text: str, json_schema_str: str) -> str:
        return f"""Task: Extract all chronological medical events from the uploaded patient records and sort them by date (ascending).

Patient Records:
---
{context_text}
---

For each event, extract:
- date_or_year (e.g. '2019', '2022-04-10')
- iso_date (YYYY-MM-DD format if possible for sorting, or YYYY-01-01 if year only)
- category (Diagnosis, Surgery, Imaging, Lab, Medication, Admission, General)
- title (e.g. 'Diabetes diagnosed', 'Brain MRI performed', 'Tumor detected')
- description (clinical detail)
- source_document

Return your response strictly in JSON matching the schema:
```json
{json_schema_str}
```
"""

    @staticmethod
    def build_medication_allergy_prompt(context_text: str, json_schema_str: str) -> str:
        return f"""Task: Extract all medications (current and previous) and allergies (drug, food, environmental) from the patient records.

Patient Records:
---
{context_text}
---

Return your output strictly in JSON matching the schema:
```json
{json_schema_str}
```
"""

    @staticmethod
    def build_contradiction_prompt(context_text: str, json_schema_str: str) -> str:
        return f"""Task: Compare the multi-date/multi-source patient records below and detect any clinical contradictions or conflicting statements.

Examples of conflicts to identify:
- Mismatched allergies across different dates/reports
- Mismatched medication dosages or orders
- Conflicting diagnosis dates or onset years
- Conflicting surgical history details

Patient Records:
---
{context_text}
---

Return your analysis strictly in JSON matching the schema:
```json
{json_schema_str}
```
"""

    @staticmethod
    def build_missing_info_prompt(context_text: str, json_schema_str: str) -> str:
        return f"""Task: Analyze the patient records below and identify clinically important information that is missing or incomplete.

Examples to check for:
- Missing baseline or previous MRI reports
- Missing family history
- Missing smoking / tobacco history
- Missing recent blood pressure or vitals
- Missing recent HbA1c or key laboratory parameters

Patient Records:
---
{context_text}
---

Return your evaluation strictly in JSON matching the schema:
```json
{json_schema_str}
```
"""

    @staticmethod
    def build_translation_prompt(raw_text: str) -> str:
        return f"""Task: Detect language and translate the following clinical record into standard English while preserving all medical terms, medication names, lab units, and anatomical names.

Raw Text:
---
{raw_text}
---

Output format:
Language Detected: [English / Hindi / Mixed English-Hindi]
Translated Text:
[Translated text preserving medical terminology]
"""

    @staticmethod
    def build_rag_chat_prompt(context_text: str, query: str, history_str: str = "") -> str:
        return f"""Context - Uploaded Patient Records & MRI Findings:
---
{context_text}
---

Previous Conversation:
{history_str if history_str else "None"}

Doctor Query: {query}

Instructions:
Answer the doctor's query using ONLY the provided patient records context above. 
Include source citations (e.g. [Document ID / Date]) where applicable. 
If the records do NOT contain the answer, reply: "Information not found in uploaded patient records."
"""


