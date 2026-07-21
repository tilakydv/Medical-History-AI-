"""
Qwen2.5-14B-Instruct LLM Client interface with HuggingFace Transformers and Mock support.
"""

import os
import re
import logging
from typing import Optional, Dict, Any, Union
from medbrief_clinical_intel.config import ClinicalIntelConfig, default_config
from medbrief_clinical_intel.llm.prompt_templates import SystemPrompts

logger = logging.getLogger(__name__)


class QwenLLMClient:
    """LLM client wrapper for Qwen2.5-14B-Instruct."""

    def __init__(self, config: Optional[ClinicalIntelConfig] = None):
        self.config = config or default_config
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.is_loaded = False

        if not self.config.use_mock_llm:
            self._try_load_model()
        else:
            logger.info("QwenLLMClient running in MOCK mode for development/testing.")

    def _try_load_model(self):
        """Load Qwen2.5-14B model using HuggingFace transformers if available."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            logger.info(f"Loading Qwen model from {self.config.model_name_or_path}...")
            
            tokenizer_kwargs = {"trust_remote_code": True}
            model_kwargs = {"trust_remote_code": True, "device_map": self.config.device}

            if self.config.load_in_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            elif self.config.load_in_8bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path, **tokenizer_kwargs
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path, **model_kwargs
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
            )
            self.is_loaded = True
            logger.info("Qwen2.5 model successfully loaded into memory.")
        except Exception as e:
            logger.warning(
                f"Failed to load HF model '{self.config.model_name_or_path}': {e}. "
                "Falling back to Mock execution mode."
            )
            self.is_loaded = False
            self.config.use_mock_llm = True

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Generate text completion from Qwen2.5-14B-Instruct model."""
        sys_prompt = system_prompt or SystemPrompts.CLINICAL_INTEL_SYSTEM

        if self.is_loaded and self.pipeline is not None:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Format using Qwen chat template
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            outputs = self.pipeline(
                prompt_text,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                temperature=temperature if temperature is not None else self.config.temperature
            )
            generated_text = outputs[0]["generated_text"]
            # Extract assistant response
            if "<|im_start|>assistant\n" in generated_text:
                return generated_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
            return generated_text[len(prompt_text):].strip()

        return self._generate_mock_response(prompt, sys_prompt)

    def _generate_mock_response(self, prompt: str, system_prompt: str) -> str:
        """Deterministic mock response generator for testing without full 14B weights."""
        logger.debug("Generating mock LLM completion.")
        # Determine intent from prompt text
        if "patient summary" in prompt.lower():
            return """```json
{
  "patient_id": "P-EXTRACTED",
  "patient_overview": "Patient record summary generated from uploaded clinical files.",
  "medical_history": ["Type 2 Diabetes mellitus", "Essential Hypertension", "Prior lumbar surgery"],
  "diagnoses": ["Brain Tumor / Intra-axial Lesion", "Controlled Type 2 Diabetes"],
  "current_conditions": ["Headaches", "Mild progressive neurological signs"],
  "mri_findings_summary": "Brain MRI reveals a well-defined lesion in the Left Frontal Lobe with peritumoral edema.",
  "laboratory_observations": ["HbA1c: 7.2%", "Fasting Blood Glucose: 135 mg/dL", "Normal renal function"],
  "current_medications": ["Metformin 500mg BID", "Amlodipine 5mg QD"],
  "allergies": ["Penicillin (Rash)"],
  "recommended_follow_up": ["Neurosurgical consultation", "Follow-up MRI in 3 months"],
  "narrative_summary_markdown": "### Patient Summary Brief\\n\\n**Overview:** Patient with history of Diabetes and Hypertension presenting with recent radiological findings.\\n\\n**MRI Summary:** Left Frontal Lobe lesion identified with associated edema.\\n\\n**Plan:** Neurosurgery follow-up."
}
```"""
        elif "chronological medical events" in prompt.lower():
            return """```json
{
  "patient_id": "P-EXTRACTED",
  "timeline": [
    {
      "date_or_year": "2019",
      "iso_date": "2019-01-01",
      "category": "Diagnosis",
      "title": "Type 2 Diabetes Diagnosed",
      "description": "Initial diagnosis of Type 2 Diabetes Mellitus based on elevated HbA1c.",
      "source_document": "History_Report_2019.pdf"
    },
    {
      "date_or_year": "2020",
      "iso_date": "2020-05-15",
      "category": "Diagnosis",
      "title": "Hypertension Diagnosed",
      "description": "Hypertension recorded and started on anti-hypertensive therapy.",
      "source_document": "Clinic_Note_2020.pdf"
    },
    {
      "date_or_year": "2022",
      "iso_date": "2022-08-10",
      "category": "Surgery",
      "title": "Lumbar Spine Surgery",
      "description": "Elective lumbar microdiscectomy performed without complications.",
      "source_document": "Surgical_Summary_2022.pdf"
    },
    {
      "date_or_year": "2024",
      "iso_date": "2024-03-20",
      "category": "Imaging",
      "title": "Brain MRI Performed",
      "description": "Baseline Brain MRI requested due to recurrent headaches.",
      "source_document": "MRI_Report_2024.pdf"
    },
    {
      "date_or_year": "2026",
      "iso_date": "2026-02-14",
      "category": "Diagnosis",
      "title": "Brain Tumor Detected",
      "description": "nnU-Net Brain MRI segmentation confirmed presence of Left Frontal Lobe tumor.",
      "source_document": "MRI_Segmentation_2026.json"
    }
  ],
  "formatted_timeline_text": "2019 - Diabetes diagnosed\\n2020 - Hypertension\\n2022 - Surgery\\n2024 - Brain MRI\\n2026 - Tumor detected"
}
```"""
        elif "medications" in prompt.lower() and "allergies" in prompt.lower():
            return """```json
{
  "patient_id": "P-EXTRACTED",
  "current_medications": [
    {"name": "Metformin", "dosage": "500 mg", "frequency": "BID (Twice daily)", "duration": "Ongoing", "status": "Current"}
  ],
  "previous_medications": [
    {"name": "Glibenclamide", "dosage": "5 mg", "frequency": "Once daily", "duration": "Discontinued 2021", "status": "Previous"}
  ],
  "drug_allergies": [
    {"allergen": "Penicillin", "allergy_type": "Drug", "reaction": "Skin rash and hives", "severity": "Moderate"}
  ],
  "food_allergies": [],
  "environmental_allergies": [
    {"allergen": "Pollen", "allergy_type": "Environmental", "reaction": "Allergic Rhinitis", "severity": "Mild"}
  ],
  "summary_text": "Current Meds: Metformin 500mg BID. Drug Allergies: Penicillin."
}
```"""
        elif "clinical contradictions" in prompt.lower():
            return """```json
{
  "patient_id": "P-EXTRACTED",
  "contradictions_found": true,
  "contradictions": [
    {
      "category": "Allergy Mismatch",
      "description": "Doc 2020 states 'No Known Drug Allergies (NKDA)', whereas Doc 2024 states severe Penicillin allergy.",
      "record_a_source": "Clinic_Note_2020.pdf",
      "record_a_statement": "NKDA",
      "record_b_source": "Hospital_Admission_2024.pdf",
      "record_b_statement": "Penicillin allergy documented",
      "severity": "High",
      "requires_clinician_review": true
    }
  ],
  "summary_text": "Attention: 1 clinical contradiction identified regarding Penicillin allergy documentation across reports."
}
```"""
        elif "missing or incomplete" in prompt.lower():
            return """```json
{
  "patient_id": "P-EXTRACTED",
  "missing_items_count": 3,
  "items": [
    {
      "field_name": "Previous Baseline MRI",
      "category": "Imaging History",
      "clinical_significance": "Essential to evaluate tumor progression rate against past neuroimaging.",
      "status": "Missing"
    },
    {
      "field_name": "Smoking / Tobacco History",
      "category": "Social/Family History",
      "clinical_significance": "Relevant for vascular and peri-operative risk assessment.",
      "status": "Missing"
    },
    {
      "field_name": "Recent HbA1c",
      "category": "Lab Parameters",
      "clinical_significance": "Crucial to monitor glycemic control prior to steroid therapy.",
      "status": "Missing"
    }
  ],
  "recommendations_text": "Recommend requesting previous baseline MRI, tobacco history, and recent HbA1c lab work."
}
```"""
        elif "translate" in prompt.lower():
            return "Language Detected: Mixed English-Hindi\nTranslated Text:\nPatient has diabetes for 5 years. Takes sugar tablet Dabai (Metformin 500mg). Complaining of head pain (sir dard) for 2 weeks. Brain MRI shows tumor."
        else:
            query_match = re.search(r"Question:\s*(.+)", prompt, re.IGNORECASE)
            user_query = query_match.group(1).strip() if query_match else prompt
            query_lower = user_query.lower()

            if re.search(r"\b(hello|hi|hey|greetings|good\s+morning|good\s+afternoon)\b", query_lower) and len(query_lower.split()) <= 4:
                return "Hello! I am your AI Clinical Assistant. I can answer questions about the patient's uploaded medical records, laboratory values, and MRI scans. How can I help you today?"

            if "document id:" in prompt.lower() or "extracted uploaded patient documents" in prompt.lower():
                doc_lines = []
                in_doc = False
                for line in prompt.splitlines():
                    if "[document id:" in line.lower():
                        in_doc = True
                        continue
                    if in_doc and line.strip() and not line.startswith("---"):
                        doc_lines.append(line.strip())

                if doc_lines:
                    text_snippet = "\n• ".join(doc_lines[:10])
                    return f"Based on the patient's uploaded medical records:\n\n• {text_snippet}\n\nAll findings above are grounded directly in the patient's uploaded records."

                return (
                    "Based on the patient's uploaded records:\n"
                    "- The stored clinical and laboratory documents have been processed.\n"
                    "- Relevant findings and values are recorded in the system.\n\n"
                    "Please ask any specific question regarding lab trends, diagnoses, medications, or MRI findings."
                )

            return (
                "No medical reports or lab files have been extracted for this patient yet. "
                "Please upload a report in the 'Reports & Labs' tab and click 'Extract selected report' to analyze findings."
            )

