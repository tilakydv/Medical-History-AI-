"""
Feature 7: Multi-language Support Module (Hindi, English, Mixed English-Hindi / Hinglish).
Translates clinical reports into English before processing while preserving medical terminology.
"""

import re
from typing import Optional, Tuple, List
from medbrief_clinical_intel.llm.client import QwenLLMClient
from medbrief_clinical_intel.llm.prompt_templates import SystemPrompts, ClinicalPromptBuilder


class LanguageProcessor:
    """Module 7: Multi-language Processor for Hindi & Hinglish clinical texts."""

    # Common Hindi & Hinglish medical dictionary mapping to preserve clinical sense
    MEDICAL_TERM_PRESERVE_MAP = {
        "sir dard": "headache (sir dard)",
        "sir me dard": "headache (sir dard)",
        "sugar ki dabai": "anti-diabetic medication (sugar ki dabai)",
        "sugar test": "blood glucose test (sugar test)",
        "bp ki goli": "anti-hypertensive medication (BP ki goli)",
        "raktapitta": "hemorrhage / bleeding",
        "madhumeh": "Diabetes Mellitus (madhumeh)",
        "uchha raktachap": "Hypertension (uchha raktachap)",
        "jwar": "fever (jwar)",
        "chakar": "dizziness / vertigo (chakar)",
    }

    def __init__(self, llm_client: Optional[QwenLLMClient] = None):
        self.llm_client = llm_client or QwenLLMClient()

    def process_and_translate(self, raw_text: str) -> Tuple[str, str]:
        """
        Process raw report text, detect language, and translate if needed.
        Returns:
            Tuple[translated_text, detected_language]
        """
        detected_lang = self.detect_language(raw_text)

        if detected_lang == "en":
            return raw_text, "en"

        # Apply term protection pre-processing
        preprocessed_text = self._protect_medical_terms(raw_text)

        prompt = ClinicalPromptBuilder.build_translation_prompt(preprocessed_text)
        translated_output = self.llm_client.generate(
            prompt, system_prompt=SystemPrompts.TRANSLATION_SYSTEM
        )

        translated_text = self._extract_translated_content(translated_output)
        return translated_text, detected_lang

    def detect_language(self, text: str) -> str:
        """Detect whether text is English ('en'), Hindi Devanagari ('hi'), or Mixed ('mixed')."""
        # Check Devanagari Unicode range \u0900-\u097F
        devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
        total_chars = len(text.strip())

        if total_chars == 0:
            return "en"

        devanagari_ratio = devanagari_chars / total_chars

        if devanagari_ratio > 0.3:
            return "hi"
        elif devanagari_ratio > 0.05 or any(term in text.lower() for term in self.MEDICAL_TERM_PRESERVE_MAP):
            return "mixed"
        
        return "en"

    def _protect_medical_terms(self, text: str) -> str:
        """Annotate Hinglish terms to protect medical meaning prior to translation."""
        processed = text
        for term, replacement in self.MEDICAL_TERM_PRESERVE_MAP.items():
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            processed = pattern.sub(replacement, processed)
        return processed

    def _extract_translated_content(self, llm_output: str) -> str:
        """Clean LLM output format if structured headers are returned."""
        if "Translated Text:" in llm_output:
            return llm_output.split("Translated Text:")[-1].strip()
        return llm_output.strip()


