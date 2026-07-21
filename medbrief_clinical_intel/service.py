"""
Unified MedBriefClinicalService Class.
Provides simple, high-level API methods for easy integration into FastAPI backend endpoints.
"""

import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from medbrief_clinical_intel.config import ClinicalIntelConfig, default_config
from medbrief_clinical_intel.llm.client import QwenLLMClient
from medbrief_clinical_intel.schemas.input_models import PatientRecordInput
from medbrief_clinical_intel.schemas.summary import PatientSummaryResponse
from medbrief_clinical_intel.schemas.timeline import MedicalTimelineResponse
from medbrief_clinical_intel.schemas.chatbot import ChatRequest, ChatResponse
from medbrief_clinical_intel.schemas.medication import MedicationAllergyReport
from medbrief_clinical_intel.schemas.contradiction import ContradictionReport
from medbrief_clinical_intel.schemas.missing_info import MissingInfoChecklist

from medbrief_clinical_intel.modules.summarizer import PatientSummarizer
from medbrief_clinical_intel.modules.timeline import TimelineGenerator
from medbrief_clinical_intel.modules.chatbot import ClinicalChatbot
from medbrief_clinical_intel.modules.medication import MedicationIntelligence
from medbrief_clinical_intel.modules.contradiction import ContradictionDetector
from medbrief_clinical_intel.modules.missing_info import MissingInfoDetector
from medbrief_clinical_intel.modules.multilanguage import LanguageProcessor

logger = logging.getLogger(__name__)


class FullClinicalPipelineResponse(BaseModel):
    """Unified response containing output of all 7 Clinical Intelligence modules."""
    patient_id: str
    summary: PatientSummaryResponse
    timeline: MedicalTimelineResponse
    medications_and_allergies: MedicationAllergyReport
    contradictions: ContradictionReport
    missing_information: MissingInfoChecklist


class MedBriefClinicalService:
    """
    Unified entry-point for the MedBrief AI Clinical Intelligence Layer.
    Designed for seamless FastAPI integration.
    """

    def __init__(self, config: Optional[ClinicalIntelConfig] = None):
        self.config = config or default_config
        self.llm_client = QwenLLMClient(config=self.config)

        # Initialize all 7 clinical intelligence modules
        self.summarizer = PatientSummarizer(llm_client=self.llm_client)
        self.timeline_generator = TimelineGenerator(llm_client=self.llm_client)
        self.chatbot = ClinicalChatbot(llm_client=self.llm_client)
        self.medication_intel = MedicationIntelligence(llm_client=self.llm_client)
        self.contradiction_detector = ContradictionDetector(llm_client=self.llm_client)
        self.missing_info_detector = MissingInfoDetector(llm_client=self.llm_client)
        self.language_processor = LanguageProcessor(llm_client=self.llm_client)

    def summarize(self, record: PatientRecordInput) -> PatientSummaryResponse:
        """Module 1: Generate concise doctor-friendly patient summary."""
        record = self._preprocess_record_languages(record)
        return self.summarizer.generate_summary(record)

    def get_timeline(self, record: PatientRecordInput) -> MedicalTimelineResponse:
        """Module 2: Generate chronological medical timeline."""
        record = self._preprocess_record_languages(record)
        return self.timeline_generator.generate_timeline(record)

    def chat(self, request: ChatRequest, record: PatientRecordInput) -> ChatResponse:
        """Module 3: AI Clinical Chatbot grounded in patient records."""
        record = self._preprocess_record_languages(record)
        return self.chatbot.answer_question(request, record)

    def analyze_medications(self, record: PatientRecordInput) -> MedicationAllergyReport:
        """Module 4: Medication & Allergy Intelligence."""
        record = self._preprocess_record_languages(record)
        return self.medication_intel.analyze_medications_and_allergies(record)

    def detect_contradictions(self, record: PatientRecordInput) -> ContradictionReport:
        """Module 5: Clinical Contradiction Detection."""
        record = self._preprocess_record_languages(record)
        return self.contradiction_detector.detect_contradictions(record)

    def detect_missing_info(self, record: PatientRecordInput) -> MissingInfoChecklist:
        """Module 6: Missing Information Detection."""
        record = self._preprocess_record_languages(record)
        return self.missing_info_detector.detect_missing_information(record)

    def process_multilanguage(self, raw_text: str):
        """Module 7: Multi-language Support (Hindi/Hinglish to English translation)."""
        return self.language_processor.process_and_translate(raw_text)

    def run_full_clinical_pipeline(self, record: PatientRecordInput) -> FullClinicalPipelineResponse:
        """Execute all clinical intelligence modules in a single call."""
        record = self._preprocess_record_languages(record)
        
        summary = self.summarizer.generate_summary(record)
        timeline = self.timeline_generator.generate_timeline(record)
        meds = self.medication_intel.analyze_medications_and_allergies(record)
        contradictions = self.contradiction_detector.detect_contradictions(record)
        missing_info = self.missing_info_detector.detect_missing_information(record)

        return FullClinicalPipelineResponse(
            patient_id=record.patient_id,
            summary=summary,
            timeline=timeline,
            medications_and_allergies=meds,
            contradictions=contradictions,
            missing_information=missing_info
        )

    def _preprocess_record_languages(self, record: PatientRecordInput) -> PatientRecordInput:
        """Preprocess documents in Hindi/Hinglish to English preserving medical terms."""
        new_docs = []
        for doc in record.documents:
            translated_text, lang = self.language_processor.process_and_translate(doc.extracted_text)
            doc_copy = doc.model_copy()
            doc_copy.extracted_text = translated_text
            doc_copy.language = lang
            new_docs.append(doc_copy)
        
        record_copy = record.model_copy()
        record_copy.documents = new_docs
        return record_copy


