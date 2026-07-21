"""
Schema for AI Patient Summarization (Module 1).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PatientSummaryResponse(BaseModel):
    """Concise, clinician-grade patient summary output."""
    patient_id: str = Field(..., description="Patient Identifier")
    patient_overview: str = Field(..., description="Brief demographic and primary clinical background summary")
    medical_history: List[str] = Field(default_factory=list, description="Historical illnesses, conditions, surgeries")
    diagnoses: List[str] = Field(default_factory=list, description="Confirmed and working diagnoses")
    current_conditions: List[str] = Field(default_factory=list, description="Active health conditions and symptoms")
    mri_findings_summary: Optional[str] = Field(default=None, description="Structured summary of Brain MRI segmentation and imaging findings")
    laboratory_observations: List[str] = Field(default_factory=list, description="Key lab values, abnormal findings, trends")
    current_medications: List[str] = Field(default_factory=list, description="Active prescriptions and dosages")
    allergies: List[str] = Field(default_factory=list, description="Drug, food, or environmental allergies recorded")
    recommended_follow_up: List[str] = Field(default_factory=list, description="Follow-ups strictly documented in uploaded reports")
    narrative_summary_markdown: str = Field(..., description="Formatted markdown doctor-friendly brief")


