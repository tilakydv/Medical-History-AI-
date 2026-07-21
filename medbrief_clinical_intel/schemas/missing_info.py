"""
Schema for Missing Information Detector (Module 6).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class MissingInfoItem(BaseModel):
    """Specific clinically important piece of missing information."""
    field_name: str = Field(..., description="Key clinical field missing (e.g., 'Baseline MRI', 'Family History', 'Smoking History', 'Blood Pressure', 'HbA1c')")
    category: str = Field(..., description="Category: 'Imaging History', 'Social/Family History', 'Vitals', 'Lab Parameters', 'Prior Surgeries'")
    clinical_significance: str = Field(..., description="Why this missing information is clinically relevant for patient care")
    status: str = Field(default="Missing", description="Status: 'Missing', 'Incomplete', 'Unconfirmed'")


class MissingInfoChecklist(BaseModel):
    """Structured checklist of missing patient information."""
    patient_id: str
    missing_items_count: int
    items: List[MissingInfoItem] = Field(default_factory=list, description="Checklist of missing elements")
    recommendations_text: str = Field(..., description="Actionable summary of missing information to request from patient/provider")


