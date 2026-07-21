"""
Schema for Chronological Medical Timeline (Module 2).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class TimelineEvent(BaseModel):
    """Individual event in patient chronological medical history."""
    date_or_year: str = Field(..., description="Extracted date or year (e.g. '2019', '2024-05-12', 'Unknown Date')")
    iso_date: Optional[str] = Field(default=None, description="Normalized ISO date string YYYY-MM-DD for sorting")
    category: str = Field(..., description="Category: Diagnosis, Surgery, Imaging, Lab, Medication, Admission, General")
    title: str = Field(..., description="Short title of event (e.g., 'Diabetes diagnosed', 'Tumor detected')")
    description: str = Field(..., description="Clinically relevant details of the event")
    source_document: Optional[str] = Field(default=None, description="Reference document ID or source file")


class MedicalTimelineResponse(BaseModel):
    """Chronologically sorted timeline of patient records."""
    patient_id: str
    timeline: List[TimelineEvent] = Field(default_factory=list, description="Events ordered chronologically")
    formatted_timeline_text: str = Field(..., description="Formatted readable timeline summary text")


