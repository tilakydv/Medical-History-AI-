"""
Schema for Clinical Contradiction Detection (Module 5).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ContradictionItem(BaseModel):
    """Specific detected clinical contradiction across records."""
    category: str = Field(..., description="Category: 'Allergy Mismatch', 'Medication Mismatch', 'Diagnosis Date Conflict', 'Surgical History Conflict', 'Lab Trend Conflict'")
    description: str = Field(..., description="Clear explanation of the detected conflict")
    record_a_source: str = Field(..., description="First document ID / date referencing statement A")
    record_a_statement: str = Field(..., description="Statement or value in document A")
    record_b_source: str = Field(..., description="Second document ID / date referencing statement B")
    record_b_statement: str = Field(..., description="Conflicting statement or value in document B")
    severity: str = Field(default="High", description="Severity level: 'Low', 'Medium', 'High', 'Critical'")
    requires_clinician_review: bool = Field(default=True, description="Flag indicating clinician review is required")


class ContradictionReport(BaseModel):
    """Full contradiction analysis report across uploaded patient files."""
    patient_id: str
    contradictions_found: bool = Field(..., description="True if any clinical contradiction was detected")
    contradictions: List[ContradictionItem] = Field(default_factory=list, description="List of detected conflicts")
    summary_text: str = Field(..., description="Readable explanation summarizing conflicts for clinician review")


