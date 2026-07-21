"""
Schema for Medication & Allergy Intelligence (Module 4).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class MedicationItem(BaseModel):
    """Detailed medication record."""
    name: str = Field(..., description="Medication generic/brand name")
    dosage: Optional[str] = Field(default=None, description="Dosage (e.g., '500 mg', '10 units')")
    frequency: Optional[str] = Field(default=None, description="Frequency (e.g., 'Once daily', 'BID', 'As needed')")
    duration: Optional[str] = Field(default=None, description="Duration of prescription (e.g., '7 days', 'Ongoing')")
    status: str = Field(default="Current", description="Status: 'Current' or 'Previous'")
    notes: Optional[str] = Field(default=None, description="Additional administration instructions or context")


class AllergyItem(BaseModel):
    """Detailed allergy record."""
    allergen: str = Field(..., description="Allergen name (e.g. 'Penicillin', 'Peanuts', 'Pollen')")
    allergy_type: str = Field(..., description="Type: 'Drug', 'Food', 'Environmental', 'Other'")
    reaction: Optional[str] = Field(default=None, description="Clinical reaction (e.g., 'Anaphylaxis', 'Rash', 'Dyspnea')")
    severity: Optional[str] = Field(default=None, description="Severity: 'Mild', 'Moderate', 'Severe', 'Unknown'")


class MedicationAllergyReport(BaseModel):
    """Comprehensive medication & allergy intelligence report."""
    patient_id: str
    current_medications: List[MedicationItem] = Field(default_factory=list, description="Currently active medications")
    previous_medications: List[MedicationItem] = Field(default_factory=list, description="Past or discontinued medications")
    drug_allergies: List[AllergyItem] = Field(default_factory=list, description="Drug allergies")
    food_allergies: List[AllergyItem] = Field(default_factory=list, description="Food allergies")
    environmental_allergies: List[AllergyItem] = Field(default_factory=list, description="Environmental allergies")
    summary_text: str = Field(..., description="Human-readable medication and allergy intelligence summary")


