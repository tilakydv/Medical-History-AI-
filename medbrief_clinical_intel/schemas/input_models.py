"""
Input schemas for patient records and structured MRI findings.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class MRIFindingsInput(BaseModel):
    """Structured input from MONAI / nnU-Net Brain MRI Segmentation Pipeline."""
    tumor_detected: bool = Field(default=False, description="Whether a tumor/lesion was detected in MRI")
    tumor_type: Optional[str] = Field(default=None, description="Suspected tumor classification (e.g. Glioma, Meningioma, Pituitary Adenoma)")
    location: Optional[str] = Field(default=None, description="Anatomical localization (e.g. Left Frontal Lobe, Cerebellum)")
    volume_mm3: Optional[float] = Field(default=None, description="Estimated tumor volume in mmÂ³")
    volume_cm3: Optional[float] = Field(default=None, description="Estimated tumor volume in cmÂ³")
    edema_present: Optional[bool] = Field(default=None, description="Presence of peritumoral edema")
    midline_shift: Optional[bool] = Field(default=None, description="Presence of midline brain shift")
    contrast_enhancement: Optional[str] = Field(default=None, description="Enhancement characteristics (e.g. Ring-enhancing, Homogeneous)")
    additional_notes: Optional[str] = Field(default=None, description="Additional radiological observations")


class DocumentRecord(BaseModel):
    """Extracted text & metadata from a single uploaded patient document/OCR report."""
    document_id: str = Field(..., description="Unique document ID or filename")
    date: Optional[str] = Field(default=None, description="Document date (YYYY-MM-DD or readable format)")
    document_type: str = Field(default="Clinical Report", description="Type of document (e.g. Discharge Summary, Lab Report, MRI Report, Prescription)")
    extracted_text: str = Field(..., description="OCR / PyMuPDF extracted raw text")
    language: Optional[str] = Field(default="en", description="Detected language ('en', 'hi', 'mixed')")


class PatientRecordInput(BaseModel):
    """Full input structure received by Clinical Intelligence Layer."""
    patient_id: str = Field(..., description="Unique Patient ID or identifier")
    patient_name: Optional[str] = Field(default=None, description="Patient name if available")
    age: Optional[int] = Field(default=None, description="Patient age")
    gender: Optional[str] = Field(default=None, description="Patient gender")
    documents: List[DocumentRecord] = Field(default_factory=list, description="Extracted reports/documents")
    mri_findings: Optional[MRIFindingsInput] = Field(default=None, description="Structured MRI segmentation output")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")


class PatientContext(BaseModel):
    """Aggregated patient record context formatted for Qwen LLM prompt injection."""
    patient_id: str
    demographics_summary: str
    combined_records_text: str
    mri_summary_text: str


