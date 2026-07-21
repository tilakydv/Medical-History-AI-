import uuid
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import Date, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def uuid4_str() -> str:
    return str(uuid.uuid4())


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    external_id: Mapped[str | None] = mapped_column(String(100), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    sex: Mapped[str | None] = mapped_column(String(30))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    reports: Mapped[list["Report"]] = relationship(back_populates="patient")
    mri_scans: Mapped[list["MRIScan"]] = relationship(back_populates="patient")


class Upload(Base):
    __tablename__ = "uploads"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(String(100))
    file_type: Mapped[str] = mapped_column(String(20), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(30), default="uploaded", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("patient_id", "sha256", "file_type", name="uq_patient_upload"),)


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    upload_id: Mapped[str] = mapped_column(ForeignKey("uploads.id", ondelete="CASCADE"), unique=True)
    report_type: Mapped[str] = mapped_column(String(50), default="clinical")
    report_date: Mapped[date | None] = mapped_column(Date, index=True)
    extracted_text: Mapped[str | None] = mapped_column(Text)
    structured_data: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    extraction_method: Mapped[str | None] = mapped_column(String(30))
    status: Mapped[str] = mapped_column(String(30), default="uploaded", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    patient: Mapped[Patient] = relationship(back_populates="reports")
    lab_report: Mapped["LaboratoryReport | None"] = relationship(back_populates="report", uselist=False)


class LaboratoryReport(Base):
    __tablename__ = "laboratory_reports"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id", ondelete="CASCADE"), unique=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    collected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    report: Mapped[Report] = relationship(back_populates="lab_report")
    values: Mapped[list["LaboratoryValue"]] = relationship(back_populates="lab_report", cascade="all, delete-orphan")


class LaboratoryValue(Base):
    __tablename__ = "laboratory_values"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    lab_report_id: Mapped[str] = mapped_column(ForeignKey("laboratory_reports.id", ondelete="CASCADE"), index=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    test_code: Mapped[str | None] = mapped_column(String(50), index=True)
    test_name: Mapped[str] = mapped_column(String(150), index=True)
    value_numeric: Mapped[float | None] = mapped_column(Float)
    value_text: Mapped[str | None] = mapped_column(String(100))
    unit: Mapped[str | None] = mapped_column(String(40))
    reference_low: Mapped[float | None] = mapped_column(Float)
    reference_high: Mapped[float | None] = mapped_column(Float)
    flag: Mapped[str | None] = mapped_column(String(20), index=True)
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    lab_report: Mapped[LaboratoryReport] = relationship(back_populates="values")
    __table_args__ = (Index("ix_lab_patient_test_time", "patient_id", "test_name", "observed_at"),)


class MRIScan(Base):
    __tablename__ = "mri_scans"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    upload_id: Mapped[str] = mapped_column(ForeignKey("uploads.id", ondelete="CASCADE"), unique=True)
    modality: Mapped[str] = mapped_column(String(20), default="MRI")
    format: Mapped[str] = mapped_column(String(20))
    study_date: Mapped[date | None] = mapped_column(Date, index=True)
    status: Mapped[str] = mapped_column(String(30), default="uploaded", index=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    patient: Mapped[Patient] = relationship(back_populates="mri_scans")
    result: Mapped["MRIResult | None"] = relationship(back_populates="scan", uselist=False)


class MRIResult(Base):
    __tablename__ = "mri_results"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    mri_scan_id: Mapped[str] = mapped_column(ForeignKey("mri_scans.id", ondelete="CASCADE"), unique=True)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str | None] = mapped_column(String(50))
    mask_path: Mapped[str] = mapped_column(Text)
    overlay_path: Mapped[str | None] = mapped_column(Text)
    tumor_volume_mm3: Mapped[float] = mapped_column(Float)
    confidence_score: Mapped[float | None] = mapped_column(Float)
    localization: Mapped[dict[str, Any]] = mapped_column(JSON)
    findings: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    scan: Mapped[MRIScan] = relationship(back_populates="result")


class TimelineEvent(Base):
    """Storage contract only; generation belongs to the separate LLM module."""
    __tablename__ = "timeline_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    source_type: Mapped[str] = mapped_column(String(30), index=True)
    source_id: Mapped[str] = mapped_column(String(36), index=True)
    event_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    event_type: Mapped[str] = mapped_column(String(50), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

