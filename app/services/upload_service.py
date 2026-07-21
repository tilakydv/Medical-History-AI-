import uuid

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.models import MRIScan, Patient, Report, Upload
from app.utils.files import (MRI_EXTENSIONS, REPORT_EXTENSIONS, StoredFile, extension_for,
                             safe_filename, store_upload)


class UploadService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    async def save_report(self, patient_id: str, file: UploadFile, report_type: str) -> tuple[Upload, Report, bool]:
        self._patient(patient_id)
        stored = await self._store(patient_id, file, "report", REPORT_EXTENSIONS)
        existing = self._duplicate(patient_id, stored.sha256, "report")
        if existing:
            stored.path.unlink(missing_ok=True)
            report = self.db.scalar(select(Report).where(Report.upload_id == existing.id))
            return existing, report, True  # type: ignore[return-value]
        upload = self._upload(patient_id, file, stored, "report")
        report = Report(patient_id=patient_id, upload_id=upload.id, report_type=report_type)
        self.db.add(report)
        self._commit()
        return upload, report, False

    async def save_mri(self, patient_id: str, file: UploadFile) -> tuple[Upload, MRIScan, bool]:
        self._patient(patient_id)
        stored = await self._store(patient_id, file, "mri", MRI_EXTENSIONS)
        existing = self._duplicate(patient_id, stored.sha256, "mri")
        if existing:
            stored.path.unlink(missing_ok=True)
            scan = self.db.scalar(select(MRIScan).where(MRIScan.upload_id == existing.id))
            return existing, scan, True  # type: ignore[return-value]
        upload = self._upload(patient_id, file, stored, "mri")
        scan = MRIScan(patient_id=patient_id, upload_id=upload.id, format=stored.extension)
        self.db.add(scan)
        self._commit()
        return upload, scan, False

    async def _store(self, patient_id: str, file: UploadFile, kind: str, allowed: set[str]) -> StoredFile:
        suffix = extension_for(safe_filename(file.filename))
        destination = self.settings.upload_dir / patient_id / kind / f"{uuid.uuid4()}{suffix}"
        return await store_upload(file, destination, allowed, self.settings.max_upload_bytes)

    def _upload(self, patient_id: str, file: UploadFile, stored: StoredFile, kind: str) -> Upload:
        row = Upload(id=str(uuid.uuid4()), patient_id=patient_id,
                     original_filename=safe_filename(file.filename), stored_path=str(stored.path),
                     content_type=file.content_type or "application/octet-stream", file_type=kind,
                     size_bytes=stored.size, sha256=stored.sha256)
        self.db.add(row)
        self.db.flush()
        return row

    def _patient(self, patient_id: str) -> Patient:
        patient = self.db.get(Patient, patient_id)
        if not patient:
            raise NotFoundError("Patient not found")
        return patient

    def _duplicate(self, patient_id: str, sha256: str, kind: str) -> Upload | None:
        return self.db.scalar(select(Upload).where(Upload.patient_id == patient_id,
                                                   Upload.sha256 == sha256, Upload.file_type == kind))

    def _commit(self) -> None:
        try:
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            raise
