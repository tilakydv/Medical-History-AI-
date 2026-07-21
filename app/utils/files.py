import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.core.exceptions import InvalidFileError

REPORT_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MRI_EXTENSIONS = {
    ".nii", ".gz", ".dcm", ".dicom", ".zip",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff",
}
MAGIC = {
    ".pdf": (b"%PDF",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".tif": (b"II*\x00", b"MM\x00*"),
    ".tiff": (b"II*\x00", b"MM\x00*"),
    ".zip": (b"PK\x03\x04",),
    ".gz": (b"\x1f\x8b",),
}


@dataclass(frozen=True)
class StoredFile:
    path: Path
    size: int
    sha256: str
    extension: str


def safe_filename(name: str | None) -> str:
    candidate = Path(name or "upload").name
    return re.sub(r"[^A-Za-z0-9._-]", "_", candidate)[:200]


def extension_for(name: str) -> str:
    lower = name.lower()
    return ".nii.gz" if lower.endswith(".nii.gz") else Path(lower).suffix


async def store_upload(upload: UploadFile, destination: Path, allowed: set[str], max_bytes: int) -> StoredFile:
    name = safe_filename(upload.filename)
    ext = extension_for(name)
    validation_ext = ".gz" if ext == ".nii.gz" else ext
    if validation_ext not in allowed:
        raise InvalidFileError(f"Unsupported file extension: {ext or 'none'}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    header = b""
    try:
        with destination.open("wb") as target:
            while chunk := await upload.read(1024 * 1024):
                if not header:
                    header = chunk[:132]
                size += len(chunk)
                if size > max_bytes:
                    raise InvalidFileError(f"File exceeds maximum size of {max_bytes // 1024 // 1024} MB")
                digest.update(chunk)
                target.write(chunk)
        if size == 0:
            raise InvalidFileError("Uploaded file is empty")
        signatures = MAGIC.get(validation_ext)
        if signatures and not any(header.startswith(signature) for signature in signatures):
            raise InvalidFileError("File content does not match its extension")
        if ext in {".dcm", ".dicom"} and len(header) >= 132 and header[128:132] != b"DICM":
            # Some valid DICOM datasets omit the preamble; pydicom performs final validation.
            pass
        return StoredFile(destination, size, digest.hexdigest(), ext)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
