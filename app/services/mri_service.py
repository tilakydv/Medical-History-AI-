import json
import subprocess
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pydicom
from PIL import Image

from app.core.config import Settings
from app.core.exceptions import InvalidFileError, ModelUnavailableError, ProcessingError


class MRIService:
    """Validated MRI loading and configured nnU-Net inference adapter.

    This service does not ship a clinical model. Deployments must configure validated
    weights and NNUNET_COMMAND before analysis can run.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def inspect(self, path: Path) -> dict[str, Any]:
        if path.name.lower().endswith((".nii", ".nii.gz")):
            image = nib.load(str(path))
            if len(image.shape) < 3:
                raise InvalidFileError("NIfTI MRI must contain a 3D volume")
            return {"format": "nifti", "shape": list(image.shape),
                    "voxel_spacing_mm": [float(v) for v in image.header.get_zooms()[:3]]}
        if path.suffix.lower() in {".dcm", ".dicom"}:
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            except Exception as exc:
                raise InvalidFileError("Invalid DICOM file") from exc
            return {"format": "dicom", "modality": str(getattr(ds, "Modality", "")),
                    "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")),
                    "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", ""))}
        if path.suffix.lower() == ".zip":
            return {"format": "dicom_archive", "validated": "during secure extraction"}
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    return {"format": "reference_image", "dimensions": list(image.size),
                            "image_mode": image.mode, "analysis_supported": False}
            except Exception as exc:
                raise InvalidFileError("Invalid MRI reference image") from exc
        raise InvalidFileError(
            "Unsupported MRI content. Use NIfTI, DICOM, a DICOM ZIP, PNG, JPEG, or TIFF."
        )

    def analyze(self, input_path: Path, output_dir: Path) -> dict[str, Any]:
        if not self.settings.nnunet_command or not self.settings.mri_model_name:
            raise ModelUnavailableError(
                "nnU-Net is not configured. Set NNUNET_COMMAND and MRI_MODEL_NAME to a validated model."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        nifti = self._to_nifti(input_path, output_dir)
        normalized = self._preprocess(nifti, output_dir)
        mask = output_dir / "segmentation.nii.gz"
        command = [*self.settings.nnunet_command.split(), "-i", str(normalized), "-o", str(mask),
                   "-d", self.settings.mri_model_name]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=3600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            raise ProcessingError("nnU-Net inference failed") from exc
        if not mask.exists():
            raise ProcessingError("nnU-Net completed without producing a segmentation mask")
        return self._measure(normalized, mask, output_dir)

    def _to_nifti(self, path: Path, output_dir: Path) -> Path:
        if path.name.lower().endswith((".nii", ".nii.gz")):
            return path
        raise ProcessingError(
            "3D analysis requires NIfTI input. DICOM conversion requires a configured "
            "series converter; 2D reference images cannot be segmented by this pipeline."
        )

    def _preprocess(self, path: Path, output_dir: Path) -> Path:
        image = nib.load(str(path))
        data = np.asarray(image.get_fdata(), dtype=np.float32)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            raise InvalidFileError("MRI volume contains no finite voxel data")
        lo, hi = np.percentile(finite, [0.5, 99.5])
        data = np.clip(data, lo, hi)
        std = float(data.std())
        data = (data - float(data.mean())) / (std or 1.0)
        target = output_dir / "preprocessed.nii.gz"
        nib.save(nib.Nifti1Image(data, image.affine, image.header), str(target))
        return target

    def _measure(self, image_path: Path, mask_path: Path, output_dir: Path) -> dict[str, Any]:
        image, mask_img = nib.load(str(image_path)), nib.load(str(mask_path))
        data, mask = image.get_fdata(), mask_img.get_fdata() > 0
        if data.shape[:3] != mask.shape[:3]:
            raise ProcessingError("Segmentation mask shape does not match MRI")
        coords = np.argwhere(mask)
        spacing = image.header.get_zooms()[:3]
        volume = float(mask.sum() * np.prod(spacing))
        localization = {"voxel_bounding_box": None, "centroid_voxel": None}
        if coords.size:
            localization = {"voxel_bounding_box": [coords.min(0).tolist(), coords.max(0).tolist()],
                            "centroid_voxel": coords.mean(0).round(2).tolist()}
        overlay = self._overlay(data, mask, output_dir / "overlay.png")
        confidence = self._sidecar_confidence(mask_path)
        return {"model_name": self.settings.mri_model_name, "model_version": None,
                "mask_path": str(mask_path), "overlay_path": str(overlay),
                "tumor_volume_mm3": volume, "confidence_score": confidence,
                "localization": localization,
                "findings": {"segmentation_present": bool(mask.any()),
                             "disclaimer": "Algorithmic output; requires qualified clinical review."}}

    @staticmethod
    def _overlay(volume: np.ndarray, mask: np.ndarray, target: Path) -> Path:
        z = int(np.argmax(mask.sum(axis=(0, 1)))) if mask.any() else volume.shape[2] // 2
        base = volume[:, :, z]
        base = ((base - base.min()) / (np.ptp(base) or 1) * 255).astype(np.uint8)
        rgb = np.stack([base] * 3, axis=-1)
        rgb[mask[:, :, z]] = [255, 40, 40]
        Image.fromarray(np.rot90(rgb)).save(target)
        return target

    @staticmethod
    def _sidecar_confidence(mask_path: Path) -> float | None:
        sidecar = mask_path.with_suffix("").with_suffix(".json")
        if sidecar.exists():
            try:
                return float(json.loads(sidecar.read_text())["confidence"])
            except (ValueError, KeyError, json.JSONDecodeError):
                return None
        return None
