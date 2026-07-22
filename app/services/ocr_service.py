import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from app.core.exceptions import ProcessingError


@dataclass
class OCRResult:
    text: str
    method: str
    pages: list[dict[str, Any]]


class OCRService:
    """Extracts embedded PDF text first, falling back to PaddleOCR per scanned page."""

    def __init__(self, language: str = "en") -> None:
        self.language = language
        self._engine: Any = None

    def extract(self, path: Path, sort_layout: bool = False) -> OCRResult:
        suffix = path.name.lower()
        if suffix.endswith(".pdf"):
            return self._pdf(path, sort_layout=sort_layout)
        return self._images([Image.open(path)])

    def _pdf(self, path: Path, sort_layout: bool = False) -> OCRResult:
        try:
            document = fitz.open(path)
        except Exception as exc:
            raise ProcessingError("Unable to open PDF") from exc
        pages: list[dict[str, Any]] = []
        used_ocr = False
        try:
            for number, page in enumerate(document, start=1):
                # Coordinate sorting preserves visual rows in table-heavy report types.
                embedded = self.clean_text(page.get_text("text", sort=sort_layout))
                # Very short content is typically a header/watermark on a scanned page.
                if len(embedded) >= 40:
                    text = embedded
                    method = "pymupdf"
                else:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = self._ocr_image(image)
                    method = "paddleocr"
                    used_ocr = True
                pages.append({"page": number, "text": text, "method": method})
        finally:
            document.close()
        return OCRResult("\n\n".join(p["text"] for p in pages if p["text"]),
                         "hybrid" if used_ocr else "pymupdf", pages)

    def _images(self, images: list[Image.Image]) -> OCRResult:
        pages = [{"page": i, "text": self._ocr_image(image), "method": "paddleocr"}
                 for i, image in enumerate(images, start=1)]
        return OCRResult("\n\n".join(p["text"] for p in pages), "paddleocr", pages)

    def _ocr_image(self, image: Image.Image) -> str:
        engine = self._paddle()
        processed = self.preprocess(image)
        try:
            result = engine.ocr(np.asarray(processed), cls=True)
            lines: list[str] = []
            for page in result or []:
                for item in page or []:
                    if len(item) > 1 and item[1]:
                        lines.append(str(item[1][0]))
            return self.clean_text("\n".join(lines))
        except Exception as exc:
            raise ProcessingError("OCR processing failed") from exc

    def _paddle(self) -> Any:
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ProcessingError(
                    "Scanned content requires optional OCR dependencies; install medbrief-backend[ocr]"
                ) from exc
            self._engine = PaddleOCR(use_angle_cls=True, lang=self.language, show_log=False)
        return self._engine

    @staticmethod
    def preprocess(image: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray)
        gray = ImageEnhance.Contrast(gray).enhance(1.5)
        return gray.filter(ImageFilter.MedianFilter(size=3))

    @staticmethod
    def clean_text(text: str) -> str:
        # Some PDFs use non-printing C0 characters as custom-font glyphs.
        # In those files PyMuPDF returns \x02 for a space and \x03 for a
        # hyphen, which browsers render as square replacement symbols.
        text = text.translate({0x02: " ", 0x03: "-"})
        # Repair common UTF-8 text decoded as a legacy single-byte encoding.
        # These are encoding sequences, not report-specific words or values.
        for broken, repaired in {
            "Ã—": "x", "Âµ": "u", "Â°": " degrees ",
            "â€“": "-", "â€”": "-", "â€¢": "-", "â€™": "'",
            "â‰¥": ">=", "â‰¤": "<=",
        }.items():
            text = text.replace(broken, repaired)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "".join(
            character if character in "\n\t" or unicodedata.category(character) not in {"Cc", "Co"}
            else " "
            for character in text
        )
        text = re.sub(r"\(cid:\d+\)", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    PATIENT_NAME_HEADER = re.compile(
        r"(?:patient\s+name|name\s+of\s+patient|patient['’]?s\s+name|pt\s+name|patient)\s*[:|-]\s*([A-Za-z\s.,'-]{2,60})",
        re.IGNORECASE,
    )

    def verify_patient_name(self, extracted_text: str, patient_name: str) -> None:
        """Verifies that if a patient name header exists in the document, it matches the registered patient."""
        if not extracted_text or not patient_name:
            return
        reg_tokens = {w.lower() for w in re.findall(r"[A-Za-z]{2,}", patient_name)}
        if not reg_tokens:
            return

        for line in extracted_text.splitlines()[:30]:
            match = self.PATIENT_NAME_HEADER.search(line)
            if match:
                doc_name_raw = match.group(1).strip()
                doc_name_clean = re.split(
                    r"\b(?:age|sex|gender|dob|date|mrn|id|ref|doctor|dr)\b", doc_name_raw, flags=re.IGNORECASE
                )[0].strip()
                doc_tokens = {w.lower() for w in re.findall(r"[A-Za-z]{2,}", doc_name_clean)}
                doc_tokens -= {"male", "female", "other", "years", "yrs", "year", "old", "name"}
                if doc_tokens and not (reg_tokens & doc_tokens):
                    raise ProcessingError(
                        f"Document patient name ('{doc_name_clean}') does not match registered patient ('{patient_name}')"
                    )

