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

    def extract(self, path: Path) -> OCRResult:
        suffix = path.name.lower()
        if suffix.endswith(".pdf"):
            return self._pdf(path)
        return self._images([Image.open(path)])

    def _pdf(self, path: Path) -> OCRResult:
        try:
            document = fitz.open(path)
        except Exception as exc:
            raise ProcessingError("Unable to open PDF") from exc
        pages: list[dict[str, Any]] = []
        used_ocr = False
        try:
            for number, page in enumerate(document, start=1):
                embedded = self.clean_text(page.get_text("text"))
                has_key_terms = any(w in embedded.lower() for w in ["patient", "name", "complaint", "history", "diagnosis", "impression", "reference", "result", "date", "report", "examination", "rx"])
                if len(embedded) >= 150 and has_key_terms:
                    text = embedded
                    method = "pymupdf"
                else:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = self._ocr_image(image)
                    merged_lines = []
                    seen_lines = set()
                    for line in (embedded + "\n" + ocr_text).splitlines():
                        l_clean = line.strip()
                        if l_clean and l_clean.lower() not in seen_lines:
                            merged_lines.append(l_clean)
                            seen_lines.add(l_clean.lower())
                    text = "\n".join(merged_lines)
                    method = "mixed" if len(embedded) > 10 else "paddleocr"
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
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "".join(
            character if character in "\n\t" or unicodedata.category(character) != "Cc" else " "
            for character in text
        )
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

        lines = [l.strip() for l in extracted_text.splitlines() if l.strip()]
        doc_name_clean = ""
        for idx, line in enumerate(lines[:30]):
            line_lower = line.lower()
            if any(h in line_lower for h in ["patient name", "name of patient", "patient's name", "pt name", "patient:"]):
                parts = re.split(r"[:\-]", line, 1)
                candidate = parts[1].strip() if len(parts) > 1 else ""
                if not candidate and idx + 1 < len(lines):
                    candidate = lines[idx + 1].strip()
                doc_name_clean = re.split(r"\b(?:age|sex|gender|dob|date|mrn|id|ref|doctor|dr)\b", candidate, flags=re.IGNORECASE)[0].strip()
                break

        if doc_name_clean:
            doc_tokens = {w.lower() for w in re.findall(r"[A-Za-z]{2,}", doc_name_clean)}
            doc_tokens -= {"male", "female", "other", "years", "yrs", "year", "old", "name", "patient"}
            if doc_tokens and not (reg_tokens & doc_tokens):
                raise ProcessingError(
                    f"Document patient name ('{doc_name_clean}') does not match registered patient ('{patient_name}')"
                )

