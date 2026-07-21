import io
import re
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
        text = text.replace("\x00", "").replace("\r\n", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

