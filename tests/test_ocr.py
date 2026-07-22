from app.services.ocr_service import OCRService


def test_clean_text_normalizes_custom_pdf_spacing_and_hyphens():
    source = "Case:\x02\x02MD\x0309\nDate:\x02August\x028,\x022009\nD\x03dimer"

    cleaned = OCRService.clean_text(source)

    assert cleaned == "Case: MD-09\nDate: August 8, 2009\nD-dimer"
    assert not any(ord(character) < 32 and character != "\n" for character in cleaned)


def test_clean_text_removes_other_nonprinting_control_characters():
    assert OCRService.clean_text("Clinical\x00\x07 report") == "Clinical report"


def test_clean_text_removes_private_glyphs_and_repairs_encoding_artifacts():
    cleaned = OCRService.clean_text(
        "WBC 11.4 Ã— 103 / ÂµL\nValue\uf000 (cid:127) next"
    )

    assert cleaned == "WBC 11.4 x 103 / uL\nValue next"
