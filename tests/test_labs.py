from app.services.lab_service import LaboratoryService


def test_lab_parser_handles_numeric_and_qualitative_values():
    values = LaboratoryService().parse(
        "Hemoglobin 13.5 g/dL 12.0-16.0\nCRP 18 mg/L 0-5\nHIV negative"
    )
    assert len(values) == 3
    assert values[0].flag == "N"
    assert values[1].flag == "H"
    assert values[2].value_text == "negative"


def test_lab_parser_ignores_prose():
    assert LaboratoryService().parse("Patient feels better today.") == []

