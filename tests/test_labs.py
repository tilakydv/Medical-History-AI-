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


def test_lab_parser_handles_vertical_followup_comparison_table():
    text = """Fasting Glucose
118 mg/dL
88 mg/dL
70 – 99 mg/dL
Normal
HbA1c
6.1 %
5.4 %
4.0 – 5.6 %
Normal (Normalized)
Total Cholesterol
224 mg/dL
175 mg/dL
< 200 mg/dL
Optimal
LDL Cholesterol
142 mg/dL
98 mg/dL
< 100 mg/dL
Optimal
HDL Cholesterol
44 mg/dL
58 mg/dL
> 40 mg/dL
Improved"""

    values = LaboratoryService().parse(text)

    assert [(item.test_name, item.value_numeric, item.flag) for item in values] == [
        ("Fasting Glucose", 88.0, "N"),
        ("HbA1c", 5.4, "N"),
        ("Total Cholesterol", 175.0, "N"),
        ("LDL Cholesterol", 98.0, "N"),
        ("HDL Cholesterol", 58.0, "N"),
    ]
