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


def test_lab_parser_handles_visual_table_rows_and_page_split_name():
    text = """Test Description Result Reference Interval Units Status / Flag
Total Leukocyte Count (WBC) 11.4 4.5 - 11.0 × 103 / µL HIGH
Serum Glucose (Fasting) 92 70 - 99 mg/dL Normal
139 136 - 145 mmol/L Normal
CONFIDENTIAL MEDICAL RECORD Page 1 of 2
Test Description Result Reference Interval Units Status / Flag
Sodium (Na+)
Potassium (K+) 4.2 3.5 - 5.1 mmol/L Normal"""

    values = LaboratoryService().parse(text)
    found = {item.test_name: item for item in values}

    assert found["Total Leukocyte Count (WBC)"].flag == "H"
    assert found["Serum Glucose (Fasting)"].value_numeric == 92
    assert found["Sodium (Na+)"].value_numeric == 139
    assert found["Potassium (K+)"].value_numeric == 4.2


def test_lab_parser_recomputes_flags_from_printed_ranges():
    values = LaboratoryService().parse(
        "Eosinophils (%) 0.6 1 - 6 % Normal\n"
        "Estimated GFR (eGFR) 90 > 90 mL/min/1.73m2 Normal\n"
        "Procalcitonin 0.18 < 0.25 ng/mL Normal"
    )
    found = {item.test_name: item.flag for item in values}

    assert found["Eosinophils (%)"] == "L"
    assert found["Estimated GFR (eGFR)"] == "L"
    assert found["Procalcitonin"] == "N"


def test_lab_parser_preserves_comparator_value_and_inclusive_reference():
    overview = LaboratoryService().overview(
        "Estimated GFR (eGFR) > 90 >= 90 mL/min/1.73m2 Normal"
    )

    assert overview["counts"]["outside_reference_range"] == 0
    assert overview["within_reference_range"][0]["value"] == ">90"
    assert overview["within_reference_range"][0]["reference"] == ">90"
