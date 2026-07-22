from app.services.lab_service import LaboratoryService


def test_page_number_is_not_a_laboratory_value():
    text = "Case No. MD-42\nDate July 25, 2009\nPage 2\nClinical narrative follows."

    assert LaboratoryService().parse(text) == []
