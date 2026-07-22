from app.services.timeline_service import TimelineService


def test_timeline_orders_dates_and_keeps_key_dated_events():
    content = {
        "metadata": {"Date": "August 8, 2009"},
        "sections": {
            "Clinical history": (
                "On 03/16/2006 the patient presented with syncope. A CT scan found an aneurysm. "
                "The patient was discharged on medication. On 3/21/2006 the patient returned "
                "with pain. Imaging demonstrated a ruptured aneurysm. Emergency surgery was "
                "performed and the patient expired during the operation."
            ),
            "Records": "Record reviewed 7/22/2009. Image CD reviewed 8/8/2009.",
        },
        "unsectioned_text": "",
    }

    timeline = TimelineService().build(content)

    dates = [row["date"] for row in timeline]
    assert dates == ["2006-03-16", "2006-03-21", "2009-07-22", "2009-08-08"]
    march_21 = next(row["important_points"] for row in timeline
                    if row["date"] == "2006-03-21")
    assert any("ruptured aneurysm" in event for event in march_21)
    assert any("expired" in event for event in march_21)
    assert all(set(row) == {"date", "important_points"} for row in timeline)


def test_timeline_splits_multiple_dates_flattened_onto_one_line():
    content = {
        "sections": {
            "Records Reviewed": (
                "October 4, 2006: Physician Office Notes "
                "February 9, 2007: Physician Office Notes "
                "February 11, 2007: Emergency Department Records "
                "April 30, 2009: Physician letter"
            )
        },
        "metadata": {},
    }

    timeline = TimelineService().build(content)

    assert [row["date"] for row in timeline] == [
        "2006-10-04", "2007-02-09", "2007-02-11", "2009-04-30"
    ]
    assert timeline[0]["important_points"] == [
        "October 4, 2006: Physician Office Notes"
    ]
    assert timeline[-1]["important_points"] == [
        "April 30, 2009: Physician letter"
    ]
