import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from apps.agent import run_fixture
from apps.detector import inference


FRAME_PATH = Path("data/frames/frame_all_present.png")


def _ctx(events):
    deps = run_fixture.AgentDeps(
        event={"visible_items": {"scissors": 1}},
        resources={
            "_scene_detections": {
                "frame_width": 320,
                "frame_height": 320,
                "items": [
                    {"label": "sponge", "x": 80, "y": 80, "width": 8, "height": 8},
                    {"label": "scissors", "x": 208, "y": 240, "width": 16, "height": 16},
                ],
            }
        },
        emit=lambda component, **kwargs: events.append(
            {"component": component, **kwargs}
        ),
    )
    return SimpleNamespace(deps=deps)


def test_local_vlm_processes_each_centroid_and_aggregates_progress(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        inference,
        "detect",
        lambda _: pytest.fail("dashboard detector geometry should be reused"),
    )
    monkeypatch.setattr(run_fixture, "DATA_DIR", tmp_path)
    requests = []
    answers = iter([False, True])

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": json.dumps({"answer": next(answers)})}}
                ]
            }

    def fake_post(url, **kwargs):
        requests.append({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr(run_fixture.httpx, "post", fake_post)

    events = []
    result = run_fixture._inspect_local(
        _ctx(events),
        FRAME_PATH,
        str(FRAME_PATH),
        run_fixture.STERILE_ZONE_QUESTION,
    )

    assert result["verdict"] is True
    assert result["segments_processed"] == result["segments_total"] == 2
    assert [segment["verdict"] for segment in result["segments"]] == [False, True]
    assert "vlm_skipped" not in result
    assert "candidates" not in result
    assert len(requests) == 2
    assert all(request["timeout"] == run_fixture.VLM_TIMEOUT_SECONDS for request in requests)
    assert all(request["json"]["max_tokens"] == 16 for request in requests)
    assert "detected sponge" in requests[0]["json"]["messages"][0]["content"][1]["text"]

    crop_sizes = []
    for request in requests:
        data_url = request["json"]["messages"][0]["content"][0]["image_url"]["url"]
        encoded = data_url.split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as sent_image:
            crop_sizes.append(sent_image.size)
    assert crop_sizes == [
        (run_fixture.VLM_SEGMENT_IMAGE_SIZE, run_fixture.VLM_SEGMENT_IMAGE_SIZE),
        (run_fixture.VLM_SEGMENT_IMAGE_SIZE, run_fixture.VLM_SEGMENT_IMAGE_SIZE),
    ]
    assert "The gray instrument itself does not count" in (
        requests[0]["json"]["messages"][0]["content"][1]["text"]
    )

    vlm_events = [event for event in events if event["component"] == "vlm_local"]
    assert [event.get("segment_status") for event in vlm_events[:-1]] == [
        "running", "complete", "running", "complete"
    ]
    assert vlm_events[-1]["segments_processed"] == vlm_events[-1]["segments_total"] == 2
    assert all(
        segment["image_url"].startswith("/data/segments/")
        for segment in result["segments"]
    )
    display_path = tmp_path / result["segments"][0]["image_url"].split("?", 1)[0].removeprefix("/data/")
    with Image.open(display_path) as display_image:
        assert display_image.size != crop_sizes[0]