import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from apps.agent import run_fixture
from apps.detector import inference


FRAME_PATH = Path("data/frames/frame_all_present.png")


def _ctx():
    deps = run_fixture.AgentDeps(
        event={"visible_items": {"scissors": 1}},
        resources={},
    )
    return SimpleNamespace(deps=deps)


def test_green_context_fraction_distinguishes_drape_from_metal():
    detection = inference.Detection("scissors", 0.99, 24, 24, 8, 8)
    drape = np.full((64, 64, 3), (110, 190, 165), dtype=np.uint8)
    metal = np.full((64, 64, 3), (110, 100, 95), dtype=np.uint8)

    assert inference._green_context_fraction(drape, detection) == 1.0
    assert inference._green_context_fraction(metal, detection) == 0.0


def test_clean_scene_skips_vlm(monkeypatch):
    detection = inference.Detection(
        "scissors",
        0.99,
        24,
        24,
        8,
        8,
        green_context_fraction=0.9,
    )
    detector_result = SimpleNamespace(
        model_name="test detector",
        detections=[detection],
        frame_height=320,
    )
    monkeypatch.setattr(inference, "detect", lambda _: detector_result)

    def unexpected_post(*args, **kwargs):
        raise AssertionError("clean scenes must not call the VLM")

    monkeypatch.setattr(run_fixture.httpx, "post", unexpected_post)

    result = run_fixture._inspect_local(
        _ctx(),
        FRAME_PATH,
        str(FRAME_PATH),
        run_fixture.STERILE_ZONE_QUESTION,
    )

    assert result["verdict"] is False
    assert result["vlm_skipped"] is True
    assert result["candidates"] == []


def test_candidate_scene_stays_actionable_when_vlm_disagrees(monkeypatch):
    detection = inference.Detection(
        "scissors",
        0.99,
        208,
        240,
        16,
        16,
        green_context_fraction=0.0,
    )
    detector_result = SimpleNamespace(
        model_name="test detector",
        detections=[detection],
        frame_height=320,
    )
    monkeypatch.setattr(inference, "detect", lambda _: detector_result)
    request = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": json.dumps({"answer": False})}}
                ]
            }

    def fake_post(url, **kwargs):
        request["url"] = url
        request.update(kwargs)
        return Response()

    monkeypatch.setattr(run_fixture.httpx, "post", fake_post)

    result = run_fixture._inspect_local(
        _ctx(),
        FRAME_PATH,
        str(FRAME_PATH),
        run_fixture.STERILE_ZONE_QUESTION,
    )

    assert result["verdict"] is True
    assert result["vlm_verdict"] is False
    assert result["candidates"] == [
        {"label": "scissors", "green_context_fraction": 0.0}
    ]
    assert request["timeout"] == run_fixture.VLM_TIMEOUT_SECONDS
    assert request["json"]["max_tokens"] == 16
    assert len(request["json"]["messages"]) == 1