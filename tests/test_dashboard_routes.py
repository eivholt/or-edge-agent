import asyncio
import json

from apps.dashboard import server


def test_run_scenario_defaults_to_local_models(monkeypatch):
    calls = []

    def fake_pipeline(scenario_path, cloud_connected):
        calls.append((scenario_path, cloud_connected))
        return {"ok": True}

    monkeypatch.setattr(server, "_run_pipeline", fake_pipeline)

    result = asyncio.run(server.run_scenario("all_present"))

    assert result == {"ok": True}
    assert calls[0][1] is False


def test_run_scenario_propagates_pipeline_error(monkeypatch):
    monkeypatch.setattr(
        server,
        "_run_pipeline",
        lambda *_: {"ok": False, "error": "model unavailable"},
    )

    response = asyncio.run(server.run_scenario("all_present"))

    assert response.status_code == 500
    assert json.loads(response.body) == {
        "ok": False,
        "error": "model unavailable",
    }


def test_pipeline_marks_agent_exception_as_failure(monkeypatch, tmp_path):
    scenario = tmp_path / "agent_failure.json"
    scenario.write_text(
        json.dumps({"case_id": "CASE-TEST", "room_id": "OR-2", "visible_items": {}})
    )
    events = []

    def fail_agent(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("apps.agent.run_fixture.ask_agent", fail_agent)
    monkeypatch.setattr(
        server,
        "_emit",
        lambda component, **kwargs: events.append(
            {"component": component, **kwargs}
        ),
    )

    result = server._run_pipeline(str(scenario), cloud_connected=False)

    assert result == {"ok": False, "error": "Agent error: model unavailable"}
    assert [event["status"] for event in events if event["component"] == "agent"] == [
        "thinking",
        "error",
    ]
    assert events[-1] == {
        "component": "validation",
        "errors": ["Agent error: model unavailable"],
        "detail": "Agent error: model unavailable",
    }