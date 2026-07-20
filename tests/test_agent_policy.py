from types import SimpleNamespace

from apps.agent.run_fixture import AgentDeps, set_stacklight


def test_stacklight_policy_enforces_sterile_red_and_ignores_duplicates():
    deps = AgentDeps(
        event={},
        resources={},
        reconciliation={"deficits": [{"item": "sponge", "have": 1, "need": 2}]},
        sterile_verdict=True,
    )
    ctx = SimpleNamespace(deps=deps)

    first = set_stacklight(ctx, "OR-2", "yellow", "Supply deficit")
    duplicate = set_stacklight(ctx, "OR-2", "green", "Ready")

    assert first["color"] == "red"
    assert first["requested_color"] == "yellow"
    assert first["policy_enforced"] is True
    assert duplicate["status"] == "ignored_duplicate"
    assert duplicate["color"] == "red"
    assert deps.stacklight_colors == ["red"]