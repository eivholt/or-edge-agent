import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent, ModelSettings, RunContext, Tool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from apps.agent.reconcile import reconcile

load_dotenv(override=True)

logfire.configure(service_name="or-edge-agent")
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()

VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8001/v1")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "local-dev-key")

EMR_BASE_URL = "http://localhost:9000"

INSTRUCTIONS = """\
You are a local OR logistics agent for a synthetic demo.

You coordinate simulated OR setup, supply, porter, specimen, and review workflows.
You must not diagnose, prescribe, select treatment, or clear a real clinical case.

You receive a reconciliation result that has ALREADY compared the detector's
observations against the surgical pathway's required items. Trust the
reconciliation — do NOT second-guess it or re-derive the gap list.

If reconciliation shows no gaps (actionable_missing and unaccounted are both
empty) AND no proposed_tool_calls, respond with a short summary and do not
call any tools.

For each item in actionable_missing, call create_synthetic_or_task with
task_type="missing_supply" — UNLESS the event indicates uncertainty
(event_type is sterile_zone_ambiguity or confidence < 0.80), in which case
use task_type="human_review" instead.

For each item in unaccounted, call create_synthetic_or_task with
task_type="missing_supply".

Tool routing rules:
- For a confirmed missing required item → create_synthetic_or_task with task_type="missing_supply".
- For uncertainty about a required item (sterile_zone_ambiguity, confidence < 0.80)
  → create_synthetic_or_task with task_type="human_review".
  Do NOT assert sterile or contaminated status — defer to human review.
- For a visually_ready_but_pathway_changed event, do ALL three:
  1. create_synthetic_or_task with task_type="procedure_change_review", priority="high".
  2. create_synthetic_or_task with task_type="porter_hold".
  3. set_or_prep_light with color="yellow".
- Use request_spd_resupply only when explicitly told to order a sterile resupply delivery.
- If request_spd_robot_delivery is available, prefer it over request_spd_resupply.
- Never set an actuator (set_or_prep_light) unless the event is high confidence and operational-only."""


# ── Dependencies (passed via RunContext) ─────────────────────────────


@dataclass
class AgentDeps:
    event: dict
    case: dict
    resources: dict
    reconciliation: dict


# ── Agent ────────────────────────────────────────────────────────────

_model = OpenAIChatModel(
    VLM_MODEL,
    provider=OpenAIProvider(base_url=VLM_BASE_URL, api_key=OPENAI_API_KEY),
    profile=OpenAIModelProfile(openai_supports_strict_tool_definition=False),
)

or_agent = Agent(
    _model,
    instructions=INSTRUCTIONS,
    deps_type=AgentDeps,
    model_settings=ModelSettings(temperature=0, max_tokens=700),
)


# ── Tools (dynamically discoverable via native tool calling) ─────────


@or_agent.tool
def create_synthetic_or_task(
    ctx: RunContext[AgentDeps],
    case_id: str,
    task_type: str,
    priority: str,
    summary: str,
    reason: str,
) -> dict:
    """Create a synthetic OR workflow task.

    Use for missing_supply, human_review, porter_hold, porter_release,
    specimen_handoff, wrong_case_cart, or procedure_change_review tasks.

    Args:
        case_id: The case identifier.
        task_type: One of missing_supply, human_review, porter_hold,
            porter_release, specimen_handoff, wrong_case_cart,
            procedure_change_review.
        priority: One of low, normal, high.
        summary: Short description of what is missing or needs review.
        reason: Why this task is being created.
    """
    return {
        "status": "created",
        "case_id": case_id,
        "task_type": task_type,
        "priority": priority,
        "summary": summary,
        "reason": reason,
    }


@or_agent.tool
def request_spd_resupply(
    ctx: RunContext[AgentDeps],
    item_name: str,
    room_id: str,
    urgency: str,
) -> dict:
    """Request sterile processing delivery for a missing item.

    Use only when explicitly told to order a sterile resupply delivery.
    Do not use for items requiring direct human sign-off.

    Args:
        item_name: Name of the item to resupply.
        room_id: The OR room identifier.
        urgency: One of low, normal, high.
    """
    return {
        "request_id": f"SPD-{item_name}-{room_id}",
        "item_name": item_name,
        "room_id": room_id,
        "urgency": urgency,
        "status": "requested",
    }


@or_agent.tool
def set_or_prep_light(
    ctx: RunContext[AgentDeps],
    room_id: str,
    color: str,
    duration_seconds: int,
) -> dict:
    """Set the OR prep status light.

    Use green for logistics-ready, yellow for review-needed,
    red only for high-confidence safety exceptions.
    Never use below confidence 0.8.

    Args:
        room_id: The OR room identifier.
        color: One of green, yellow, red.
        duration_seconds: Duration in seconds (1-10).
    """
    return {
        "room_id": room_id,
        "color": color,
        "duration_seconds": duration_seconds,
        "status": "set",
    }


# ── Helpers ──────────────────────────────────────────────────────────


def load_event(path: str) -> dict:
    return json.loads(Path(path).read_text())


@logfire.instrument("get_case case_id={case_id}")
def get_case(case_id: str) -> dict:
    r = httpx.get(f"{EMR_BASE_URL}/cases/{case_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def get_resources(room_id: str) -> dict:
    return {
        "room_id": room_id,
        "sterile_processing_robot": {"available": True, "eta_seconds": 180},
        "human_runner": {"available": True, "eta_seconds": 420},
        "porter": {"available": True, "eta_seconds": 300},
        "local_vlm": {"available": True, "estimated_latency_seconds": 5},
        "pc_gpu_vlm": {"available": True, "estimated_latency_seconds": 3},
    }


@logfire.instrument("reconcile_setup case_id={case[case_id]}")
def reconcile_setup(event: dict, case: dict) -> dict:
    """Run deterministic reconciliation — no LLM needed."""
    calls = reconcile(event, case)
    required = set(case.get("required_items", []))
    visible = set(event.get("visible_items", []))
    missing = set(event.get("missing_or_uncertain", []))

    return {
        "actionable_missing": sorted(missing & required),
        "unaccounted": sorted(required - visible - missing),
        "all_present": len(calls) == 0,
        "proposed_tool_calls": calls,
    }


@logfire.instrument("ask_agent case_id={case[case_id]} event_type={event[event_type]}")
def ask_agent(event: dict, case: dict, resources: dict) -> dict:
    recon = reconcile_setup(event, case)

    # Fast path: no gaps → no LLM call needed
    if recon["all_present"]:
        return {
            "decision_summary": "All required items are present. No action needed.",
            "tool_calls": [],
            "requires_human_review": False,
        }

    deps = AgentDeps(
        event=event,
        case=case,
        resources=resources,
        reconciliation=recon,
    )

    prompt = json.dumps(
        {
            "event": event,
            "synthetic_pathway": case,
            "reconciliation": recon,
            "resources": resources,
        },
        indent=2,
    )
    result = or_agent.run_sync(prompt, deps=deps)

    # Extract tool calls from the run result
    tool_calls = []
    for msg in result.all_messages():
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if hasattr(part, "tool_name") and hasattr(part, "args"):
                    tool_calls.append({
                        "name": part.tool_name,
                        "arguments": (
                            part.args if isinstance(part.args, dict)
                            else json.loads(part.args) if isinstance(part.args, str)
                            else {}
                        ),
                    })

    has_human_review = any(
        tc["arguments"].get("task_type") == "human_review"
        for tc in tool_calls
        if tc["name"] == "create_synthetic_or_task"
    )

    return {
        "decision_summary": result.output if isinstance(result.output, str) else str(result.output),
        "tool_calls": tool_calls,
        "requires_human_review": has_human_review,
    }


def main():
    import argparse
    from apps.agent.validation import validate_decision

    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    args = parser.parse_args()

    event = load_event(args.scenario)
    case = get_case(event["case_id"])
    resources = get_resources(event["room_id"])
    decision = ask_agent(event, case, resources)

    errors = validate_decision(decision, event)
    if errors:
        print(json.dumps({"validation_errors": errors, "decision": decision}, indent=2))
        raise SystemExit(2)

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()