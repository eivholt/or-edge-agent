import base64
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

VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8081/v1")
# VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
# VLM_MODEL = os.getenv("VLM_MODEL", "mistralai/Ministral-3-3B-Instruct-2512")
VLM_MODEL = os.getenv("VLM_MODEL", "mistralai/Ministral-3-3B-Instruct-2512-BF16")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "local-dev-key")

EMR_BASE_URL = "http://localhost:9000"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

INSTRUCTIONS = """\
You are a local OR logistics agent (synthetic demo). Coordinate supply, porter, specimen, and review workflows. Never diagnose or prescribe.

CRITICAL: Execute ONLY the proposed_tool_calls from reconciliation. Do NOT invent or add extra tasks.
- If proposed_tool_calls is empty [], set green light and return a short summary. Nothing else.
- Items in flagged_no_deficit have NO deficit (visible >= required). NEVER create tasks for them.
- confidence_level is "high" (>=0.80) or "low" (<0.80). The proposed calls already have the correct task_type.

Rules:

1. all_present=true, empty proposed_tool_calls → set_or_prep_light green, short summary. No tasks.

2. proposed_tool_calls not empty → execute each proposed call exactly as given. Do not change task_type. Then set_or_prep_light yellow.

3. visually_ready_but_pathway_changed → proposed_tool_calls includes procedure_change_review + porter_hold + yellow light. Execute all of them, plus any deficit tasks.

4. VLM events (confidence_level="low" or these event_types) → VLM inspection required:
   - Check resources.cloud_connected to decide routing:
     * If cloud_connected=true → call inspect_scene_remote (GPT-4o, higher quality).
     * If cloud_connected=false → call inspect_scene_local (on-device Ministral 3B).
   - Event-type actions:
     - instrument_out_of_zone → human_review + yellow light
     - specimen_ready_check → specimen_handoff or human_review + yellow light
     - room_turnover_check → porter_hold + red light
     - ppe_compliance_check → human_review + yellow light UNLESS VLM explicitly says YES (compliant)
   If the event contains a "vlm_question" field, use that EXACT text as the question for the chosen VLM tool.
   For ppe_compliance_check: if the VLM answer does NOT start with "YES", treat it as non-compliant and create human_review + yellow light.
   If the preferred VLM fails, fall back to the other one.

IMPORTANT: Every scenario MUST end with exactly one set_or_prep_light call. Green = no issues, Yellow = action needed, Red = critical.

Never actuate (set_or_prep_light) when confidence_level is "low" unless VLM has confirmed.
Use request_spd_resupply only when explicitly told to order sterile resupply.

Never ask the user for input or confirmation. Always decide and act autonomously."""


# ── Dependencies (passed via RunContext) ─────────────────────────────


@dataclass
class AgentDeps:
    event: dict
    case: dict
    resources: dict
    reconciliation: dict
    emit: object = None  # optional SSE callback: emit(component, **kwargs)


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
    result = {
        "status": "created",
        "case_id": case_id,
        "task_type": task_type,
        "priority": priority,
        "summary": summary,
        "reason": reason,
    }
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="create_synthetic_or_task",
                      tool_args={"case_id": case_id, "task_type": task_type, "priority": priority, "summary": summary},
                      tool_result=result,
                      detail=f"create_synthetic_or_task(task_type={task_type})")
        ctx.deps.emit("tasks",
                      task_type=task_type,
                      priority=priority,
                      summary=summary,
                      detail=summary)
    return result


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
    result = {
        "request_id": f"SPD-{item_name}-{room_id}",
        "item_name": item_name,
        "room_id": room_id,
        "urgency": urgency,
        "status": "requested",
    }
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="request_spd_resupply",
                      tool_args={"item_name": item_name, "room_id": room_id, "urgency": urgency},
                      tool_result=result,
                      detail=f"request_spd_resupply(item={item_name})")
        ctx.deps.emit("spd",
                      item_name=item_name,
                      urgency=urgency,
                      detail=f"Runner: {item_name}")
    return result


@or_agent.tool
def set_or_prep_light(
    ctx: RunContext[AgentDeps],
    room_id: str,
    color: str,
) -> dict:
    """Set the OR prep status light.

    Use green for logistics-ready, yellow for review-needed,
    red only for high-confidence safety exceptions.
    Never use below confidence 0.8.

    Args:
        room_id: The OR room identifier.
        color: One of green, yellow, red.
    """
    result = {
        "room_id": room_id,
        "color": color,
        "status": "set",
    }
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="set_or_prep_light",
                      tool_args={"room_id": room_id, "color": color},
                      tool_result=result,
                      detail=f"set_or_prep_light(color={color})")
        ctx.deps.emit("prep_light",
                      color=color,
                      reason="",
                      detail=f"Set to {color}")
    return result


@or_agent.tool
def inspect_scene_local(
    ctx: RunContext[AgentDeps],
    image_path: str,
    question: str,
) -> dict:
    """Inspect an OR scene image using the LOCAL VLM (Ministral 3B on-device).

    This is the preferred, low-latency visual inspection tool.
    Use it to verify instrument presence, tray layout, or setup state.
    Do not use for clinical diagnosis.

    Args:
        image_path: Path to an image file (relative to data/ or absolute).
        question: What to ask the VLM about the image.
    """
    # Override question with vlm_question from event if available
    vlm_q = ctx.deps.event.get("vlm_question")
    if vlm_q:
        question = vlm_q

    resolved = Path(image_path)
    if not resolved.is_absolute():
        resolved = DATA_DIR / image_path

    # Try alternate extensions if exact path not found
    if not resolved.is_file():
        for alt in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = resolved.with_suffix(alt)
            if candidate.is_file():
                resolved = candidate
                break

    if not resolved.is_file():
        return {"error": f"image not found: {resolved}"}

    suffix = resolved.suffix.lower().lstrip(".")
    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}
    mime_subtype = mime_map.get(suffix)
    if not mime_subtype:
        return {"error": f"unsupported image format: {suffix}"}

    encoded = base64.b64encode(resolved.read_bytes()).decode()
    data_url = f"data:image/{mime_subtype};base64,{encoded}"

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0,
    }

    r = httpx.post(
        f"{VLM_BASE_URL}/chat/completions",
        json=payload,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )
    r.raise_for_status()
    answer = r.json()["choices"][0]["message"]["content"]
    result = {"image": str(resolved), "question": question, "answer": answer}
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="inspect_scene_local",
                      tool_args={"image_path": image_path, "question": question},
                      tool_result=result,
                      detail=f"inspect_scene_local: {question[:80]}")
        ctx.deps.emit("vlm_local",
                      question=question,
                      answer=answer,
                      detail=f"Local VLM: {question[:80]}")
    return result


@or_agent.tool
def inspect_scene_remote(
    ctx: RunContext[AgentDeps],
    image_path: str,
    question: str,
) -> dict:
    """Inspect an OR scene image using the REMOTE Azure VLM (gpt-4o).

    This is a cloud fallback — only use when inspect_scene_local is
    unavailable or returns inconclusive results.  Higher quality but
    slower and incurs cloud costs.

    Args:
        image_path: Path to an image file (relative to data/ or absolute).
        question: What to ask the VLM about the image.
    """
    # Override question with vlm_question from event if available
    vlm_q = ctx.deps.event.get("vlm_question")
    if vlm_q:
        question = vlm_q

    from apps.vlm.ask_vlm import ask_vlm

    resolved = Path(image_path)
    if not resolved.is_absolute():
        resolved = DATA_DIR / image_path

    # Try alternate extensions if exact path not found
    if not resolved.is_file():
        for alt in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = resolved.with_suffix(alt)
            if candidate.is_file():
                resolved = candidate
                break

    if not resolved.is_file():
        return {"error": f"image not found: {resolved}"}

    answer = ask_vlm(str(resolved), question)
    result = {"image": str(resolved), "question": question, "answer": answer}
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="inspect_scene_remote",
                      tool_args={"image_path": image_path, "question": question},
                      tool_result=result,
                      detail=f"inspect_scene_remote: {question[:80]}")
        ctx.deps.emit("vlm_remote",
                      question=question,
                      answer=answer,
                      detail=f"Remote VLM (GPT-4o): {question[:80]}")
    return result


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

    # Normalise required_items to dict[str, int]
    raw_required = case.get("required_items", [])
    if isinstance(raw_required, dict):
        required = dict(raw_required)
    else:
        required = {}
        for item in raw_required:
            required[item] = required.get(item, 0) + 1

    # Normalise visible_items to dict[str, int]
    raw_visible = event.get("visible_items", [])
    if isinstance(raw_visible, dict):
        visible = dict(raw_visible)
    else:
        visible = {}
        for item in raw_visible:
            visible[item] = visible.get(item, 0) + 1

    missing = set(event.get("missing_or_uncertain", []))

    # Items where detector flagged uncertain AND pathway needs them AND there's a deficit
    actionable_missing = sorted(
        item for item in missing
        if item in required and visible.get(item, 0) < required[item]
    )

    # Items flagged uncertain but with no actual deficit (have >= need) — ignore these
    flagged_no_deficit = sorted(
        item for item in missing
        if item in required and visible.get(item, 0) >= required[item]
    )

    # Items with count deficit (need > have) that weren't flagged
    unaccounted = sorted(
        item for item, need in required.items()
        if item not in missing and visible.get(item, 0) < need
    )

    # Build detailed lists with have/need counts for dashboard display
    actionable_missing_detail = [
        f"{item} ({visible.get(item, 0)}/{required[item]})"
        for item in actionable_missing
    ]
    unaccounted_detail = [
        f"{item} ({visible.get(item, 0)}/{required[item]})"
        for item in unaccounted
    ]

    truly_clear = len(calls) == 0 and not actionable_missing and not unaccounted

    # Categorical confidence label for LLM consumption
    conf = event.get("confidence", 1.0)
    if conf >= 0.80:
        confidence_level = "high"
    else:
        confidence_level = "low"

    return {
        "confidence_level": confidence_level,
        "actionable_missing": actionable_missing,
        "actionable_missing_detail": actionable_missing_detail,
        "flagged_no_deficit": flagged_no_deficit,
        "unaccounted": unaccounted,
        "unaccounted_detail": unaccounted_detail,
        "all_present": truly_clear,
        "proposed_tool_calls": calls,
    }


@logfire.instrument("ask_agent case_id={case[case_id]} event_type={event[event_type]}")
def ask_agent(event: dict, case: dict, resources: dict, emit=None) -> dict:
    recon = reconcile_setup(event, case)

    deps = AgentDeps(
        event=event,
        case=case,
        resources=resources,
        reconciliation=recon,
        emit=emit,
    )

    prompt = json.dumps(
        {
            "event": {k: v for k, v in event.items() if k != "missing_or_uncertain"},
            "synthetic_pathway": case,
            "reconciliation": recon,
            "resources": resources,
        },
        indent=2,
    )
    result = or_agent.run_sync(prompt, deps=deps)

    # Extract tool calls and their results from the run result
    tool_calls = []
    tool_results = {}  # tool_call_id → result content
    for msg in result.all_messages():
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if getattr(part, "part_kind", "") == "tool-return":
                    tool_results[part.tool_call_id] = part.content
    for msg in result.all_messages():
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if getattr(part, "part_kind", "") == "tool-call":
                    call_id = part.tool_call_id
                    tc = {
                        "name": part.tool_name,
                        "arguments": (
                            part.args if isinstance(part.args, dict)
                            else json.loads(part.args) if isinstance(part.args, str)
                            else {}
                        ),
                    }
                    if call_id and call_id in tool_results:
                        tc["result"] = tool_results[call_id]
                    tool_calls.append(tc)

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