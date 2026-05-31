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

VLM_SYSTEM_PROMPT = (
    "You are a surgical-scene analyst. Describe exactly what you see in the "
    "image, then answer the user's question. Respond ONLY with JSON:\n"
    '{"description": "one or two sentences", "answer": true/false}\n'
    '"answer" is true when YES, false when NO. '
    "Base your answer strictly on what is visible — do not guess or assume."
)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

INSTRUCTIONS = """\
You are a local OR logistics agent (synthetic demo). Coordinate supply, porter, specimen, and review workflows. Never diagnose or prescribe.

WORKFLOW — always follow this order:
1. Call get_surgical_pathway(case_id) to fetch the EMR case and required equipment.
2. Call reconcile_instruments() to compare detected items against requirements.
3. VLM STEP (MANDATORY for these event_types): instrument_out_of_zone, sterile_zone_ambiguity, specimen_ready_check, specimen_container_seen, room_turnover_check, ppe_compliance_check.
   If the event_type is in this list, you MUST call a VLM tool NOW, BEFORE any other action:
   - If resources.cloud_connected=true → call inspect_scene_remote (higher quality).
   - If resources.cloud_connected=false → call inspect_scene_local (on-device).
   - Use image_path from the event. If the event has a "vlm_question" field, use that EXACT text as the question.
   - For sterile-zone checks (instrument_out_of_zone, sterile_zone_ambiguity): construct the VLM question by listing ALL instrument type names from required_items (no quantities).
     Use this template exactly — fill in ALL types, never omit any:
     "Look at the green/blue drape. Is even ONE of these outside the drape, on the bare table: scalpel, scissors, sponge, tweezers?"
     Replace the type list with the actual types from required_items. Always include EVERY type. Do NOT shorten or summarize the list.
   - If the preferred VLM fails, fall back to the other one.
   Additionally, if the EI object-detection model has detected enough of the instruments required by the EMR surgical pathway (i.e. all_present=true or most items accounted for), use VLM to verify whether any instruments are outside the sterile zone.
4. Read the reconciliation results. Execute the proposed_tool_calls from reconciliation. Do NOT invent extra tasks beyond what reconciliation and the rules below require.
5. For each missing_supply task, ALSO call request_spd_resupply for that item. A nurse runner delivers sterile items into the OR (robots cannot enter the sterile field).
6. End with exactly one set_or_prep_light call.

Rules:

- If proposed_tool_calls is empty [] and all_present=true AND event_type is NOT in the VLM list above → set_or_prep_light green, short summary. Nothing else.
- Items in flagged_no_deficit have NO deficit (visible >= required). NEVER create tasks for them.

- proposed_tool_calls not empty → execute each proposed call exactly as given. Do not change task_type.
  For each missing_supply task, also call request_spd_resupply with the item name, room_id, and the SAME urgency as the task priority (e.g. task priority="normal" → SPD urgency="normal").
  Then set_or_prep_light yellow.

- visually_ready_but_pathway_changed → proposed_tool_calls includes procedure_change_review + porter_hold + yellow light. Execute all of them, plus any deficit tasks.

- VLM event-type actions (AFTER calling VLM in step 3):
    - instrument_out_of_zone → human_review + yellow light
    - sterile_zone_ambiguity → human_review + yellow light
    - specimen_ready_check / specimen_container_seen → You MUST create EXACTLY TWO tasks:
      1. create_or_task with task_type="specimen_handoff" (nurse receives specimen for pathology)
      2. create_or_task with task_type="porter_hold" (porter picks up specimen and transports to pathology lab)
      Then set yellow light.
    - room_turnover_check → Include the VLM findings in a porter_hold task summary. Then set red light.
    - ppe_compliance_check → human_review + yellow light UNLESS VLM verdict is true (compliant), then green light.

IMPORTANT: Every scenario MUST end with exactly one set_or_prep_light call. Green = no issues, Yellow = action needed, Red = critical.

Never ask the user for input or confirmation. Always decide and act autonomously."""


# ── Dependencies (passed via RunContext) ─────────────────────────────


@dataclass
class AgentDeps:
    event: dict
    resources: dict
    case: dict = None  # populated by get_surgical_pathway tool
    reconciliation: dict = None  # populated by reconcile_instruments tool
    emit: object = None  # optional SSE callback: emit(component, **kwargs)
    _tool_count: int = 0  # tracks tool calls for iteration progress
    _tools_used: list = None  # names of tools called so far

    def __post_init__(self):
        if self._tools_used is None:
            self._tools_used = []

    _tool_display_names = {
        "create_or_task": "create OR task",
        "set_or_prep_light": "set OR prep light",
    }

    def emit_tool_progress(self, tool_name: str):
        """Emit an agent status update showing tool progress."""
        self._tool_count += 1
        display = self._tool_display_names.get(tool_name, tool_name.replace("_", " "))
        self._tools_used.append(display)
        if self.emit:
            progress = " → ".join(self._tools_used)
            self.emit("agent", status="thinking",
                      summary=progress,
                      detail=progress)


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
def get_surgical_pathway(
    ctx: RunContext[AgentDeps],
    case_id: str,
) -> dict:
    """Fetch the surgical pathway and required equipment list from the EMR.

    Call this FIRST to learn what instruments/supplies the procedure requires.
    The returned data includes required_items, procedure name, phase, and
    open workflow items.

    Args:
        case_id: The surgical case identifier (e.g. CASE-1042).
    """
    r = httpx.get(f"{EMR_BASE_URL}/cases/{case_id}", timeout=10)
    r.raise_for_status()
    case = r.json()
    # Store on deps so reconcile_instruments can use it
    ctx.deps.case = case
    ctx.deps.emit_tool_progress("get_surgical_pathway")
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="get_surgical_pathway",
                      tool_args={"case_id": case_id},
                      tool_result={"procedure": case.get("procedure"), "required_items": case.get("required_items")},
                      detail=f"get_surgical_pathway(case_id={case_id})")
        ctx.deps.emit("emr_api",
                      case_id=case_id,
                      procedure=case.get("procedure", ""),
                      required_items=case.get("required_items", {}),
                      detail=f"Fetched case {case_id}: {case.get('procedure', '?')}")
    return case


@or_agent.tool
def reconcile_instruments(
    ctx: RunContext[AgentDeps],
) -> dict:
    """Compare detected instruments against the surgical pathway requirements.

    Call this AFTER get_surgical_pathway. Uses the detector's visible_items
    from the current scene event and the case requirements to identify
    supply gaps, unaccounted items, and proposed corrective actions.

    Returns reconciliation results including all_present flag,
    actionable_missing items, unaccounted items, and proposed_tool_calls.
    """
    event = ctx.deps.event
    case = ctx.deps.case
    if case is None:
        return {"error": "No case data — call get_surgical_pathway first."}
    recon = reconcile_setup(event, case)
    # Store on deps for prompt context
    ctx.deps.reconciliation = recon
    ctx.deps.emit_tool_progress("reconcile_instruments")
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="reconcile_instruments",
                      tool_args={},
                      tool_result=recon,
                      detail=f"reconcile_instruments()")
        ctx.deps.emit("reconcile",
                      all_present=recon["all_present"],
                      actionable_missing=recon.get("actionable_missing_detail", recon["actionable_missing"]),
                      unaccounted=recon.get("unaccounted_detail", recon["unaccounted"]),
                      detail=_reconcile_detail(recon))
    return recon


def _reconcile_detail(recon: dict) -> str:
    if recon["all_present"]:
        return "All present"
    gaps = len(recon["actionable_missing"])
    unacc = len(recon["unaccounted"])
    return f"{gaps} gaps, {unacc} unaccounted"


@or_agent.tool
def create_or_task(
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
    ctx.deps.emit_tool_progress("create_or_task")
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="create_or_task",
                      tool_args={"case_id": case_id, "task_type": task_type, "priority": priority, "summary": summary},
                      tool_result=result,
                      detail=f"create_or_task(task_type={task_type})")
        ctx.deps.emit("tasks",
                      task_type=task_type,
                      priority=priority,
                      summary=summary,
                      detail=summary)
        # Porter tasks also appear in the SPD/Porter node
        if task_type in ("porter_hold", "porter_release"):
            ctx.deps.emit("spd",
                          item_name=task_type.replace("_", " "),
                          urgency=priority,
                          delivery_method="porter",
                          detail=f"Porter: {summary}")
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
    ctx.deps.emit_tool_progress("request_spd_resupply")
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="request_spd_resupply",
                      tool_args={"item_name": item_name, "room_id": room_id, "urgency": urgency},
                      tool_result=result,
                      detail=f"request_spd_resupply(item={item_name})")
        ctx.deps.emit("spd",
                      item_name=item_name,
                      urgency=urgency,
                      delivery_method="nurse",
                      detail=f"Nurse Runner: {item_name}")
    return result


@or_agent.tool
def set_or_prep_light(
    ctx: RunContext[AgentDeps],
    room_id: str,
    color: str,
    reason: str,
) -> dict:
    """Set the OR prep status light.

    Use green for logistics-ready, yellow for review-needed,
    red only for safety exceptions.

    Args:
        room_id: The OR room identifier.
        color: One of green, yellow, red.
        reason: Short explanation of why this color was chosen.
    """
    result = {
        "room_id": room_id,
        "color": color,
        "reason": reason,
        "status": "set",
    }
    ctx.deps.emit_tool_progress("set_or_prep_light")
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="set_or_prep_light",
                      tool_args={"room_id": room_id, "color": color, "reason": reason},
                      tool_result=result,
                      detail=f"set_or_prep_light(color={color})")
        ctx.deps.emit("prep_light",
                      color=color,
                      reason=reason,
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

    ctx.deps.emit_tool_progress("inspect_scene_local")
    if ctx.deps.emit:
        ctx.deps.emit("vlm_local", question=question, detail="Local VLM: working\u2026")

    suffix = resolved.suffix.lower().lstrip(".")
    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}
    mime_subtype = mime_map.get(suffix)
    if not mime_subtype:
        return {"error": f"unsupported image format: {suffix}"}

    encoded = base64.b64encode(resolved.read_bytes()).decode()
    data_url = f"data:image/{mime_subtype};base64,{encoded}"

    guided_json = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "answer": {"type": "boolean"},
        },
        "required": ["description", "answer"],
        "additionalProperties": False,
    }

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
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
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "vlm_response",
                "schema": guided_json,
                "strict": True,
            },
        },
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
    raw = r.json()["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(raw)
        answer = parsed.get("description", raw)
        verdict = parsed.get("answer")
    except (json.JSONDecodeError, AttributeError):
        answer = raw
        verdict = None
    result = {"image": str(resolved), "question": question, "answer": answer, "verdict": verdict}
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="inspect_scene_local",
                      tool_args={"image_path": image_path, "question": question},
                      tool_result=result,
                      detail=f"inspect_scene_local: {question[:80]}")
        ctx.deps.emit("vlm_local",
                      question=question,
                      answer=answer,
                      verdict=verdict,
                      detail=f"Local VLM: {question[:80]}")
    return result


@or_agent.tool
def inspect_scene_remote(
    ctx: RunContext[AgentDeps],
    image_path: str,
    question: str,
) -> dict:
    """Inspect an OR scene image using the REMOTE Azure VLM (Claude Opus 4-7).

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

    ctx.deps.emit_tool_progress("inspect_scene_remote")
    if ctx.deps.emit:
        ctx.deps.emit("vlm_remote", question=question, detail="Remote VLM: working\u2026")

    answer = ask_vlm(str(resolved), question)
    try:
        parsed = json.loads(answer)
        description = parsed.get("description", answer)
        verdict = parsed.get("answer")
    except (json.JSONDecodeError, AttributeError, TypeError):
        description = answer
        verdict = None
    result = {"image": str(resolved), "question": question, "answer": description, "verdict": verdict}
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="inspect_scene_remote",
                      tool_args={"image_path": image_path, "question": question},
                      tool_result=result,
                      detail=f"inspect_scene_remote: {question[:80]}")
        from apps.vlm.ask_vlm import AZURE_VLM_DEPLOYMENT
        ctx.deps.emit("vlm_remote",
                      question=question,
                      answer=description,
                      verdict=verdict,
                      detail=f"Remote VLM ({AZURE_VLM_DEPLOYMENT}): {question[:80]}")
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

    return {
        "actionable_missing": actionable_missing,
        "actionable_missing_detail": actionable_missing_detail,
        "flagged_no_deficit": flagged_no_deficit,
        "unaccounted": unaccounted,
        "unaccounted_detail": unaccounted_detail,
        "all_present": truly_clear,
        "proposed_tool_calls": calls,
    }


@logfire.instrument("ask_agent case_id={event[case_id]} event_type={event[event_type]}")
def ask_agent(event: dict, resources: dict, emit=None) -> dict:
    deps = AgentDeps(
        event=event,
        resources=resources,
        emit=emit,
    )

    prompt = json.dumps(
        {
            "event": {k: v for k, v in event.items() if k != "missing_or_uncertain"},
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
        if tc["name"] == "create_or_task"
    )

    # Count LLM iterations (each ModelResponse = one LLM round-trip)
    llm_iterations = sum(
        1 for msg in result.all_messages()
        if getattr(msg, "kind", "") == "response"
    )

    return {
        "decision_summary": result.output if isinstance(result.output, str) else str(result.output),
        "tool_calls": tool_calls,
        "requires_human_review": has_human_review,
        "llm_iterations": llm_iterations,
    }


def main():
    import argparse
    from apps.agent.validation import validate_decision

    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    args = parser.parse_args()

    event = load_event(args.scenario)
    resources = get_resources(event["room_id"])
    decision = ask_agent(event, resources)

    errors = validate_decision(decision, event)
    if errors:
        print(json.dumps({"validation_errors": errors, "decision": decision}, indent=2))
        raise SystemExit(2)

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()