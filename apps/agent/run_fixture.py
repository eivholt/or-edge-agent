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
    "You are a surgical-scene analyst. Respond ONLY with JSON:\n"
    '{"description": "...", "answer": true/false}\n'
    "First describe the position of each instrument relative to the drape edge. "
    "Then set \"answer\" to true ONLY if you can see an instrument resting on "
    "the bare table beyond the drape. "
    "If all instruments are on the drape, set \"answer\" to false."
)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

INSTRUCTIONS = """\
You are a local OR logistics agent (synthetic demo). Coordinate supply and review workflows. Never diagnose or prescribe.

WORKFLOW — always follow this order:
1. Call get_case(case_id) to fetch the EMR case and required equipment.
2. Call check_supplies() to compare detected items against requirements.
3. Call inspect_scene to check the sterile zone (see its docstring for when to call).
   The image_path MUST be taken from the event's "image_path" field — do NOT invent a path.
4. Now take ALL applicable actions together:
   a) Call set_stacklight:
      - set red if inspect_scene verdict is true (sterile zone issue).
      - set yellow if check_supplies shows deficits OR get_case returned open_items.
      - set green ONLY when: no sterile issue, no deficits, AND open_items is empty.
   b) If inspect_scene verdict is true, call create_task with task_type="human_review" for the sterile zone issue.
   c) For EACH deficit item from check_supplies, call request_resupply.
   d) If the case open_items list is non-empty, call create_task with task_type="human_review".

RULES:
- open_items from get_case ALWAYS means yellow — never set green when open_items exist.
- Deficits need request_resupply — never use create_task for a deficit item.
- Do NOT create tasks based on missing_or_uncertain from the event. Only act on deficits from check_supplies and open_items from get_case.
- Every run MUST include exactly one set_stacklight call.
- Never ask the user for input. Always decide and act autonomously.
- Your final text response must be a SHORT summary (2-4 sentences max)."""


# ── Dependencies (passed via RunContext) ─────────────────────────────


@dataclass
class AgentDeps:
    event: dict
    resources: dict
    case: dict = None  # populated by get_case tool
    reconciliation: dict = None  # populated by check_supplies tool
    emit: object = None  # optional SSE callback: emit(component, **kwargs)
    _tool_count: int = 0  # tracks tool calls for iteration progress
    _tools_used: list = None  # names of tools called so far

    def __post_init__(self):
        if self._tools_used is None:
            self._tools_used = []

    _tool_display_names = {
        "create_task": "create task",
        "set_stacklight": "set stacklight",
    }

    _MAX_CONTEXT = 8192  # Ministral 3B context window
    _prev_cumulative_input: int = 0  # for computing per-request input

    def emit_tool_progress(self, tool_name: str, ctx=None):
        """Emit an agent status update showing tool progress."""
        self._tool_count += 1
        display = self._tool_display_names.get(tool_name, tool_name.replace("_", " "))
        self._tools_used.append(display)
        if self.emit:
            progress = " → ".join(self._tools_used)
            extra = {}
            if ctx and hasattr(ctx, "usage") and ctx.usage and ctx.usage.input_tokens:
                # input_tokens is cumulative; compute per-request delta = current context size
                cumulative = ctx.usage.input_tokens
                per_request = cumulative - self._prev_cumulative_input
                self._prev_cumulative_input = cumulative
                extra["context_tokens"] = per_request
                extra["max_context"] = self._MAX_CONTEXT
            self.emit("agent", status="thinking",
                      summary=progress,
                      detail=progress,
                      **extra)


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
    model_settings=ModelSettings(temperature=0, max_tokens=1024),
)


# ── Tools (dynamically discoverable via native tool calling) ─────────


@or_agent.tool
def get_case(
    ctx: RunContext[AgentDeps],
    case_id: str,
) -> dict:
    """Fetch the surgical case and required equipment list from the EMR.

    Call this FIRST to learn what instruments/supplies the procedure requires.

    Args:
        case_id: The surgical case identifier (e.g. CASE-1042).
    """
    r = httpx.get(f"{EMR_BASE_URL}/cases/{case_id}", timeout=10)
    r.raise_for_status()
    case = r.json()
    # Strip fields irrelevant to agent decisions to avoid 3B hallucinations
    for key in ("porter_release_allowed", "patient_id", "phase", "priority"):
        case.pop(key, None)
    ctx.deps.case = case
    ctx.deps.emit_tool_progress("get_case", ctx)
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="get_case",
                      tool_args={"case_id": case_id},
                      tool_result={"procedure": case.get("procedure"), "required_items": case.get("required_items")},
                      detail=f"get_case(case_id={case_id})")
        ctx.deps.emit("emr_api",
                      case_id=case_id,
                      procedure=case.get("procedure", ""),
                      required_items=case.get("required_items", {}),
                      detail=f"Fetched case {case_id}: {case.get('procedure', '?')}")
    return case


@or_agent.tool
def check_supplies(
    ctx: RunContext[AgentDeps],
) -> dict:
    """Compare detected instruments against the surgical case requirements.

    Call this AFTER get_case. Returns all_present flag and
    a list of deficits with item name, have count, and need count.
    """
    event = ctx.deps.event
    case = ctx.deps.case
    if case is None:
        return {"error": "No case data — call get_case first."}
    recon = reconcile_setup(event, case)
    ctx.deps.reconciliation = recon
    ctx.deps.emit_tool_progress("check_supplies", ctx)
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="check_supplies",
                      tool_args={},
                      tool_result=recon,
                      detail="check_supplies()")
        ctx.deps.emit("reconcile",
                      all_present=recon["all_present"],
                      actionable_missing=[f"{d['item']} ({d['have']}/{d['need']})" for d in recon.get("deficits", [])],
                      unaccounted=[],
                      detail=_reconcile_detail(recon))
    return recon


def _reconcile_detail(recon: dict) -> str:
    if recon["all_present"]:
        return "All present"
    n = len(recon.get("deficits", []))
    return f"{n} deficit(s)"


@or_agent.tool
def create_task(
    ctx: RunContext[AgentDeps],
    case_id: str,
    task_type: str,
    priority: str,
    summary: str,
    reason: str,
) -> dict:
    """Create a workflow task.

    Args:
        case_id: The case identifier.
        task_type: One of human_review.
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
    ctx.deps.emit_tool_progress("create_task", ctx)
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="create_task",
                      tool_args={"case_id": case_id, "task_type": task_type, "priority": priority, "summary": summary},
                      tool_result=result,
                      detail=f"create_task(task_type={task_type})")
        ctx.deps.emit("tasks",
                      task_type=task_type,
                      priority=priority,
                      summary=summary,
                      detail=summary)
    return result


@or_agent.tool
def request_resupply(
    ctx: RunContext[AgentDeps],
    item_name: str,
    room_id: str,
    urgency: str,
) -> dict:
    """Request sterile processing delivery for a missing item.

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
    ctx.deps.emit_tool_progress("request_resupply", ctx)
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="request_resupply",
                      tool_args={"item_name": item_name, "room_id": room_id, "urgency": urgency},
                      tool_result=result,
                      detail=f"request_resupply(item={item_name})")
        ctx.deps.emit("spd",
                      item_name=item_name,
                      urgency=urgency,
                      delivery_method="nurse",
                      detail=f"Nurse Runner: {item_name}")
    return result


@or_agent.tool
def set_stacklight(
    ctx: RunContext[AgentDeps],
    room_id: str,
    color: str,
    reason: str,
) -> dict:
    """Set the OR prep status stacklight.

    Green = logistics-ready, Yellow = supply deficit, Red = sterile contamination.

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
    ctx.deps.emit_tool_progress("set_stacklight", ctx)
    if ctx.deps.emit:
        ctx.deps.emit("tool_call",
                      tool_name="set_stacklight",
                      tool_args={"room_id": room_id, "color": color, "reason": reason},
                      tool_result=result,
                      detail=f"set_stacklight(color={color})")
        ctx.deps.emit("prep_light",
                      color=color,
                      reason=reason,
                      detail=f"Set to {color}")
    return result


STERILE_ZONE_QUESTION = (
    "Are any sponge, scissors, tweezers, scalpel on the bare table outside the sterile drape?"
)


@or_agent.tool
def inspect_scene(
    ctx: RunContext[AgentDeps],
    image_path: str,
) -> dict:
    """Inspect the OR scene image for sterile zone violations.

    Call this tool whenever at least one instrument was detected by the EI model
    (i.e. visible_items is non-empty). Skip it only if no objects were detected.

    Sends the image to a VLM with a fixed sterile-zone question that covers
    all Edge Impulse instrument classes (sponge, scissors, tweezers, scalpel).

    Returns a dict with:
      - "answer": text description from the VLM
      - "verdict": boolean — true means YES there IS an issue, false means NO issue found

    IMPORTANT — check the "verdict" field in the result:
      verdict=true  → sterile zone issue exists → set red light and create human_review task
      verdict=false → sterile zone is clear → do NOT create a human_review task for sterile issues

    Args:
        image_path: Path to an image file (relative to data/ or absolute).
    """

    question = STERILE_ZONE_QUESTION

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

    cloud = ctx.deps.resources.get("cloud_connected", False)

    if cloud:
        result = _inspect_remote(ctx, resolved, image_path, question)
    else:
        result = _inspect_local(ctx, resolved, image_path, question)

    return result


def _inspect_local(ctx: RunContext[AgentDeps], resolved: Path, image_path: str, question: str) -> dict:
    """Run VLM inspection via the local Ministral 3B model."""
    ctx.deps.emit_tool_progress("inspect_scene_local", ctx)
    if ctx.deps.emit:
        ctx.deps.emit("vlm_local", question=question, detail="Local VLM: working\u2026")

    suffix = resolved.suffix.lower().lstrip(".")
    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}
    mime_subtype = mime_map.get(suffix)
    if not mime_subtype:
        return {"error": f"unsupported image format: {suffix}"}

    # Upscale small images so the VLM can reason about spatial positions
    from PIL import Image
    import io
    img = Image.open(resolved)
    if img.width < 512 or img.height < 512:
        img = img.resize((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format=mime_subtype.upper().replace("JPEG", "JPEG"))
        encoded = base64.b64encode(buf.getvalue()).decode()
    else:
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
                      tool_name="inspect_scene",
                      tool_args={"image_path": image_path, "question": question},
                      tool_result=result,
                      detail=f"inspect_scene (local): {question[:80]}")
        ctx.deps.emit("vlm_local",
                      question=question,
                      answer=answer,
                      verdict=verdict,
                      detail=f"Local VLM: {question[:80]}")
    return result


def _inspect_remote(ctx: RunContext[AgentDeps], resolved: Path, image_path: str, question: str) -> dict:
    """Run VLM inspection via the remote cloud VLM."""
    from apps.vlm.ask_vlm import ask_vlm

    ctx.deps.emit_tool_progress("inspect_scene_remote", ctx)
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
                      tool_name="inspect_scene",
                      tool_args={"image_path": image_path, "question": question},
                      tool_result=result,
                      detail=f"inspect_scene (remote): {question[:80]}")
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
    deficits = reconcile(event, case)
    all_present = len(deficits) == 0

    # Check if the case has unresolved open items or a procedure change
    open_items = case.get("open_items", [])
    phase = case.get("phase", "")
    needs_review = bool(open_items) or "changed" in phase

    pending = [item.replace("_", " ") for item in open_items]

    if all_present and not needs_review:
        summary = "All instrument types match or exceed requirements."
    elif all_present and needs_review:
        summary = (
            "All instrument types match or exceed requirements, "
            f"but case has pending tasks requiring review: {', '.join(pending)}. "
            "Create human_review task."
        )
    else:
        parts = [f"{d['item']} ({d['have']}/{d['need']})" for d in deficits]
        summary = f"{len(deficits)} deficit(s): {', '.join(parts)}. Request resupply for each."
        if needs_review:
            summary += f" Pending tasks also require review: {', '.join(pending)}."

    return {
        "all_present": all_present,
        "deficits": deficits,
        "summary": summary,
    }


@logfire.instrument("ask_agent case_id={event[case_id]}")
def ask_agent(event: dict, resources: dict, emit=None) -> dict:
    deps = AgentDeps(
        event=event,
        resources=resources,
        emit=emit,
    )

    prompt = json.dumps(
        {
            "event": event,
            "resources": {k: v for k, v in resources.items() if k != "cloud_connected"},
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
        if tc["name"] == "create_task"
    )

    # Count LLM iterations (each ModelResponse = one LLM round-trip)
    llm_iterations = sum(
        1 for msg in result.all_messages()
        if getattr(msg, "kind", "") == "response"
    )

    # Collect per-iteration token usage for context gauge
    usage_total = result.usage()
    context_usage = {
        "input_tokens": usage_total.input_tokens,
        "output_tokens": usage_total.output_tokens,
        "requests": usage_total.requests,
        "max_context": AgentDeps._MAX_CONTEXT,
    }
    # Peak input tokens = max across iterations
    peak_input = 0
    for msg in result.all_messages():
        if getattr(msg, "kind", "") == "response":
            u = getattr(msg, "usage", None)
            if u and u.input_tokens:
                peak_input = max(peak_input, u.input_tokens)
    context_usage["peak_input_tokens"] = peak_input

    return {
        "decision_summary": result.output if isinstance(result.output, str) else str(result.output),
        "tool_calls": tool_calls,
        "requires_human_review": has_human_review,
        "llm_iterations": llm_iterations,
        "context_usage": context_usage,
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