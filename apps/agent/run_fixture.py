import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent, ModelRetry, ModelSettings, RunContext, Tool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from apps.agent.reconcile import reconcile

load_dotenv(override=True)

logfire.configure(service_name="or-edge-agent")
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "local-dev-key")
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8081/v1")
VLM_MODEL = os.getenv("VLM_MODEL", "mistralai/Ministral-3-3B-Instruct-2512-BF16")
VLM_API_KEY = os.getenv("VLM_API_KEY", OPENAI_API_KEY)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", VLM_BASE_URL)
LLM_MODEL = os.getenv("LLM_MODEL", VLM_MODEL)
LLM_API_KEY = os.getenv("LLM_API_KEY", OPENAI_API_KEY)
EMR_BASE_URL = os.getenv("EMR_BASE_URL", "http://localhost:9000")
VLM_TIMEOUT_SECONDS = float(os.getenv("VLM_TIMEOUT_SECONDS", "600"))
VLM_SEGMENT_RADIUS = int(os.getenv("VLM_SEGMENT_RADIUS", "64"))
VLM_SEGMENT_IMAGE_SIZE = int(os.getenv("VLM_SEGMENT_IMAGE_SIZE", "224"))

VLM_SYSTEM_PROMPT = (
    "You are a surgical-scene analyst. Respond ONLY with JSON: "
    '{"answer": true/false}. Set "answer" to true if an instrument rests on '
    "bare table outside or across the green sterile drape edge."
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
      - set yellow if check_supplies shows deficits.
      - set green ONLY when: no sterile issue AND no deficits.
   b) If inspect_scene verdict is true → call create_task(human_review) for the sterile zone issue.
   c) Deficit items → call request_resupply for each one. That is all — move on.

ACTION MAPPING (use exactly the right tool for each situation):
- deficit item              → request_resupply (the ONLY correct tool for deficits)
- sterile zone verdict=true → create_task

RULES:
- Do NOT create tasks for deficit items. Only use request_resupply for deficits.
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
    resupplied_items: list = None
    stacklight_colors: list = None
    review_task_count: int = 0
    sterile_verdict: bool | None = None

    def __post_init__(self):
        if self._tools_used is None:
            self._tools_used = []
        if self.resupplied_items is None:
            self.resupplied_items = []
        if self.stacklight_colors is None:
            self.stacklight_colors = []

    _tool_display_names = {
        "create_task": "create task",
        "set_stacklight": "set stacklight",
    }

    _MAX_CONTEXT = int(os.getenv("LLM_MAX_CONTEXT", "8192"))
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
    LLM_MODEL,
    provider=OpenAIProvider(base_url=LLM_BASE_URL, api_key=LLM_API_KEY),
    profile=OpenAIModelProfile(openai_supports_strict_tool_definition=False),
)

or_agent = Agent(
    _model,
    instructions=INSTRUCTIONS,
    deps_type=AgentDeps,
    model_settings=ModelSettings(temperature=0, max_tokens=1024),
    retries={"output": 3},
)


def _expected_stacklight_color(deps: AgentDeps) -> str:
    deficits = deps.reconciliation.get("deficits", []) if deps.reconciliation else []
    if deps.sterile_verdict is True:
        return "red"
    if deficits:
        return "yellow"
    return "green"


@or_agent.output_validator
def validate_agent_completion(ctx: RunContext[AgentDeps], output: str) -> str:
    """Keep hard workflow completion rules outside the language model."""
    missing_actions = []
    reconciliation = ctx.deps.reconciliation
    if ctx.deps.case is None:
        missing_actions.append("call get_case")
    if reconciliation is None:
        missing_actions.append("call check_supplies")

    deficits = reconciliation.get("deficits", []) if reconciliation else []
    for deficit in deficits:
        if deficit["item"] not in ctx.deps.resupplied_items:
            missing_actions.append(
                f'call request_resupply for {deficit["item"]}'
            )

    has_visible_items = bool(ctx.deps.event.get("visible_items"))
    if has_visible_items and ctx.deps.sterile_verdict is None:
        missing_actions.append("call inspect_scene")

    if ctx.deps.sterile_verdict is True and ctx.deps.review_task_count != 1:
        missing_actions.append("call create_task once with task_type human_review")

    expected_color = _expected_stacklight_color(ctx.deps)
    if ctx.deps.stacklight_colors != [expected_color]:
        if not ctx.deps.stacklight_colors:
            missing_actions.append(f"call set_stacklight once with color {expected_color}")
        else:
            missing_actions.append(
                f"use exactly one set_stacklight call with color {expected_color}; "
                f"colors already used: {ctx.deps.stacklight_colors}"
            )

    if missing_actions:
        raise ModelRetry(
            "The workflow is incomplete. Before finishing, "
            + "; ".join(missing_actions)
            + ". Do not repeat tools that already succeeded."
        )
    return output


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
    started = time.perf_counter()
    r = httpx.get(f"{EMR_BASE_URL}/cases/{case_id}", timeout=10)
    r.raise_for_status()
    case = r.json()
    # Strip fields irrelevant to agent decisions
    for key in ("patient_id",):
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
                      duration_ms=(time.perf_counter() - started) * 1000,
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
    started = time.perf_counter()
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
                      duration_ms=(time.perf_counter() - started) * 1000,
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
    """Create a workflow task for sterile zone issues only.

    Use ONLY when inspect_scene verdict is true (sterile zone issue).

    For supply deficits, use request_resupply instead.

    Args:
        case_id: The case identifier.
        task_type: One of human_review.
        priority: One of low, normal, high.
        summary: Short description of what needs review.
        reason: Why this task is being created.
    """
    if task_type == "human_review" and ctx.deps.review_task_count:
        return {
            "status": "ignored_duplicate",
            "task_type": task_type,
            "reason": "A human review task was already created for this run.",
        }

    result = {
        "status": "created",
        "case_id": case_id,
        "task_type": task_type,
        "priority": priority,
        "summary": summary,
        "reason": reason,
    }
    if task_type == "human_review":
        ctx.deps.review_task_count += 1
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
    if item_name in ctx.deps.resupplied_items:
        return {
            "status": "ignored_duplicate",
            "item_name": item_name,
            "reason": "This item was already requested for this run.",
        }

    result = {
        "request_id": f"SPD-{item_name}-{room_id}",
        "item_name": item_name,
        "room_id": room_id,
        "urgency": urgency,
        "status": "requested",
    }
    ctx.deps.resupplied_items.append(item_name)
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
    if ctx.deps.stacklight_colors:
        return {
            "status": "ignored_duplicate",
            "color": ctx.deps.stacklight_colors[0],
            "reason": "The stacklight was already set for this run.",
        }

    requested_color = color
    color = _expected_stacklight_color(ctx.deps)
    result = {
        "room_id": room_id,
        "color": color,
        "reason": reason,
        "status": "set",
    }
    if requested_color != color:
        result["requested_color"] = requested_color
        result["policy_enforced"] = True
    ctx.deps.stacklight_colors.append(color)
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
    "Inspect the entire image, including its edges. The green or turquoise cloth "
    "is the sterile drape; exposed gray or silver metal is outside it. Is any "
    "scissors, scalpel, tweezers, or white gauze sponge resting partly or entirely "
    "on exposed metal outside the green cloth?"
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

    if not ctx.deps.event.get("visible_items"):
        ctx.deps.sterile_verdict = False
        return {
            "image": image_path,
            "question": STERILE_ZONE_QUESTION,
            "answer": "Skipped because no instruments were detected.",
            "verdict": False,
            "skipped": True,
        }

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

    ctx.deps.sterile_verdict = result.get("verdict")
    return result


def _inspect_local(ctx: RunContext[AgentDeps], resolved: Path, image_path: str, question: str) -> dict:
    """Inspect detector-centered image segments with local Ministral 3B."""
    started = time.perf_counter()
    ctx.deps.emit_tool_progress("inspect_scene_local", ctx)

    detections = ctx.deps.resources.get("_scene_detections")
    if not detections:
        from apps.detector.inference import detect

        detection_result = detect(resolved)
        detections = {
            "frame_width": detection_result.frame_width,
            "frame_height": detection_result.frame_height,
            "items": [
                {
                    "label": detection.label,
                    "x": detection.x,
                    "y": detection.y,
                    "width": detection.width,
                    "height": detection.height,
                }
                for detection in detection_result.detections
            ],
        }

    segments = _build_vlm_segments(resolved, detections, VLM_SEGMENT_RADIUS)
    if not segments:
        raise ValueError("Local VLM inspection requires detector centroids")

    guided_json = {
        "type": "object",
        "properties": {"answer": {"type": "boolean"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    segment_results = []
    total_segments = len(segments)

    for index, segment in enumerate(segments, start=1):
        segment_started = time.perf_counter()
        segment_question = (
            f"Focus on the detected {segment['label']} centered in this image. Is "
            "the surface directly underneath it bare gray or silver metal rather "
            "than green or turquoise cloth? The gray instrument itself does not "
            "count. Return true for bare metal and false for green cloth."
        )
        progress = {
            "question": segment_question,
            "segments_total": total_segments,
            "segment_index": index,
            "segment_label": segment["label"],
            "segment_image_url": segment["image_url"],
        }
        if ctx.deps.emit:
            ctx.deps.emit(
                "vlm_local",
                **progress,
                segments_processed=index - 1,
                segment_status="running",
                segments=segment_results,
                detail=f"Segment {index}/{total_segments}: {segment['label']} working…",
            )

        payload = {
            "model": VLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": segment["data_url"]},
                        },
                        {"type": "text", "text": segment_question},
                    ],
                }
            ],
            "max_tokens": 16,
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
        response = httpx.post(
            f"{VLM_BASE_URL}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {VLM_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=VLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        segment_verdict = json.loads(raw).get("answer")
        if not isinstance(segment_verdict, bool):
            raise ValueError("Local VLM response did not contain a boolean answer")

        segment_result = {
            "index": index,
            "label": segment["label"],
            "verdict": segment_verdict,
            "crop_box": segment["crop_box"],
            "image_url": segment["image_url"],
            "duration_ms": (time.perf_counter() - segment_started) * 1000,
        }
        segment_results.append(segment_result)
        if ctx.deps.emit:
            ctx.deps.emit(
                "vlm_local",
                **progress,
                segments_processed=index,
                segment_status="complete",
                segment_verdict=segment_verdict,
                segment_duration_ms=segment_result["duration_ms"],
                segments=segment_results,
                detail=(
                    f"Segment {index}/{total_segments}: {segment['label']} "
                    f"answered {str(segment_verdict).lower()}"
                ),
            )

    verdict = any(segment["verdict"] for segment in segment_results)
    positives = [
        f"{segment['label']} #{segment['index']}"
        for segment in segment_results
        if segment["verdict"]
    ]
    answer = (
        f"Local VLM flagged {', '.join(positives)}."
        if positives
        else f"Local VLM cleared all {total_segments} detector segments."
    )
    result = {
        "image": str(resolved),
        "question": question,
        "answer": answer,
        "verdict": verdict,
        "segments": segment_results,
        "segments_processed": total_segments,
        "segments_total": total_segments,
    }
    if ctx.deps.emit:
        ctx.deps.emit(
            "tool_call",
            tool_name="inspect_scene",
            tool_args={"image_path": image_path, "question": question},
            tool_result=result,
            detail=f"inspect_scene (local): {question[:80]}",
        )
        ctx.deps.emit(
            "vlm_local",
            question=question,
            answer=answer,
            verdict=verdict,
            segments=segment_results,
            segments_processed=total_segments,
            segments_total=total_segments,
            duration_ms=(time.perf_counter() - started) * 1000,
            detail=f"Local VLM: {question[:80]}",
        )
    return result


def _build_vlm_segments(resolved: Path, detections: dict, radius: int) -> list[dict]:
    """Create and persist fixed-size image contexts around detector centroids."""
    from io import BytesIO

    from PIL import Image

    frame_width = detections.get("frame_width") or 0
    frame_height = detections.get("frame_height") or 0
    items = detections.get("items") or []
    if not frame_width or not frame_height or not items:
        return []

    source = Image.open(resolved).convert("RGB")
    scale_x = source.width / frame_width
    scale_y = source.height / frame_height
    crop_width = min(source.width, round(radius * 2 * scale_x))
    crop_height = min(source.height, round(radius * 2 * scale_y))
    output_dir = DATA_DIR / "segments"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.time_ns()
    segments = []

    for index, detection in enumerate(items, start=1):
        center_x = (detection["x"] + detection["width"] / 2) * scale_x
        center_y = (detection["y"] + detection["height"] / 2) * scale_y
        left = round(min(max(center_x - crop_width / 2, 0), source.width - crop_width))
        top = round(min(max(center_y - crop_height / 2, 0), source.height - crop_height))
        right = left + crop_width
        bottom = top + crop_height
        crop = source.crop((left, top, right, bottom))
        display_buffer = BytesIO()
        crop.save(display_buffer, format="JPEG", quality=90)
        display_bytes = display_buffer.getvalue()
        filename = f"{resolved.stem}_{run_id}_{index}_{detection['label']}.jpg"
        (output_dir / filename).write_bytes(display_bytes)

        inference_crop = crop.copy()
        inference_crop.thumbnail(
            (VLM_SEGMENT_IMAGE_SIZE, VLM_SEGMENT_IMAGE_SIZE),
            Image.Resampling.LANCZOS,
        )
        inference_buffer = BytesIO()
        inference_crop.save(inference_buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(inference_buffer.getvalue()).decode()
        segments.append(
            {
                "label": detection["label"],
                "crop_box": [left, top, right, bottom],
                "image_url": f"/data/segments/{filename}?t={run_id}",
                "data_url": f"data:image/jpeg;base64,{encoded}",
            }
        )

    return segments


def _inspect_remote(ctx: RunContext[AgentDeps], resolved: Path, image_path: str, question: str) -> dict:
    """Run VLM inspection via the remote cloud VLM."""
    from apps.vlm.ask_vlm import ask_vlm

    started = time.perf_counter()
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
                      duration_ms=(time.perf_counter() - started) * 1000,
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

    if all_present:
        summary = "All instrument types match or exceed requirements."
    else:
        parts = [f"{d['item']} ({d['have']}/{d['need']})" for d in deficits]
        summary = f"{len(deficits)} deficit(s): {', '.join(parts)}. Action: request_resupply for each."

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
            "resources": {
                key: value
                for key, value in resources.items()
                if key not in {"cloud_connected", "_scene_detections"}
            },
        },
        indent=2,
    )
    result = or_agent.run_sync(prompt, deps=deps)

    # Extract tool calls and match results in message order. Some local model
    # adapters reuse tool_call_id values across turns.
    tool_calls = []
    pending_calls = {}
    for msg in result.all_messages():
        if hasattr(msg, "parts"):
            for part in msg.parts:
                part_kind = getattr(part, "part_kind", "")
                if part_kind == "tool-call":
                    call_id = part.tool_call_id
                    tc = {
                        "name": part.tool_name,
                        "arguments": (
                            part.args if isinstance(part.args, dict)
                            else json.loads(part.args) if isinstance(part.args, str)
                            else {}
                        ),
                    }
                    tool_calls.append(tc)
                    if call_id:
                        pending_calls.setdefault(call_id, []).append(tc)
                elif part_kind == "tool-return":
                    waiting = pending_calls.get(part.tool_call_id, [])
                    if waiting:
                        waiting.pop(0)["result"] = part.content

    tool_calls = [
        tool_call
        for tool_call in tool_calls
        if not (
            isinstance(tool_call.get("result"), dict)
            and tool_call["result"].get("status") == "ignored_duplicate"
        )
    ]
    for tool_call in tool_calls:
        result_content = tool_call.get("result")
        if tool_call["name"] == "set_stacklight" and isinstance(result_content, dict):
            tool_call["arguments"] = {
                **tool_call["arguments"],
                "color": result_content["color"],
            }

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
    usage_total = result.usage
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