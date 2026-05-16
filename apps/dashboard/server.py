"""Dashboard server — SSE broadcast of OR-edge-agent events."""

import asyncio
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import logfire
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from starlette.responses import StreamingResponse

logfire.configure(service_name="or-edge-agent-dashboard")

# ── Paths ────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
SCENARIOS_DIR = ROOT_DIR / "scenarios"

_subscribers: list[asyncio.Queue] = []
_last_states: dict[str, dict] = {}  # component_id → last event


def _broadcast(event: dict):
    """Push an event dict to every connected SSE client."""
    comp = event.get("component")
    if comp:
        _last_states[comp] = event
    data = json.dumps(event)
    for q in list(_subscribers):
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _subscribers.clear()


app = FastAPI(title="OR Edge Agent Dashboard", lifespan=lifespan)
logfire.instrument_fastapi(app)


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/data/{filepath:path}")
async def serve_data(filepath: str):
    """Serve images from data/."""
    target = (DATA_DIR / filepath).resolve()
    if not str(target).startswith(str(DATA_DIR.resolve())):
        return {"error": "forbidden"}
    if not target.exists():
        return {"error": "not found"}
    return FileResponse(target)


@app.get("/api/images")
async def list_images():
    """List sample images available in data/."""
    images = []
    for p in sorted(DATA_DIR.rglob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            rel = str(p.relative_to(DATA_DIR))
            images.append({"path": rel, "url": f"/data/{rel}", "name": p.stem})
    return images


@app.get("/api/cases")
async def list_cases():
    """List available surgical cases from the EMR API."""
    import httpx
    try:
        # The EMR API doesn't have a list endpoint, so we use known case IDs
        case_ids = ["CASE-1042", "CASE-2001", "CASE-3001", "CASE-4001", "CASE-5001"]
        cases = []
        async with httpx.AsyncClient() as client:
            for cid in case_ids:
                r = await client.get(f"http://localhost:9000/cases/{cid}", timeout=5)
                if r.status_code == 200:
                    c = r.json()
                    cases.append({
                        "case_id": c["case_id"],
                        "procedure": c.get("procedure", ""),
                        "phase": c.get("phase", ""),
                        "required_items": c.get("required_items", []),
                    })
        return cases
    except Exception:
        return []


@app.get("/api/scenarios")
async def list_scenarios():
    """List available scenario files."""
    scenarios = []
    for p in sorted(SCENARIOS_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        scenarios.append({
            "name": p.stem,
            "case_id": data.get("case_id"),
            "event_type": data.get("event_type"),
        })
    return scenarios


@app.get("/events")
async def sse(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    _subscribers.append(q)

    async def stream():
        try:
            # Send current state snapshot on connect
            for data in _last_states.values():
                yield f"data: {json.dumps(data)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _subscribers.remove(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/ingest")
async def ingest(request: Request):
    """Receive a component event and broadcast it."""
    event = await request.json()
    _broadcast(event)
    return {"ok": True}


@app.get("/state")
async def state():
    """Return the full current state snapshot."""
    return _last_states


# ── Scenario runner ──────────────────────────────────────────────────


def _ts():
    return datetime.now(timezone.utc).isoformat()


def _emit(component: str, **kwargs):
    """Broadcast a dashboard event synchronously (called from thread)."""
    event = {"component": component, "timestamp": _ts(), **kwargs}
    _broadcast(event)


def _reconcile_detail(recon: dict) -> str:
    if recon["all_present"]:
        return "All present"
    gaps = len(recon["actionable_missing"])
    unacc = len(recon["unaccounted"])
    return f"{gaps} gaps, {unacc} unaccounted"


@app.post("/run/{scenario}")
async def run_scenario(scenario: str):
    """Run a scenario through the agent pipeline and stream events to SSE."""
    scenario_file = SCENARIOS_DIR / f"{scenario}.json"
    if not scenario_file.exists():
        return {"error": f"Scenario {scenario} not found"}

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_pipeline, str(scenario_file))
    return {"ok": True}


def _run_pipeline(scenario_path: str):
    """Execute the full agent pipeline, emitting events at each step."""
    import httpx

    from apps.agent.validation import validate_decision

    # 1. Load event
    event = json.loads(Path(scenario_path).read_text())
    scenario_name = Path(scenario_path).stem

    # Find matching frame image
    image_url = None
    frame = None
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = DATA_DIR / "frames" / f"frame_{scenario_name}{ext}"
        if candidate.exists():
            frame = candidate
            image_url = f"/data/frames/frame_{scenario_name}{ext}"
            break
    if frame is None:
        # Fallback: use first available image
        for p in DATA_DIR.rglob("*.jpg"):
            image_url = f"/data/{p.relative_to(DATA_DIR)}"
            break

    # Run Edge Impulse object detection (if model is available)
    detection_image_url = None
    inference_ms = None
    detected_labels = []
    if frame is not None and frame.exists():
        from apps.detector.inference import detect
        det_result = detect(frame)
        if det_result.annotated_path:
            detection_image_url = f"/data/{det_result.annotated_path}"
            inference_ms = det_result.inference_ms
            detected_labels = [d.label for d in det_result.detections]

    visible = event.get("visible_items", {})
    total_visible = sum(visible.values()) if isinstance(visible, dict) else len(visible)

    _emit("detector",
          event_type=event.get("event_type"),
          confidence=event.get("confidence"),
          visible_items=visible,
          missing_items=event.get("missing_or_uncertain", []),
          image_url=image_url,
          detection_image_url=detection_image_url,
          inference_ms=inference_ms,
          detected_labels=detected_labels,
          detail=f"Detected {total_visible} items, "
                 f"{len(event.get('missing_or_uncertain', []))} uncertain")

    # 2. Fetch case from EMR
    case_id = event["case_id"]
    r = httpx.get(f"http://localhost:9000/cases/{case_id}", timeout=10)
    r.raise_for_status()
    case = r.json()
    _emit("emr_api",
          case_id=case_id,
          procedure=case.get("procedure", ""),
          detail=f"Fetched case {case_id}: {case.get('procedure', '?')}")

    # 3. Reconcile
    from apps.agent.run_fixture import reconcile_setup
    recon = reconcile_setup(event, case)
    _emit("reconcile",
          all_present=recon["all_present"],
          actionable_missing=recon["actionable_missing"],
          unaccounted=recon["unaccounted"],
          detail=_reconcile_detail(recon))

    # 4. Resources
    resources = {
        "room_id": event.get("room_id", "OR-?"),
        "sterile_processing_robot": {"available": True, "eta_seconds": 180},
        "human_runner": {"available": True, "eta_seconds": 420},
        "porter": {"available": True, "eta_seconds": 300},
    }
    _emit("resources", detail="Robot: 180s | Runner: 420s | Porter: 300s")

    # 5. Agent decision (import inline to avoid circular / heavy load)
    _emit("agent", status="thinking", detail="LLM processing…")

    from apps.agent.run_fixture import ask_agent
    decision = ask_agent(event, case, resources)

    tool_calls = decision.get("tool_calls", [])
    _emit("agent",
          status="done",
          tool_calls=len(tool_calls),
          summary=decision.get("decision_summary", "")[:200],
          detail=f"{len(tool_calls)} tool call(s)")

    # 6. Emit individual tool call results
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})

        if name == "create_synthetic_or_task":
            _emit("tasks",
                  task_type=args.get("task_type"),
                  priority=args.get("priority"),
                  summary=args.get("summary", ""),
                  detail=args.get("summary", ""))
        elif name == "set_or_prep_light":
            _emit("prep_light",
                  color=args.get("color", "yellow"),
                  duration_seconds=args.get("duration_seconds"),
                  detail=f"Set to {args.get('color')} for {args.get('duration_seconds')}s")
        elif name in ("request_spd_resupply", "request_spd_robot_delivery"):
            _emit("spd",
                  item_name=args.get("item_name"),
                  urgency=args.get("urgency"),
                  detail=f"{'Robot' if 'robot' in name else 'Runner'}: {args.get('item_name')}")
        elif name == "inspect_scene_local":
            _emit("vlm_local",
                  answer=args.get("question", ""),
                  image_url=image_url,
                  detail=f"Local VLM (Ministral 3B): {args.get('question', '')[:80]}")
        elif name == "inspect_scene_remote":
            _emit("vlm_remote",
                  answer=args.get("question", ""),
                  image_url=image_url,
                  detail=f"Remote VLM (GPT-4o): {args.get('question', '')[:80]}")

    # 7. Validation
    errors = validate_decision(decision, event)
    _emit("validation",
          errors=errors,
          detail=f"{'Passed' if not errors else f'{len(errors)} error(s): ' + '; '.join(errors)}")
