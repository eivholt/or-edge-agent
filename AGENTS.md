# OR Edge Agent Guidelines

This file is the canonical project context for coding agents. Keep it current
when architecture, behavioral contracts, or verified test status changes.

## Safety And Scope

- This is a synthetic operating-room logistics demo. All records are synthetic.
- Never turn this into clinical diagnosis, treatment selection, case clearance,
  or real clinical alarm software.
- Keep actions operational and bounded: supply delivery, review workflows,
  stacklight state, logging, and demo integrations.

## Current Pipeline

1. A scenario supplies exactly `case_id`, `room_id`, and `image_path`.
2. Edge Impulse detects instruments and populates `visible_items` at runtime.
3. The agent fetches EMR `required_items` and compares quantity counts.
4. The local or cloud VLM inspects the scene and returns a sterile-zone verdict.
5. The agent performs every applicable action and sets one stacklight state.

Owning files:

- `apps/detector/inference.py`: Edge Impulse inference and runtime counts.
- `apps/detector/models.py`: minimal `ORSceneEvent` contract.
- `apps/agent/reconcile.py`: pure quantity reconciliation, no LLM or network.
- `apps/agent/run_fixture.py`: pydantic-ai agent, prompts, and native tools.
- `apps/agent/validation.py`: allowed tool and argument validation.
- `synthetic_emr/api.py`: synthetic cases and required item counts.
- `apps/dashboard/server.py`: pipeline orchestration and SSE events.

## Behavioral Contract

The native agent tools are `get_case`, `check_supplies`, `inspect_scene`,
`request_resupply`, `create_task`, and `set_stacklight`.

- Supply deficit -> `request_resupply` only.
- Never call `create_task` for a supply deficit.
- Sterile-zone `verdict=true` -> red light and one `human_review` task.
- Deficit without sterile violation -> yellow light.
- No deficit and no sterile violation -> green light.
- Every run must contain exactly one `set_stacklight` call.
- `reconcile()` returns deficit dictionaries shaped as `{item, have, need}`.

## Deliberately Removed Inputs

Do not restore these as scenario or decision hints without an explicit design
change: `event_id`, `event_type`, `zone`, `timestamp`, `confidence`,
`missing_or_uncertain`, `vlm_hint`, `phase`, `open_items`, and
`porter_release_allowed`.

Scenarios intentionally contain no detector results or expected-answer hints.
This is required to test whether the detector and agent handle each case.
Procedure text alone must not trigger review or alter the stacklight.

## VLM Decisions

- `inspect_scene` is a native tool in `apps/agent/run_fixture.py`.
- It uses local Ministral when `cloud_connected` is false and Azure when true.
- Every local inspection crops context around each detector centroid and sends
  each segment to Ministral with guided JSON: `{answer: boolean}`.
- Detector geometry may locate and order context, but it must never produce,
  bypass, or override the sterile-zone verdict. Do not add color heuristics.
- A scene is positive when any segment VLM returns `true`. Segment thumbnails,
  progress, verdicts, and durations are emitted to the dashboard.
- Keep sterile workflow policy deterministic and outside the small model.
- Ministral 3B is sensitive to tool names, docstrings, and result wording. Keep
  contracts short and explicit.
- Do not reintroduce `pending_actions` or `proposed_tool_calls` hints merely to
  force tool use; that defeats the autonomous-agent evaluation.

Standalone modules under `mcp_servers/` are alternate integration surfaces.
The current `ask_agent` execution path uses native tools in `run_fixture.py`.

## Detector Model

- Detector classes are scalpel, scissors, sponge, and tweezers.
- `models/modelfile.eim` must be the float32 Edge Impulse Linux x86_64 model.
- `models/modelfile.aarch64.eim` is the float32 AArch64 runner selected on ARM.
- The confidence threshold is 0.9.
- The model is tracked with Git LFS; clones must run `git lfs install`.
- `models/modelfile.eim.int8.bak` stays ignored because int8 produced bad
  classifications.
- The download script must use `ENGINE="tflite"`, not `tflite-eon`.

## Running And Testing

Services:

- IQ9 text/tool adapter: port 8001; C++ Genie upstream: port 8911.
- IQ9 multimodal `llama-server`: port 8082.
- Synthetic EMR API: port 9000, `.venv`.
- Dashboard: port 8000, `.venv`.
- On IQ9, `./start.sh app` manages EMR and dashboard while model runtimes are
  externally managed. See `IQ9_EVK.md`.

Commands:

```bash
source .venv/bin/activate
python -m pytest tests/ -m 'not llm' -q --tb=short
python -m pytest tests/ -v --tb=short
```

Verified 2026-07-21: 29 service-independent tests pass on AArch64. The 64-token
centroid benchmark processed all 42 detections across five cases but scored 2/5;
256-token probes repaired the three decisive crop errors at approximately 116
seconds per segment. On-device VLM limitations are documented in `IQ9_EVK.md`.
Do not claim the full suite passes without running it against required services.

## Repository Hygiene

- Keep `.env`, virtual environments, logs, generated detections, Logfire
  credentials, and model backups out of Git.
- `.env.example` must list every supported environment variable without secrets.
- Known stale utility: `apps/detector/fixture_detector.py` references removed
  `scenarios/procedure_changed.json`; repair it before relying on that utility.
- Make focused changes and do not revive removed workflow features as incidental
  cleanup.