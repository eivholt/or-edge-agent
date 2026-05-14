import json
import os
from pathlib import Path
from xml.parsers.expat import errors

import httpx
from openai import OpenAI

from apps.agent.validation import validate_decision

TEXT_LLM_BASE_URL = os.getenv("TEXT_LLM_BASE_URL", "http://localhost:8000/v1")
TEXT_LLM_MODEL = os.getenv("TEXT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "local-dev-key")

EMR_BASE_URL = "http://localhost:9000"

SYSTEM_PROMPT = """You are a local OR logistics agent for a synthetic demo.

You coordinate simulated OR setup, supply, porter, specimen, and review workflows.
You must not diagnose, prescribe, select treatment, or clear a real clinical case.

Use only operational actions.
For uncertainty, create a human_review task.
For missing setup items, create a missing_supply task.
Never call an actuator unless the event is high confidence and operational-only.

Return only valid JSON with this shape:
{
  "decision_summary": "...",
  "tool_calls": [
    {
      "name": "...",
      "arguments": {}
    }
  ],
  "requires_human_review": true_or_false
}
"""


def load_event(path: str) -> dict:
    return json.loads(Path(path).read_text())


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
        "pc_gpu_vlm": {"available": True, "estimated_latency_seconds": 3}
    }


def ask_agent(event: dict, case: dict, resources: dict) -> dict:
    client = OpenAI(base_url=TEXT_LLM_BASE_URL, api_key=OPENAI_API_KEY)

    payload = {
        "event": event,
        "synthetic_pathway": case,
        "resources": resources,
        "allowed_tools": [
            "create_synthetic_or_task",
            "request_spd_resupply",
            "set_or_prep_light"
        ]
    }

    r = client.chat.completions.create(
        model=TEXT_LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)}
        ],
        temperature=0,
        max_tokens=700
    )

    text = r.choices[0].message.content.strip()
    return json.loads(text)


def main():
    import argparse
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