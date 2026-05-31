#!/usr/bin/env python3
"""VLM model comparison benchmark.

Runs every VLM scenario against multiple Azure OpenAI deployments
and compares answers side-by-side.

Usage:
    python -m tests.benchmark_vlm
    python -m tests.benchmark_vlm --models gpt-4o gpt-4.1 o4-mini
"""
import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SCENARIOS_DIR = ROOT / "scenarios"

AZURE_ENDPOINT = os.getenv("AZURE_VLM_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_VLM_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_VLM_API_VERSION", "2024-12-01-preview")

AZURE_ANTHROPIC_ENDPOINT = "https://synthetic-patient-resource.services.ai.azure.com/anthropic/"

LOCAL_VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8081/v1")
LOCAL_VLM_MODEL = os.getenv("VLM_MODEL", "mistralai/Ministral-3-3B-Instruct-2512-BF16")
LOCAL_VLM_API_KEY = os.getenv("OPENAI_API_KEY", "local-dev-key")
LOCAL_VLM_ALIAS = "ministral-3b-local"

# Ground truth for scored scenarios (YES or NO)
# The question framing determines the expected answer:
#   "Are any X outside the drape?" → YES means instruments ARE outside
GROUND_TRUTH = {
    "all_present": "NO",                   # all instruments are ON the drape
    "instrument_out_of_zone": "YES",      # instruments are outside the drape
    "missing_scissors": "NO",             # all instruments are ON the drape
    "missing_something": "NO",             # all instruments are ON the drape
    "ppe_compliance_check": "NO",          # person is NOT wearing a mask
    "procedure_changed": "NO",            # all instruments are ON the drape
    "room_turnover_check": "NO",           # room is NOT clear
    "specimen_handoff": "YES",             # specimen container IS visible
    "specimen_ready_check": "YES",         # specimen container IS visible
    "sterile_zone_ambiguity": "YES",       # instruments ARE outside the drape
}

SYSTEM_PROMPT = (
    "You are a surgical-scene analyst. Describe exactly what you see in the "
    "image, then answer the user's question. Respond ONLY with JSON:\n"
    '{"description": "one or two sentences", "answer": true/false}\n'
    '"answer" is true when YES, false when NO. '
    "Base your answer strictly on what is visible — do not guess or assume."
)

# Sterile-zone question used for all instrument-tray scenarios
_STERILE_ZONE_Q = (
    "Look at the green drape. Is even ONE of these outside the drape, "
    "on the bare table: scalpel, scissors, sponge, tweezers?"
)

# Default questions for event types that don't have vlm_question
DEFAULT_QUESTIONS = {
    "all_present": _STERILE_ZONE_Q,
    "instrument_out_of_zone": _STERILE_ZONE_Q,
    "missing_scissors": _STERILE_ZONE_Q,
    "missing_something": _STERILE_ZONE_Q,
    "or_setup_state_change": _STERILE_ZONE_Q,
    "procedure_changed": _STERILE_ZONE_Q,
    "sterile_zone_ambiguity": _STERILE_ZONE_Q,
    "visually_ready_but_pathway_changed": _STERILE_ZONE_Q,
    "specimen_ready_check": (
        "Is a specimen container visible near the mayo stand?"
    ),
    "specimen_container_seen": (
        "Is a specimen container visible on the back table?"
    ),
}


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def load_vlm_scenarios() -> list[dict]:
    """Load all scenarios that involve VLM inspection."""
    vlm_event_types = {
        "instrument_out_of_zone",
        "or_setup_state_change",
        "specimen_ready_check",
        "room_turnover_check",
        "ppe_compliance_check",
        "sterile_zone_ambiguity",
        "specimen_container_seen",
        "visually_ready_but_pathway_changed",
    }
    scenarios = []
    for f in sorted(SCENARIOS_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        event_type = data.get("event_type", "")
        has_vlm_q = "vlm_question" in data
        low_conf = data.get("confidence", 1.0) < 0.80
        if has_vlm_q or event_type in vlm_event_types or low_conf:
            # Resolve question
            question = data.get("vlm_question") or DEFAULT_QUESTIONS.get(f.stem) or DEFAULT_QUESTIONS.get(event_type, "Describe what you see in this image.")
            # Resolve image
            img_rel = data.get("image_path", "")
            img_path = DATA_DIR / img_rel
            if not img_path.is_file():
                for ext in (".png", ".jpg", ".jpeg", ".webp"):
                    candidate = img_path.with_suffix(ext)
                    if candidate.is_file():
                        img_path = candidate
                        break
            scenarios.append({
                "name": f.stem,
                "event_type": event_type,
                "confidence": data.get("confidence"),
                "question": question,
                "image_path": img_path,
                "has_image": img_path.is_file(),
            })
    return scenarios


def _extract_verdict(answer: str) -> str | None:
    """Pull a YES / NO verdict from the model answer (JSON or free-text)."""
    if not answer or answer.startswith("ERROR"):
        return None
    # Try JSON first
    try:
        obj = json.loads(answer.strip().strip("`").removeprefix("json").strip())
        v = obj.get("answer")
        if isinstance(v, bool):
            return "YES" if v else "NO"
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: free-text
    first = answer.strip().lstrip("*").strip().upper()
    if first.startswith("YES"):
        return "YES"
    if first.startswith("NO"):
        return "NO"
    if "NOT CLEAR" in first[:40] or "NOT READY" in first[:40]:
        return "NO"
    return None


def _is_claude(deployment: str) -> bool:
    return "claude" in deployment.lower()


def _is_local(deployment: str) -> bool:
    return deployment == LOCAL_VLM_ALIAS


def ask_claude(deployment: str, image_path: Path, question: str) -> tuple[str, float]:
    """Call Azure-hosted Anthropic model. Returns (answer, latency_ms)."""
    import anthropic
    client = anthropic.Anthropic(
        base_url=AZURE_ANTHROPIC_ENDPOINT,
        api_key=AZURE_API_KEY,
    )
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    t0 = time.perf_counter()
    try:
        r = client.messages.create(
            model=deployment,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": question},
                ],
            }],
        )
        answer = r.content[0].text.strip()
    except Exception as e:
        answer = f"ERROR: {e}"
    latency = (time.perf_counter() - t0) * 1000
    return answer, latency


def ask_local(image_path: Path, question: str) -> tuple[str, float]:
    """Call local vLLM (Ministral 3B). Returns (answer, latency_ms)."""
    from openai import OpenAI
    client = OpenAI(base_url=LOCAL_VLM_BASE_URL, api_key=LOCAL_VLM_API_KEY)
    t0 = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model=LOCAL_VLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ]},
            ],
            max_tokens=300,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "vlm_response",
                    "schema": {"type": "object", "properties": {"description": {"type": "string"}, "answer": {"type": "boolean"}}, "required": ["description", "answer"], "additionalProperties": False},
                    "strict": True,
                },
            },
        )
        answer = r.choices[0].message.content.strip()
    except Exception as e:
        answer = f"ERROR: {e}"
    latency = (time.perf_counter() - t0) * 1000
    return answer, latency


def ask_model(deployment: str, image_path: Path, question: str) -> tuple[str, float]:
    """Route to the right backend. Returns (answer, latency_ms)."""
    if _is_local(deployment):
        return ask_local(image_path, question)
    if _is_claude(deployment):
        return ask_claude(deployment, image_path, question)
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )
    t0 = time.perf_counter()
    try:
        # gpt-5.5 and newer models require max_completion_tokens and don't support temperature
        is_new_model = any(tag in deployment for tag in ("5.5", "o4", "o3"))
        token_param = {"max_completion_tokens": 300} if is_new_model else {"max_tokens": 300}
        temp_param = {} if is_new_model else {"temperature": 0}
        r = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ]},
            ],
            response_format={"type": "json_object"},
            **temp_param,
            **token_param,
        )
        answer = r.choices[0].message.content.strip()
    except Exception as e:
        answer = f"ERROR: {e}"
    latency = (time.perf_counter() - t0) * 1000
    return answer, latency


def run_benchmark(models: list[str]):
    scenarios = load_vlm_scenarios()
    print(f"\n{'='*80}")
    print(f"  VLM Model Benchmark — {len(scenarios)} scenarios × {len(models)} models")
    endpoints = [AZURE_ENDPOINT]
    if any(_is_claude(m) for m in models):
        endpoints.append(AZURE_ANTHROPIC_ENDPOINT)
    if any(_is_local(m) for m in models):
        endpoints.append(f"{LOCAL_VLM_BASE_URL}  ({LOCAL_VLM_MODEL})")
    for ep in endpoints:
        print(f"  Endpoint: {ep}")
    print(f"{'='*80}\n")

    results = []

    for i, sc in enumerate(scenarios, 1):
        print(f"─── [{i}/{len(scenarios)}] {sc['name']} ───")
        print(f"  Event type : {sc['event_type']}")
        print(f"  Confidence : {sc['confidence']}")
        print(f"  Image      : {sc['image_path'].name} ({'found' if sc['has_image'] else 'MISSING'})")
        print(f"  Question   : {sc['question'][:100]}{'…' if len(sc['question']) > 100 else ''}")
        print()

        if not sc["has_image"]:
            print("  ⚠ Skipping — image not found\n")
            continue

        row = {"scenario": sc["name"], "event_type": sc["event_type"], "question": sc["question"]}

        for model in models:
            print(f"  [{model}] calling…", end=" ", flush=True)
            answer, latency = ask_model(model, sc["image_path"], sc["question"])
            print(f"{latency:.0f}ms")

            # Truncate for display; pretty-print JSON if possible
            try:
                parsed = json.loads(answer.strip())
                verdict_flag = "YES" if parsed.get("answer") else "NO"
                desc = parsed.get("description", "")
                short = f"{verdict_flag} — {desc[:160]}{'…' if len(desc) > 160 else ''}"
            except (json.JSONDecodeError, AttributeError):
                short = answer[:200] + ("…" if len(answer) > 200 else "")
            print(f"    → {short}\n")

            row[f"{model}_answer"] = answer
            row[f"{model}_ms"] = round(latency)

        results.append(row)

    # Summary table
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}\n")

    col_w = 16
    header = f"{'Scenario':<30} | {'GT':>3}"
    for m in models:
        header += f" | {'(' + m + ')':<{col_w}} | {'ms':>5}"
    print(header)
    print("─" * len(header))

    # Track per-model accuracy
    model_correct = {m: 0 for m in models}
    model_scored = {m: 0 for m in models}

    for r in results:
        scenario = r["scenario"]
        gt = GROUND_TRUTH.get(scenario)
        gt_label = gt or "?"
        line = f"{scenario:<30} | {gt_label:>3}"
        for m in models:
            ans = r.get(f"{m}_answer", "N/A")
            verdict = _extract_verdict(ans)
            # Mark correct/wrong
            if gt and verdict:
                model_scored[m] += 1
                correct = verdict == gt
                if correct:
                    model_correct[m] += 1
                mark = "✓" if correct else "✗"
            else:
                mark = "–"
            display = (verdict or "???") + " " + mark
            ms = r.get(f"{m}_ms", 0)
            line += f" | {display:<{col_w}} | {ms:>5}"
        print(line)

    # Accuracy footer
    print("─" * len(header))
    acc_line = f"{'ACCURACY':<30} |    "
    for m in models:
        scored = model_scored[m]
        if scored:
            pct = model_correct[m] / scored * 100
            acc_line += f" | {model_correct[m]}/{scored} ({pct:.0f}%){'':>{col_w - 11}}"
        else:
            acc_line += f" | {'N/A':<{col_w}}"
        acc_line += f" |      "
    print(acc_line)

    avg_line = f"{'AVG LATENCY':<30} |    "
    for m in models:
        all_ms = [r.get(f"{m}_ms", 0) for r in results if r.get(f"{m}_ms")]
        avg = sum(all_ms) / len(all_ms) if all_ms else 0
        avg_line += f" | {avg:.0f}ms{'':>{col_w - 5 - len(str(int(avg)))}} |      "
    print(avg_line)

    # Save full results as JSON (include ground truth + verdicts)
    for r in results:
        scenario = r["scenario"]
        r["ground_truth"] = GROUND_TRUTH.get(scenario)
        for m in models:
            r[f"{m}_verdict"] = _extract_verdict(r.get(f"{m}_answer", ""))
            r[f"{m}_correct"] = (
                r[f"{m}_verdict"] == r["ground_truth"]
                if r["ground_truth"] and r[f"{m}_verdict"]
                else None
            )
    out_path = ROOT / "tests" / "vlm_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Full results saved to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM model comparison benchmark")
    parser.add_argument(
        "--models", nargs="+", default=["gpt-4o", "gpt-5.5", "claude-opus-4-7", LOCAL_VLM_ALIAS],
        help=f"Deployment names to compare. Use '{LOCAL_VLM_ALIAS}' for local vLLM.",
    )
    args = parser.parse_args()
    run_benchmark(args.models)
