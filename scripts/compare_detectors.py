"""Compare detection capabilities: Edge Impulse vs Ministral vs GPT-4o.

Runs each model on the same set of sample images and prints a comparison table.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMAGES = [
    ROOT / "data/frames/ei_rgb_9996.png",   # lots of instruments
    ROOT / "data/frames/ei_rgb_6997.png",   # fewer instruments
    ROOT / "data/frames/ei_rgb_5999.png",   # many instruments
]
CLASSES = ["scalpel", "scissors", "sponge", "tweezers"]

VLM_PROMPT = (
    "Count the surgical instruments in this image. "
    "Report ONLY a JSON object with these exact keys: scalpel, scissors, sponge, tweezers. "
    "Each value is the integer count visible. Example: {\"scalpel\": 2, \"scissors\": 1, \"sponge\": 3, \"tweezers\": 2}. "
    "Output ONLY the JSON, nothing else."
)


def run_edge_impulse(image_path: Path) -> dict[str, int]:
    from apps.detector.inference import detect
    result = detect(image_path)
    return dict(Counter(d.label for d in result.detections))


def run_ministral(image_path: Path) -> dict[str, int] | str:
    """Try Ministral-3B via vLLM — text-only model, image input will likely fail."""
    import base64
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8081/v1", api_key="dummy")
    mime = "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    data_url = f"data:{mime};base64,{b64}"

    try:
        r = client.chat.completions.create(
            model="mistralai/Ministral-3-3B-Instruct-2512-BF16",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=0,
            max_tokens=200,
        )
        text = r.choices[0].message.content.strip()
        return _parse_counts(text)
    except Exception as e:
        return f"ERROR: {e}"


def run_gpt4o(image_path: Path) -> dict[str, int] | str:
    from apps.vlm.ask_vlm import ask_vlm
    try:
        text = ask_vlm(str(image_path), VLM_PROMPT)
        return _parse_counts(text)
    except Exception as e:
        return f"ERROR: {e}"


def _parse_counts(text: str) -> dict[str, int] | str:
    """Extract JSON counts from model response."""
    import re
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    # Find first { ... } block
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            return {k: int(data.get(k, 0)) for k in CLASSES}
        except (json.JSONDecodeError, ValueError):
            pass
    return f"PARSE_FAIL: {text[:80]}"


def fmt_counts(result) -> str:
    if isinstance(result, str):
        return result[:40]
    return " | ".join(f"{result.get(c, 0)}" for c in CLASSES)


def main():
    rows = []

    for img in IMAGES:
        name = img.stem
        print(f"\n{'='*60}")
        print(f"Image: {name}")
        print(f"{'='*60}")

        # Edge Impulse
        print("  Running Edge Impulse...", end=" ", flush=True)
        ei = run_edge_impulse(img)
        print(f"done: {ei}")

        # Ministral
        print("  Running Ministral-3B...", end=" ", flush=True)
        mi = run_ministral(img)
        print(f"done: {mi}")

        # GPT-4o
        print("  Running GPT-4o...", end=" ", flush=True)
        g4 = run_gpt4o(img)
        print(f"done: {g4}")

        rows.append((name, ei, mi, g4))

    # Print comparison table
    print(f"\n\n{'='*90}")
    print("COMPARISON TABLE")
    print(f"{'='*90}")
    header = f"{'Image':<18} {'Model':<14} {'scalpel':>7} {'scissors':>8} {'sponge':>6} {'tweezers':>8}"
    print(header)
    print("-" * len(header))

    for name, ei, mi, g4 in rows:
        for label, result in [("EI model", ei), ("Ministral-3B", mi), ("GPT-4o", g4)]:
            if isinstance(result, str):
                print(f"{name:<18} {label:<14} {result}")
            else:
                print(
                    f"{name:<18} {label:<14} "
                    f"{result.get('scalpel', 0):>7} "
                    f"{result.get('scissors', 0):>8} "
                    f"{result.get('sponge', 0):>6} "
                    f"{result.get('tweezers', 0):>8}"
                )
        print()


if __name__ == "__main__":
    main()
