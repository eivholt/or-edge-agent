#!/usr/bin/env python3
"""Benchmark detector-centered VLM inspection across the five OR scenarios."""

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
SCENARIOS_DIR = ROOT / "scenarios"
DEFAULT_OUTPUT = ROOT / "tests" / "vlm_centroid_256_results.json"
SCENARIOS = {
    "all_present": False,
    "instrument_out_of_zone": True,
    "missing_scissors": False,
    "missing_something": False,
    "sterile_zone_ambiguity": True,
}
GUIDED_JSON = {
    "type": "object",
    "properties": {"answer": {"type": "boolean"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("VLM_BASE_URL", "http://127.0.0.1:8082/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLM_MODEL", "ministral3-3b-vl-q4"),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VLM_API_KEY", "local-dev-key"),
    )
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--image-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard an existing result file instead of resuming it.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(path)


def load_scenario(name: str) -> tuple[Path, dict[str, Any]]:
    scenario = json.loads((SCENARIOS_DIR / f"{name}.json").read_text())
    image_path = DATA_DIR / scenario["image_path"]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return image_path, scenario


def detector_segments(image_path: Path, radius: int) -> list[dict[str, Any]]:
    from apps.agent.run_fixture import _build_vlm_segments
    from apps.detector.inference import detect

    detection_result = detect(image_path)
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
    segments = _build_vlm_segments(image_path, detections, radius)
    if not segments:
        raise RuntimeError(f"Detector found no segments in {image_path}")
    return segments


def segment_question(label: str) -> str:
    return (
        f"Focus on the detected {label} centered in this image. Is the surface "
        "directly underneath it bare gray or silver metal rather than green or "
        "turquoise cloth? The gray instrument itself does not count. Return true "
        "for bare metal and false for green cloth."
    )


def inspect_segment(
    client: httpx.Client,
    model: str,
    segment: dict[str, Any],
) -> tuple[bool, float]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": segment["data_url"]},
                    },
                    {"type": "text", "text": segment_question(segment["label"])},
                ],
            }
        ],
        "max_tokens": 16,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "vlm_response",
                "schema": GUIDED_JSON,
                "strict": True,
            },
        },
    }
    started = time.perf_counter()
    response = client.post("chat/completions", json=payload)
    response.raise_for_status()
    duration_ms = (time.perf_counter() - started) * 1000
    raw = response.json()["choices"][0]["message"]["content"]
    verdict = json.loads(raw).get("answer")
    if not isinstance(verdict, bool):
        raise ValueError(f"VLM response did not contain a boolean answer: {raw}")
    return verdict, duration_ms


def scenario_summary(result: dict[str, Any]) -> None:
    segments = result["segments"]
    if len(segments) != result["segments_total"]:
        return
    verdict = any(segment["verdict"] for segment in segments)
    result.update(
        {
            "verdict": verdict,
            "correct": verdict == result["ground_truth"],
            "duration_ms": sum(segment["duration_ms"] for segment in segments),
            "completed_at": utc_now(),
        }
    )


def initial_results(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "started_at": utc_now(),
        "completed_at": None,
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "radius": args.radius,
            "image_size": 224,
            "image_tokens": args.image_tokens,
            "max_output_tokens": 16,
        },
        "scenarios": {},
        "summary": None,
    }


def load_results(args: argparse.Namespace) -> dict[str, Any]:
    expected = initial_results(args)
    if args.fresh or not args.output.exists():
        return expected
    existing = json.loads(args.output.read_text())
    if existing.get("config") != expected["config"]:
        raise ValueError(
            f"Cannot resume {args.output}: benchmark configuration differs. "
            "Use --fresh or a different --output path."
        )
    return existing


def run(args: argparse.Namespace) -> dict[str, Any]:
    results = load_results(args)
    headers = {"Authorization": f"Bearer {args.api_key}"}
    with httpx.Client(
        base_url=f"{args.base_url.rstrip('/')}/",
        headers=headers,
        timeout=args.timeout,
    ) as client:
        models = client.get("models")
        models.raise_for_status()

        for scenario_name, ground_truth in SCENARIOS.items():
            image_path, _ = load_scenario(scenario_name)
            segments = detector_segments(image_path, args.radius)
            result = results["scenarios"].setdefault(
                scenario_name,
                {
                    "image": str(image_path.relative_to(ROOT)),
                    "ground_truth": ground_truth,
                    "segments_total": len(segments),
                    "segments": [],
                },
            )
            if result["segments_total"] != len(segments):
                raise ValueError(
                    f"Cannot resume {scenario_name}: detector count changed from "
                    f"{result['segments_total']} to {len(segments)}"
                )

            completed = len(result["segments"])
            print(
                f"{scenario_name}: {len(segments)} segments, resuming at "
                f"{completed + 1}",
                flush=True,
            )
            for index, segment in enumerate(segments, start=1):
                if index <= completed:
                    continue
                verdict, duration_ms = inspect_segment(client, args.model, segment)
                result["segments"].append(
                    {
                        "index": index,
                        "label": segment["label"],
                        "crop_box": segment["crop_box"],
                        "verdict": verdict,
                        "duration_ms": duration_ms,
                    }
                )
                write_results(args.output, results)
                print(
                    f"  {index}/{len(segments)} {segment['label']}: "
                    f"{str(verdict).lower()} in {duration_ms / 1000:.1f}s",
                    flush=True,
                )

            scenario_summary(result)
            write_results(args.output, results)
            print(
                f"  verdict={str(result['verdict']).lower()} "
                f"correct={result['correct']} "
                f"time={result['duration_ms'] / 1000:.1f}s",
                flush=True,
            )

    scenario_results = list(results["scenarios"].values())
    correct = sum(result["correct"] for result in scenario_results)
    total_duration_ms = sum(result["duration_ms"] for result in scenario_results)
    results["summary"] = {
        "correct": correct,
        "total": len(scenario_results),
        "duration_ms": total_duration_ms,
        "segments": sum(result["segments_total"] for result in scenario_results),
    }
    results["completed_at"] = utc_now()
    write_results(args.output, results)
    return results


def main() -> int:
    args = parse_args()
    results = run(args)
    print(json.dumps(results["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())