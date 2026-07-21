#!/usr/bin/env python3
"""Expose GenieAPIService with Ministral's native tools as an OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return json.dumps(content, ensure_ascii=False)


def parse_arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def render_prompt(payload: dict[str, Any]) -> str:
    messages = payload.get("messages", [])
    tools = payload.get("tools", [])
    system: list[str] = []
    conversation: list[dict[str, Any]] = []
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system":
            system.append(content_text(message.get("content", "")).strip())
        else:
            conversation.append(message)

    parts = ["<s>"]
    system_text = "\n".join(item for item in system if item)
    if system_text:
        parts.extend(["[SYSTEM_PROMPT]", system_text, "[/SYSTEM_PROMPT]"])
    if isinstance(tools, list) and tools:
        parts.extend(
            [
                "[AVAILABLE_TOOLS]",
                json.dumps(tools, ensure_ascii=False, separators=(",", ":")),
                "[/AVAILABLE_TOOLS]",
            ]
        )

    tool_names: dict[str, str] = {}
    for message in conversation:
        role = str(message.get("role", "user"))
        if role == "user":
            parts.extend(["[INST]", content_text(message.get("content", "")).strip(), "[/INST]"])
        elif role == "assistant":
            text = content_text(message.get("content", "")).strip()
            if text and text != "null":
                parts.append(text)
            calls = message.get("tool_calls", [])
            for call in calls if isinstance(calls, list) else []:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                if not isinstance(function, dict):
                    continue
                name = str(function.get("name", ""))
                call_id = str(call.get("id", ""))
                if call_id and name:
                    tool_names[call_id] = name
                arguments = parse_arguments(function.get("arguments", {})) or {}
                parts.extend(
                    [
                        "[TOOL_CALLS]",
                        name,
                        "[ARGS]",
                        json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                    ]
                )
            parts.append("</s>")
        elif role == "tool":
            value = message.get("content", "")
            try:
                value = json.dumps(json.loads(value), ensure_ascii=False, separators=(",", ":"))
            except (TypeError, json.JSONDecodeError):
                value = content_text(value)
            name = message.get("name") or tool_names.get(str(message.get("tool_call_id", "")))
            if name:
                value = f"{name}: {value}"
            parts.extend(["[TOOL_RESULTS]", value, "[/TOOL_RESULTS]"])
    return "".join(parts)


def parse_response(raw: str) -> tuple[dict[str, Any], str]:
    pattern = re.compile(
        r"\[TOOL_CALLS\]\s*([A-Za-z_][\w.-]*)\s*\[ARGS\]\s*(\{[\s\S]*?})(?=\s*(?:\[TOOL_CALLS\]|</s>|\[END\]|$))",
        flags=re.IGNORECASE,
    )
    calls = []
    for index, match in enumerate(pattern.finditer(raw), start=1):
        arguments = parse_arguments(match.group(2))
        if arguments is not None:
            calls.append((f"call_{index}", match.group(1), arguments))

    if not calls:
        decoder = json.JSONDecoder()
        for index in range(len(raw)):
            if raw[index] != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            if not isinstance(candidate, dict):
                continue
            name = candidate.get("name") or candidate.get("function")
            arguments = parse_arguments(
                candidate.get("arguments", candidate.get("parameters", {}))
            )
            if isinstance(name, str) and arguments is not None:
                calls.append(("call_1", name, arguments))
                break

    if calls:
        return (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, ensure_ascii=True),
                        },
                    }
                    for call_id, name, arguments in calls
                ],
            },
            "tool_calls",
        )
    final = re.sub(
        r"\[TOOL_CALLS\][\s\S]*?(?=(?:\[TOOL_CALLS\]|</s>|\[END\]|$))",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()
    return {"role": "assistant", "content": final or raw.strip()}, "stop"


class AdapterServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], args: argparse.Namespace) -> None:
        super().__init__(address, Handler)
        self.args = args
        self.inference_lock = threading.Lock()
        self.counter = 0
        self.work_dir = args.work_dir.expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def post_upstream(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.args.upstream_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.args.timeout_s) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"C++ Genie returned HTTP {exc.code}: {detail}") from exc


class Handler(BaseHTTPRequestHandler):
    server: AdapterServer

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.args.verbose:
            super().log_message(format, *args)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"", "/health"}:
            self.send_json(200, {"status": "ok", "backend": "cpp-genie"})
        elif self.path.rstrip("/") == "/v1/models":
            self.send_json(200, {"object": "list", "data": [{"id": self.server.args.model_name}]})
        else:
            self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") not in {"/chat/completions", "/v1/chat/completions"}:
            self.send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode())
            if payload.get("stream"):
                raise ValueError("Streaming is not supported")
            prompt = render_prompt(payload)
            upstream_payload = {
                "model": self.server.args.upstream_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            for key in ("max_tokens", "temperature", "top_p", "seed"):
                if key in payload:
                    upstream_payload[key] = payload[key]
            started = time.monotonic()
            with self.server.inference_lock:
                upstream = self.server.post_upstream(upstream_payload)
            raw = upstream["choices"][0]["message"]["content"]
            message, finish_reason = parse_response(raw)
            self.server.counter += 1
            result = {
                "id": upstream.get("id", f"chatcmpl-cpp-{time.time_ns()}"),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.server.args.model_name,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": upstream.get("usage", {}),
            }
            record = {
                "elapsed_s": round(time.monotonic() - started, 3),
                "prompt": prompt,
                "raw_response": upstream,
                "adapted_response": result,
            }
            path = self.server.work_dir / f"request_{self.server.counter:04d}.json"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            self.send_json(200, result)
        except Exception as exc:
            self.send_json(502, {"error": {"type": type(exc).__name__, "message": str(exc)}})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--upstream-url", default="http://127.0.0.1:8911/v1")
    parser.add_argument("--upstream-model", default="ministral_q4_genie_export")
    parser.add_argument("--model-name", default="ministral3-3b-q4")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--work-dir", type=Path, default=Path.home() / "genie_adapter_logs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    server = AdapterServer((args.host, args.port), args)
    print(f"Genie adapter: http://{args.host}:{args.port}/v1 -> {args.upstream_url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
