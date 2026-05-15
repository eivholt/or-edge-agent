import base64
import os
from pathlib import Path

import httpx
import logfire
from mcp.server.fastmcp import FastMCP

logfire.configure(service_name="mcp-vlm-inspector")
logfire.instrument_httpx()

mcp = FastMCP("vlm-inspector")

VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8081/v1")
# VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
# VLM_MODEL = os.getenv("VLM_MODEL", "mistralai/Ministral-3-3B-Instruct-2512")
VLM_MODEL = os.getenv("VLM_MODEL", "mistralai/Ministral-3-3B-Instruct-2512-BF16")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "local-dev-key")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@mcp.tool()
def inspect_scene_local(image_path: str, question: str) -> dict:
    """
    Send an image to the LOCAL VLM (Ministral 3B on-device) for visual
    inspection of an OR scene.  This is the preferred, low-latency option.

    image_path: path to an image file relative to the project data/ directory,
                or an absolute path.
    question:   what to ask the VLM about the image
                (e.g. "List all surgical instruments visible on the tray.").

    Returns the VLM's text response.
    Do not use this for clinical diagnosis — only for operational logistics
    such as verifying instrument presence, tray layout, or setup state.
    """
    resolved = Path(image_path)
    if not resolved.is_absolute():
        resolved = DATA_DIR / image_path

    if not resolved.is_file():
        return {"error": f"image not found: {resolved}"}

    suffix = resolved.suffix.lower().lstrip(".")
    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}
    mime_subtype = mime_map.get(suffix)
    if not mime_subtype:
        return {"error": f"unsupported image format: {suffix}"}

    encoded = base64.b64encode(resolved.read_bytes()).decode()
    data_url = f"data:image/{mime_subtype};base64,{encoded}"

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0,
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
    body = r.json()
    answer = body["choices"][0]["message"]["content"]

    return {"image": str(resolved), "question": question, "answer": answer}


if __name__ == "__main__":
    mcp.run()
