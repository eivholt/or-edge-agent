import subprocess

import logfire
from mcp.server.fastmcp import FastMCP

logfire.configure(service_name="mcp-resource-broker")

mcp = FastMCP("resource-broker")


def try_nvidia_smi() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode != 0:
            return {"available": False, "reason": result.stderr.strip()}

        used, total, util = [x.strip() for x in result.stdout.splitlines()[0].split(",")]
        return {
            "available": True,
            "memory_used_mb": int(used),
            "memory_total_mb": int(total),
            "gpu_utilization_percent": int(util)
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


@mcp.tool()
def get_model_runtime_status() -> dict:
    """
    Return current model/runtime availability.

    Use this before deciding whether to call a local VLM, text LLM,
    remote model, or fallback service. Includes rough GPU memory state
    and model service endpoints for the synthetic demo.
    """
    gpu = try_nvidia_smi()

    return {
        "gpu": gpu,
        "local_text_llm": {
            "available": True,
            "endpoint": "http://localhost:8000/v1",
            "role": "agent reasoning"
        },
        "local_vlm": {
            "available": True,
            "endpoint": "http://localhost:8001/v1",
            "role": "visual ambiguity resolution"
        },
        "future_devkit_mode": {
            "detector": "local",
            "text_agent": "local or delegated",
            "vlm": "event-driven or delegated"
        }
    }


if __name__ == "__main__":
    mcp.run()