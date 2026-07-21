import logfire
from mcp.server.fastmcp import FastMCP

from apps.vlm.ask_vlm import ask_vlm

logfire.configure(service_name="mcp-vlm")

mcp = FastMCP("vlm")


@mcp.tool()
def inspect_scene_remote(image_path: str, question: str) -> str:
    """
    Inspect an OR scene image using the remote Azure OpenAI VLM.

    This is a cloud-hosted fallback — use only when the local VLM
    (inspect_scene_local) is unavailable or returns uncertain results.
    Higher quality but slower and incurs cloud costs.

    Args:
        image_path: Path to the image file to inspect.
        question: The question to ask about the image.
    """
    return ask_vlm(image_path, question)
