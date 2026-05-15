from mcp.server.fastmcp import FastMCP

from apps.vlm.ask_vlm import ask_vlm

mcp = FastMCP("vlm")


@mcp.tool()
def inspect_scene(image_path: str, question: str) -> str:
    """
    Inspect an OR scene image using the Azure VLM (gpt-4o).

    Use this to visually verify instruments, confirm item placement,
    or answer questions about what is visible on the table.

    Args:
        image_path: Path to the image file to inspect.
        question: The question to ask about the image.
    """
    return ask_vlm(image_path, question)
