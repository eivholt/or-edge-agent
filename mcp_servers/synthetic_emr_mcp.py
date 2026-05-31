import logfire
import httpx
from mcp.server.fastmcp import FastMCP

logfire.configure(service_name="mcp-synthetic-emr")
logfire.instrument_httpx()

mcp = FastMCP("synthetic-emr")

EMR_BASE_URL = "http://localhost:9000"


@mcp.tool()
async def get_case(case_id: str) -> dict:
    """
    Return the synthetic surgical pathway for a case.

    Use this when a physical OR scene event needs to be interpreted
    in the context of the scheduled synthetic procedure, open workflow
    items, required instruments, porter status, lab status, or specimen flow.

    Synthetic operational context only. Do not use this to diagnose,
    prescribe, select treatment, or clear a real clinical workflow.
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{EMR_BASE_URL}/cases/{case_id}")
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def get_case_setup_requirements(case_id: str) -> dict:
    """
    Return the physical OR setup requirements for the synthetic case.

    Use this before deciding whether a visible instrument setup is complete,
    incomplete, wrong-case, or needs human review.
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{EMR_BASE_URL}/cases/{case_id}/setup-requirements")
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def create_task(
    case_id: str,
    task_type: str,
    priority: str,
    summary: str,
    reason: str
) -> dict:
    """
    Create a synthetic OR workflow task.

    Use for logistics, verification, transport, setup, supply, or review tasks.
    Do not use this tool to diagnose, prescribe, select treatment, or
    independently clear a real clinical workflow.
    """
    payload = {
        "case_id": case_id,
        "task_type": task_type,
        "priority": priority,
        "summary": summary,
        "reason": reason
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{EMR_BASE_URL}/tasks", json=payload)
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    mcp.run()