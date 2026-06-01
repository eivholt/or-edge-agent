import logfire
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

logfire.configure(service_name="synthetic-emr-api")

app = FastAPI(title="Synthetic OR EMR API")
logfire.instrument_fastapi(app)

CASES = {
    "CASE-1042": {
        "case_id": "CASE-1042",
        "patient_id": "SYN-PAT-8842",
        "procedure": "Laparoscopic Biopsy",
        "priority": "normal",
        "required_items": {
            "scalpel": 1,
            "scissors": 3,
            "sponge": 6,
            "tweezers": 2
        },
    },
    "CASE-1045": {
        "case_id": "CASE-1045",
        "patient_id": "SYN-PAT-8845",
        "procedure": "Laparoscopic Cholecystectomy",
        "priority": "normal",
        "required_items": {
            "tweezers": 2,
            "scalpel": 2,
            "sponge": 3,
            "scissors": 1
        },
    },
    "CASE-1044": {
        "case_id": "CASE-1044",
        "patient_id": "SYN-PAT-8844",
        "procedure": "Laparoscopic Biopsy",
        "priority": "normal",
        "required_items": {
            "scalpel": 1,
            "scissors": 2,
            "sponge": 4,
            "tweezers": 1
        },
    },
    "CASE-1043": {
        "case_id": "CASE-1043",
        "patient_id": "SYN-PAT-8843",
        "procedure": "Laparoscopic Cholecystectomy",
        "priority": "normal",
        "required_items": {
            "scalpel": 2,
            "scissors": 3,
            "sponge": 3,
            "tweezers": 2
        },
    },
    "CASE-2001": {
        "case_id": "CASE-2001",
        "patient_id": "SYN-PAT-2001",
        "procedure": "Open Conversion Preparedness Pathway",
        "priority": "high",
        "required_items": {
            "scalpel": 2,
            "scissors": 2,
            "sponge": 4,
            "tweezers": 2
        },
    },

    # ── Integration-test cases ───────────────────────────────────────
    "CASE-INT-1": {
        "case_id": "CASE-INT-1",
        "patient_id": "SYN-PAT-INT1",
        "procedure": "Laparoscopic Cholecystectomy",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
    },
    "CASE-INT-2": {
        "case_id": "CASE-INT-2",
        "patient_id": "SYN-PAT-INT2",
        "procedure": "Minor Excision",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 1, "sponge": 2},
    },
    "CASE-INT-3": {
        "case_id": "CASE-INT-3",
        "patient_id": "SYN-PAT-INT3",
        "procedure": "Wound Closure",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 1},
    },
    "CASE-INT-4": {
        "case_id": "CASE-INT-4",
        "patient_id": "SYN-PAT-INT4",
        "procedure": "Laparoscopic Biopsy",
        "priority": "high",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
    },
    "CASE-INT-6": {
        "case_id": "CASE-INT-6",
        "patient_id": "SYN-PAT-INT6",
        "procedure": "Biopsy",
        "priority": "normal",
        "required_items": {"scissors": 1, "sponge": 2},
    },
    "CASE-INT-7": {
        "case_id": "CASE-INT-7",
        "patient_id": "SYN-PAT-INT7",
        "procedure": "Appendectomy",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2, "tweezers": 2},
    },
    "CASE-INT-10": {
        "case_id": "CASE-INT-10",
        "patient_id": "SYN-PAT-INT10",
        "procedure": "Laparoscopic Cholecystectomy (Changed from Appendectomy)",
        "priority": "high",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
    },
    "CASE-INT-11": {
        "case_id": "CASE-INT-11",
        "patient_id": "SYN-PAT-INT11",
        "procedure": "Minor Excision",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 1},
    },
    "CASE-INT-12": {
        "case_id": "CASE-INT-12",
        "patient_id": "SYN-PAT-INT12",
        "procedure": "Laparoscopic Biopsy",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
    },
    # ── LLM reconcile test cases ─────────────────────────────────────
    "CASE-A": {
        "case_id": "CASE-A",
        "patient_id": "SYN-PAT-A",
        "procedure": "Minor Excision",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 1, "sponge": 2},
    },
    "CASE-B": {
        "case_id": "CASE-B",
        "patient_id": "SYN-PAT-B",
        "procedure": "Laparoscopic Biopsy",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
    },
    # ── VLM test case ────────────────────────────────────────────────
    "CASE-VLM": {
        "case_id": "CASE-VLM",
        "patient_id": "SYN-PAT-VLM",
        "procedure": "Minor Excision",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 2},
    },
    # ── VLM scene-understanding cases ────────────────────────────────
    "CASE-4001": {
        "case_id": "CASE-4001",
        "patient_id": "SYN-PAT-4001",
        "procedure": "Laparoscopic Cholecystectomy (Turnover Pending)",
        "priority": "normal",
        "required_items": {
            "scalpel": 2,
            "scissors": 2,
            "sponge": 4,
            "tweezers": 2,
        },
    },
    "CASE-5001": {
        "case_id": "CASE-5001",
        "patient_id": "SYN-PAT-5001",
        "procedure": "Laparoscopic Biopsy",
        "priority": "normal",
        "required_items": {
            "scalpel": 2,
            "scissors": 2,
            "sponge": 4,
            "tweezers": 2,
        },
    },
    # ── Benchmark stress-test cases ──────────────────────────────────
    "CASE-BENCH-1": {
        "case_id": "CASE-BENCH-1",
        "patient_id": "SYN-PAT-BENCH1",
        "procedure": "Minor Excision",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 1, "sponge": 2},
    },
    "CASE-BENCH-2": {
        "case_id": "CASE-BENCH-2",
        "patient_id": "SYN-PAT-BENCH2",
        "procedure": "Laparoscopic Cholecystectomy",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
    },
    "CASE-BENCH-4": {
        "case_id": "CASE-BENCH-4",
        "patient_id": "SYN-PAT-BENCH4",
        "procedure": "Open Conversion (Changed from Laparoscopic Appendectomy)",
        "priority": "high",
        "required_items": {"scalpel": 3, "scissors": 2, "sponge": 6, "tweezers": 2},
    },
    "CASE-BENCH-5": {
        "case_id": "CASE-BENCH-5",
        "patient_id": "SYN-PAT-BENCH5",
        "procedure": "Laparoscopic Biopsy",
        "priority": "high",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
    },
}

TASKS = []


class TaskCreate(BaseModel):
    case_id: str
    task_type: Literal[
        "missing_supply",
        "human_review",
        "porter_hold",
        "porter_release",
        "wrong_case_cart",
        "procedure_change_review"
    ]
    priority: Literal["low", "normal", "high"]
    summary: str
    reason: str


@app.get("/cases/{case_id}")
def get_case(case_id: str):
    case = CASES.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Unknown case_id {case_id}")
    return case


@app.get("/cases/{case_id}/setup-requirements")
def get_setup_requirements(case_id: str):
    case = CASES.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Unknown case_id {case_id}")
    return {
        "case_id": case_id,
        "procedure": case["procedure"],
        "priority": case["priority"],
        "required_items": case["required_items"],
    }


@app.post("/tasks")
def create_task(task: TaskCreate):
    record = task.model_dump()
    record["task_id"] = f"TASK-{len(TASKS) + 1:04d}"
    record["status"] = "created"
    TASKS.append(record)
    return record


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASKS}