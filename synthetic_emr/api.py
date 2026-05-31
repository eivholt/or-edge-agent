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
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {
            "scalpel": 1,
            "scissors": 3,
            "sponge": 6,
            "tweezers": 2
        },
        "open_items": [
            "porter_not_released",
            "pathology_handoff_task_not_created"
        ],
        "porter_release_allowed": False
    },
    "CASE-1044": {
        "case_id": "CASE-1044",
        "patient_id": "SYN-PAT-8844",
        "procedure": "Laparoscopic Biopsy",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {
            "scalpel": 1,
            "scissors": 2,
            "sponge": 4,
            "tweezers": 1
        },
        "open_items": [],
        "porter_release_allowed": False
    },
    "CASE-1043": {
        "case_id": "CASE-1043",
        "patient_id": "SYN-PAT-8843",
        "procedure": "Laparoscopic Cholecystectomy",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {
            "scalpel": 2,
            "scissors": 3,
            "sponge": 4,
            "tweezers": 2
        },
        "open_items": [
            "porter_not_released"
        ],
        "porter_release_allowed": False
    },
    "CASE-2001": {
        "case_id": "CASE-2001",
        "patient_id": "SYN-PAT-2001",
        "procedure": "Open Conversion Preparedness Pathway",
        "phase": "procedure_changed_after_setup",
        "priority": "high",
        "required_items": {
            "scalpel": 2,
            "scissors": 2,
            "sponge": 4,
            "tweezers": 2
        },
        "open_items": [
            "updated_setup_not_confirmed",
            "porter_not_released"
        ],
        "porter_release_allowed": False
    },
    "CASE-3001": {
        "case_id": "CASE-3001",
        "patient_id": "SYN-PAT-3001",
        "procedure": "Biopsy",
        "phase": "case_closing_candidate",
        "priority": "normal",
        "required_items": {
            "scissors": 1,
            "sponge": 2
        },
        "open_items": [
            "pathology_handoff_task_not_created"
        ],
        "expected_specimen": True,
        "specimen_destination": "synthetic_pathology"
    },
    # ── Integration-test cases ───────────────────────────────────────
    "CASE-INT-1": {
        "case_id": "CASE-INT-1",
        "patient_id": "SYN-PAT-INT1",
        "procedure": "Laparoscopic Cholecystectomy",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
        "open_items": [],
        "porter_release_allowed": False,
    },
    "CASE-INT-2": {
        "case_id": "CASE-INT-2",
        "patient_id": "SYN-PAT-INT2",
        "procedure": "Minor Excision",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 1, "sponge": 2},
        "open_items": [],
        "porter_release_allowed": False,
    },
    "CASE-INT-3": {
        "case_id": "CASE-INT-3",
        "patient_id": "SYN-PAT-INT3",
        "procedure": "Wound Closure",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 1},
        "open_items": [],
        "porter_release_allowed": False,
    },
    "CASE-INT-4": {
        "case_id": "CASE-INT-4",
        "patient_id": "SYN-PAT-INT4",
        "procedure": "Laparoscopic Biopsy",
        "phase": "pre_op_setup",
        "priority": "high",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
        "open_items": [],
        "porter_release_allowed": False,
    },
    "CASE-INT-6": {
        "case_id": "CASE-INT-6",
        "patient_id": "SYN-PAT-INT6",
        "procedure": "Biopsy",
        "phase": "intra_op",
        "priority": "normal",
        "required_items": {"scissors": 1, "sponge": 2},
        "open_items": [],
        "porter_release_allowed": False,
    },
    "CASE-INT-7": {
        "case_id": "CASE-INT-7",
        "patient_id": "SYN-PAT-INT7",
        "procedure": "Appendectomy",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2, "tweezers": 2},
        "open_items": [],
        "porter_release_allowed": False,
    },
    "CASE-INT-10": {
        "case_id": "CASE-INT-10",
        "patient_id": "SYN-PAT-INT10",
        "procedure": "Laparoscopic Cholecystectomy (Changed from Appendectomy)",
        "phase": "pre_op_setup",
        "priority": "high",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
        "open_items": [],
        "porter_release_allowed": False,
    },
    "CASE-INT-11": {
        "case_id": "CASE-INT-11",
        "patient_id": "SYN-PAT-INT11",
        "procedure": "Minor Excision",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 1},
        "open_items": [],
        "porter_release_allowed": False,
    },
    "CASE-INT-12": {
        "case_id": "CASE-INT-12",
        "patient_id": "SYN-PAT-INT12",
        "procedure": "Laparoscopic Biopsy",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
        "open_items": [],
        "porter_release_allowed": False,
    },
    # ── LLM reconcile test cases ─────────────────────────────────────
    "CASE-A": {
        "case_id": "CASE-A",
        "patient_id": "SYN-PAT-A",
        "procedure": "Minor Excision",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 1, "sponge": 2},
        "open_items": [],
        "porter_release_allowed": False,
    },
    "CASE-B": {
        "case_id": "CASE-B",
        "patient_id": "SYN-PAT-B",
        "procedure": "Laparoscopic Biopsy",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
        "open_items": [],
        "porter_release_allowed": False,
    },
    # ── VLM test case ────────────────────────────────────────────────
    "CASE-VLM": {
        "case_id": "CASE-VLM",
        "patient_id": "SYN-PAT-VLM",
        "procedure": "Minor Excision",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 2},
        "open_items": [],
        "porter_release_allowed": False,
    },
    # ── VLM scene-understanding cases ────────────────────────────────
    "CASE-4001": {
        "case_id": "CASE-4001",
        "patient_id": "SYN-PAT-4001",
        "procedure": "Laparoscopic Cholecystectomy (Turnover Pending)",
        "phase": "room_turnover",
        "priority": "normal",
        "required_items": {
            "scalpel": 2,
            "scissors": 2,
            "sponge": 4,
            "tweezers": 2,
        },
        "open_items": ["previous_case_equipment_not_cleared"],
        "porter_release_allowed": False,
    },
    "CASE-5001": {
        "case_id": "CASE-5001",
        "patient_id": "SYN-PAT-5001",
        "procedure": "Laparoscopic Biopsy",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {
            "scalpel": 2,
            "scissors": 2,
            "sponge": 4,
            "tweezers": 2,
        },
        "open_items": ["ppe_compliance_not_verified"],
        "porter_release_allowed": False,
    },
    # ── Benchmark stress-test cases ──────────────────────────────────
    "CASE-BENCH-1": {
        "case_id": "CASE-BENCH-1",
        "patient_id": "SYN-PAT-BENCH1",
        "procedure": "Minor Excision",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 1, "sponge": 2},
        "open_items": [],
        "porter_release_allowed": False,
    },
    "CASE-BENCH-2": {
        "case_id": "CASE-BENCH-2",
        "patient_id": "SYN-PAT-BENCH2",
        "procedure": "Laparoscopic Cholecystectomy",
        "phase": "pre_op_setup",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
        "open_items": [],
        "porter_release_allowed": False,
    },
    "CASE-BENCH-3": {
        "case_id": "CASE-BENCH-3",
        "patient_id": "SYN-PAT-BENCH3",
        "procedure": "Biopsy",
        "phase": "case_closing_candidate",
        "priority": "normal",
        "required_items": {"scissors": 1, "sponge": 2},
        "open_items": ["pathology_handoff_task_not_created"],
        "expected_specimen": True,
        "specimen_destination": "synthetic_pathology",
    },
    "CASE-BENCH-4": {
        "case_id": "CASE-BENCH-4",
        "patient_id": "SYN-PAT-BENCH4",
        "procedure": "Open Conversion (Changed from Laparoscopic Appendectomy)",
        "phase": "procedure_changed_after_setup",
        "priority": "high",
        "required_items": {"scalpel": 3, "scissors": 2, "sponge": 6, "tweezers": 2},
        "open_items": ["updated_setup_not_confirmed", "porter_not_released"],
        "porter_release_allowed": False,
    },
    "CASE-BENCH-5": {
        "case_id": "CASE-BENCH-5",
        "patient_id": "SYN-PAT-BENCH5",
        "procedure": "Laparoscopic Biopsy",
        "phase": "pre_op_setup",
        "priority": "high",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
        "open_items": [],
        "porter_release_allowed": False,
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
        "specimen_handoff",
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
        "phase": case["phase"],
        "priority": case["priority"],
        "required_items": case["required_items"],
        "open_items": case["open_items"],
        "porter_release_allowed": case.get("porter_release_allowed", False)
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