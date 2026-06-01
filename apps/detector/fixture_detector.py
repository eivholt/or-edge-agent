import json
import time
from pathlib import Path


def main():
    scenario_paths = [
        "scenarios/missing_scissors.json",
        "scenarios/procedure_changed.json"
    ]

    Path("logs").mkdir(exist_ok=True)

    with open("logs/events.ndjson", "a", encoding="utf-8") as out:
        for path in scenario_paths:
            event = json.loads(Path(path).read_text())
            out.write(json.dumps(event) + "\n")
            out.flush()
            print("emitted", event["case_id"])
            time.sleep(2)


if __name__ == "__main__":
    main()