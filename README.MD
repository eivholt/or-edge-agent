# OR Edge Agent

An operating-room edge logistics demo that detects instruments, reconciles the
scene with a synthetic surgical pathway, checks sterile-drape boundaries, and
executes bounded logistics actions.

The complete application runs on the Qualcomm IQ9 IQ9075 EVK:

- Edge Impulse FOMO detector on AArch64
- Ministral 3B Q4 text and tool model on the HTP through QAIRT Genie
- Ministral 3B multimodal projector through `llama.cpp` on the CPU
- Pydantic AI orchestration, synthetic EMR, and dashboard in Python 3.12

See [IQ9_EVK.md](IQ9_EVK.md) for the tested device setup, service commands,
validation steps, and troubleshooting notes.

## Repository Setup

The detector runners are stored with Git LFS. Install LFS before checking out
the repository so the model files are materialized rather than left as pointer
files.

```bash
git lfs install
git lfs pull
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
cp .env.example .env
```

On x86-64, the detector uses `models/modelfile.eim`. On AArch64, it selects
`models/modelfile.aarch64.eim`. Set `EI_MODEL_PATH` only when an explicit model
override is needed.

## Local Services

For a workstation with separately managed OpenAI-compatible model endpoints:

```bash
./start.sh app
```

This starts the synthetic EMR on port `9000` and dashboard on port `8000`.
Configure the text and vision endpoints in `.env` before running a scenario.

```bash
python -m apps.agent.run_fixture scenarios/all_present.json
```

## Tests

```bash
python -m pytest tests/ -m 'not llm' -q --tb=short
python -m pytest tests/ -v --tb=short
```

The full suite requires the text model, multimodal model, and synthetic EMR.
The IQ9 guide contains the exact tested endpoint configuration.

## Safety Scope

This repository uses synthetic records and scenes. It is an operational
logistics demonstration, not clinical decision support, diagnosis, treatment,
case clearance, or a real clinical alarm system.