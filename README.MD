# OR Edge Agent

Operating-room edge intelligence agent: real-time instrument detection, reconciliation against the surgical pathway, and autonomous logistics actions (SPD resupply, prep-light control, task creation).

## Quick Start

```bash
source .venv/bin/activate
./start.sh            # starts EMR (:9000), Dashboard (:8000), vLLM (:8081)
./start.sh stop       # stops all services
./start.sh status     # show running services
```

## Edge Impulse Model Management

The object-detection model lives at `models/modelfile.eim` (Edge Impulse Linux Runner format).

**Important:** The `.eim` must be built as **float32** for accurate detection. An int8-quantized build will misclassify instruments (e.g. scalpels detected as tweezers) due to quantization loss in the FOMO architecture.

### Download float32 model via API

```bash
export EI_API_KEY="ei_..."   # from EI Studio > project 371734 > Dashboard > Keys
python scripts/download_float32_model.py
```

This script will:
1. Trigger a **float32** `runner-linux-x86_64` build via the EI Studio API
2. Poll until the build job completes (~1–3 min)
3. Back up the current model as `models/modelfile.eim.int8.bak`
4. Save the new float32 model to `models/modelfile.eim`

**Project:** 371734 — "Surgery Inventory Synthetic NVIDIA"
**Labels:** scalpel, scissors, sponge, tweezers
**Input:** 320×320 RGB, FOMO (constrained object detection)
**Confidence threshold:** 0.9 (set in `apps/detector/inference.py`)

### Verify detection accuracy

```bash
python -c "
from apps.detector.inference import detect_instruments
result = detect_instruments('data/frames/frame_missing_scissors3.png')
print(result)
"
```

Expected float32 output should match EI Studio results (scalpel:2, scissors:2, sponge:4, tweezers:2 at ≥0.99 confidence).

---

## Dev Notes

ssh eivho@192.168.1.107 -p 2222

uvicorn synthetic_emr.api:app --reload --port 9000


uv pip install \
  --extra-index-url https://wheels.vllm.ai/0.20.2/cu129 \
  vllm==0.20.2

uv venv --python python3.12 .venv-vllm --seed
source .venv-vllm/bin/activate

uv pip install vllm --torch-backend=cu129

python - <<'PY'
import torch, vllm
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print(vllm.__version__)
PY

nvidia-smi

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.70 \
  --api-key local-dev-key
  --enable-auto-tool-choice --tool-call-parser hermes

  eivho@DESKTOP-OM4KVB3:~/repos/or-edge-agent$ source .venv/bin/activate

  python - <<'PY'
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local-dev-key"
)

r = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "Return only JSON."},
        {"role": "user", "content": "Return {\"status\":\"ok\"}."}
    ],
    temperature=0
)

print(r.choices[0].message.content)
PY


eivho@DESKTOP-OM4KVB3:~/repos/or-edge-agent$ source .venv-vllm/bin/activate

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8081 \
  --gpu-memory-utilization 0.45 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --api-key local-dev-key


  vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8081 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --allowed-local-media-path /home/eivho/repos-native/or-edge-agent/data \
  --api-key local-dev-key


  cd /home/eivho/repos-native/or-edge-agent && source .venv-vllm/bin/activate && vllm serve mistralai/Ministral-3-3B-Instruct-2512-BF16 --host 0.0.0.0 --port 8081 --dtype auto --max-model-len 4096 --gpu-memory-utilization 0.70 --enable-auto-tool-choice --tool-call-parser mistral 2>&1

curl http://localhost:8081/v1/chat/completions \
  -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this image briefly."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///home/eivho/repos-native/or-edge-agent/data/surgery_tools_1024.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 200
  }'

source .venv/bin/activate
export TEXT_LLM_BASE_URL=http://localhost:8000/v1
export TEXT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export OPENAI_API_KEY=local-dev-key

python apps/agent/run_fixture.py scenarios/missing_scissors.json


ip addr

netsh interface portproxy add v4tov4 `
listenaddress=0.0.0.0 listenport=2222 `
connectaddress=172.24.98.15 connectport=22

sudo service ssh start
sudo service ssh status

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

Image prompts:

1. Sterile Zone Ambiguity

Top-down clinical photograph of a stainless-steel back table in an operating room, covered with a blue sterile drape. Surgical instruments (scalpel, scissors, sponges, tweezers) are laid out, but two instruments are partially hanging off the edge of the sterile drape, ambiguously straddling the boundary between sterile and non-sterile zones. Bright OR overhead lighting, slightly out-of-focus background. No people visible. DSLR, 35mm lens, ISO 400, clinical documentation style. No text, no watermarks, no logos. Stock photo style, editorial, for medical training simulation. 1:1 aspect ratio.

2. Instrument Out of Zone

Top-down clinical photograph of an operating room back table covered with a blue sterile surgical drape. Neatly arranged surgical instruments on the drape (scissors, sponges, tweezers). One stainless-steel scalpel is clearly placed 15 cm outside the drape boundary, sitting directly on the bare stainless-steel table surface. Sharp focus on the scalpel outside the drape. Bright overhead surgical lighting. DSLR, 35mm lens, ISO 400, clinical documentation style. No text, no watermarks, no logos. Stock photo style, editorial, for medical training simulation. 1:1 aspect ratio.

3. Specimen Ready Check

Clinical photograph of a mayo stand (small wheeled surgical tray) in an operating room. On the mayo stand are a pair of scissors and two gauze sponges. Next to them sits a small translucent plastic container with a screw-top lid resembling a specimen cup. The container is unlabeled and partially obscured by the sponges. Stainless-steel tray surface, blue drape underneath, blurred OR background. DSLR, 35mm lens, ISO 400, clinical documentation style. No text, no watermarks, no logos. Stock photo style, editorial, for medical training simulation. 1:1 aspect ratio.

4. Room Turnover Check

Wide-angle clinical photograph of an operating room between procedures. The room is partially cleaned but a hospital gurney with rumpled white sheets is still parked in the corner. An IV pole with an empty fluid bag hangs nearby. The main surgical table is bare. Overhead OR lights are on. Floor shows wheel marks. The scene conveys an incomplete room turnover — previous patient equipment not yet removed. DSLR, 35mm lens, ISO 400, clinical documentation style. No text, no watermarks, no logos. Stock photo style, editorial, for medical training simulation. 1:1 aspect ratio.

5. PPE Compliance Check

Clinical photograph of an operating room instrument table from a ceiling-mounted camera angle. A person in blue surgical scrubs stands next to the sterile instrument table, wearing a surgical cap and gloves but conspicuously missing a surgical face mask — their nose and mouth are uncovered. Instruments are visible on the table. Bright surgical lighting, other masked staff blurred in the background. DSLR, 35mm lens, ISO 400, clinical documentation style. No text, no watermarks, no logos. Stock photo style, editorial, for medical training simulation. 1:1 aspect ratio.