# Run OR Edge Agent on the IQ9 EVK

This tutorial deploys the entire OR Edge Agent to a Qualcomm IQ9 IQ9075 EVK.
It uses the already-exported Ministral QAIRT bundle. Do not rebuild or re-export
the text model when these artifacts are present.

The configuration below was validated on:

- Ubuntu AArch64, kernel `6.8.0-1077-qcom`
- Python `3.12.3`
- QAIRT `2.47.0.260601`, IQ9075, Hexagon v73
- Ministral 3B Q4 text model on QNN HTP
- Ministral 3B Q4 decoder plus official BF16 vision projector on CPU
- Edge Impulse float32 AArch64 runner

## 1. Verify The Existing QAIRT Setup

Complete the `qai-nemotron` EVK setup first. This application reuses its QAIRT
environment, Genie service, native tool-call adapter, and exported model bundle.
The required paths are:

```text
~/qairt-env.sh
~/qairt-2.47.0.260601/
~/qai-nemotron/
~/src/qai-appbuilder-full/samples/genie/c++/Service/GenieService_v2.1.5_qnnunknown/GenieAPIService
~/ministral_q4_genie_export/genie_config.agent.json
~/ministral_q4_genie_export/artifacts/split_model_1.bin
~/ministral_q4_genie_export/artifacts/split_model_2.bin
~/ministral_q4_genie_export/artifacts/tokenizer.json
```

Check them on the EVK:

```bash
source "$HOME/qairt-env.sh"
test "$PRODUCT_SOC" = 9075
test "$DSP_ARCH" = 73
test -x "$HOME/src/qai-appbuilder-full/samples/genie/c++/Service/GenieService_v2.1.5_qnnunknown/GenieAPIService"
test -f "$HOME/ministral_q4_genie_export/genie_config.agent.json"
test -f "$HOME/ministral_q4_genie_export/artifacts/split_model_1.bin"
test -f "$HOME/ministral_q4_genie_export/artifacts/split_model_2.bin"
```

The checked-in Genie config has a 4096-token context and a `QnnHtp` backend.
Model conversion is intentionally outside this tutorial.

## 2. Install The Application

Install Git LFS before cloning because both detector runners are LFS objects.

```bash
sudo apt update
sudo apt install -y git git-lfs python3.12-venv
git lfs install

cd "$HOME"
git clone <repository-url> or-edge-agent
cd or-edge-agent
git lfs pull

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
```

Verify that the ARM detector is a binary, not a Git LFS pointer:

```bash
file models/modelfile.aarch64.eim
sha256sum models/modelfile.aarch64.eim
```

The validated SHA-256 is:

```text
90b5809422276b14db3ba0c47ecc3061bbb4b76c78b7ac0d093ca182b2ac0238
```

`apps/detector/inference.py` selects this file automatically on `aarch64` and
`arm64`. It also avoids the unused `edge_impulse_linux.audio` import, so PyAudio
is not required for image inference.

## 3. Place The Vision Files

The on-device vision server needs the Q4 Ministral decoder and the matching
official BF16 projector from the same Ministral release:

```text
~/models/ministral3_vlm/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
~/models/ministral3_vlm/Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf
```

If the files already exist on a workstation, copy them without rebuilding:

```bash
ssh ubuntu@<evk-ip> 'mkdir -p ~/models/ministral3_vlm'
rsync -ah --progress \
  Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf \
  ubuntu@<evk-ip>:models/ministral3_vlm/
```

Expected sizes are approximately 2.15 GB for the decoder and 842 MB for the
projector. The projector is essential: the Genie text endpoint does not consume
OpenAI `image_url` content.

The tested `llama.cpp` binary is:

```text
~/llama.cpp/build-native/bin/llama-server
```

It reports build `0dc74e3` for Linux AArch64.

## 4. Configure Endpoint Splitting

Copy the tracked IQ9 profile:

```bash
cd "$HOME/or-edge-agent"
cp .env.iq9.example .env
```

It contains:

```dotenv
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_MODEL=ministral3-3b-q4
LLM_API_KEY=local-dev-key
LLM_MAX_CONTEXT=4096

VLM_BASE_URL=http://127.0.0.1:8082/v1
VLM_MODEL=ministral3-3b-vl-q4
VLM_API_KEY=local-dev-key
VLM_TIMEOUT_SECONDS=180
STERILE_GREEN_CONTEXT_THRESHOLD=0.70

EMR_BASE_URL=http://127.0.0.1:9000
OPENAI_API_KEY=local-dev-key
```

The split is deliberate. Port `8001` provides native tool calls through HTP.
Port `8082` provides actual image processing through the Ministral projector.

## 5. Start The HTP Text Model

The `qai-nemotron` launcher starts the C++ Genie service on `8911` and exposes
it as an OpenAI-compatible endpoint on `8001`:

```bash
cd "$HOME/qai-nemotron"
mkdir -p "$HOME/or-edge-agent/logs"

nohup env \
  QAIRT_ENV="$HOME/qairt-env.sh" \
  QAIRT_ROOT="$HOME/qairt-2.47.0.260601" \
  BUNDLE="$HOME/ministral_q4_genie_export" \
  CONFIG_FILE=genie_config.agent.json \
  CPP_PORT=8911 \
  PORT=8001 \
  bash shipping_agent/run_ministral_cpp_server.sh \
  > "$HOME/or-edge-agent/logs/ministral-text.log" 2>&1 < /dev/null &
```

Wait for the adapter and confirm the model alias:

```bash
curl --fail http://127.0.0.1:8001/v1/models
```

The response must contain `ministral3-3b-q4`.

## 6. Start Multimodal Ministral

CPU vision is the expensive path. Use one server slot, a 4096-token context,
and a 128-256 image-token budget:

```bash
cd "$HOME/or-edge-agent"
nohup "$HOME/llama.cpp/build-native/bin/llama-server" \
  -m "$HOME/models/ministral3_vlm/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf" \
  --mmproj "$HOME/models/ministral3_vlm/Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf" \
  --alias ministral3-3b-vl-q4 \
  --host 127.0.0.1 \
  --port 8082 \
  --ctx-size 4096 \
  --parallel 1 \
  --threads 8 \
  --threads-batch 8 \
  --image-min-tokens 128 \
  --image-max-tokens 256 \
  --jinja \
  > logs/ministral-vlm.log 2>&1 < /dev/null &
```

Verify that the server loaded both model files:

```bash
curl --fail http://127.0.0.1:8082/health
curl --fail http://127.0.0.1:8082/v1/models
grep -E 'loaded multimodal model|model loaded' logs/ministral-vlm.log
```

The model capabilities must include `multimodal`. An HTTP 200 from a text-only
adapter is not proof that image content reached the model.

## 7. Start The EMR And Dashboard

The model processes are externally managed on the EVK, so use `app` mode:

```bash
cd "$HOME/or-edge-agent"
OPEN_BROWSER=0 ./start.sh app
```

Check all application endpoints:

```bash
curl --fail http://127.0.0.1:8001/v1/models
curl --fail http://127.0.0.1:8082/health
curl --fail http://127.0.0.1:9000/openapi.json >/dev/null
curl --fail http://127.0.0.1:8000/ >/dev/null
```

Open the dashboard from another machine at:

```text
http://<evk-ip>:8000
```

## 8. Validate On Device

Run service-independent tests first:

```bash
cd "$HOME/or-edge-agent"
.venv/bin/python -m pytest tests/ -m 'not llm' -q --tb=short
```

Verify real AArch64 detector inference:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from apps.detector.inference import detect

for image in sorted(Path("data/frames").glob("*.png")):
    result = detect(image)
    print(image.name, len(result.detections), f"{result.inference_ms:.1f} ms")
PY
```

Run one complete autonomous workflow:

```bash
.venv/bin/python -m apps.agent.run_fixture scenarios/all_present.json
```

Run the five detector-to-agent integration scenarios:

```bash
.venv/bin/python -m pytest tests/test_integration.py -q --tb=short
```

The reliability test repeats each scenario three times. Positive sterile-zone
fixtures invoke CPU vision and therefore take several minutes.

## 9. Expected Performance And Policy

The validated device behavior is:

- FOMO detection: approximately 12-22 ms per 1024x1024 fixture
- HTP text/tool request: approximately 7-30 seconds depending on context
- first targeted CPU vision request: approximately 103-106 seconds
- clean sterile scene: VLM skipped after detector context analysis

FOMO provides instrument centroids on its 320x320 processed frame. The local
inspection path measures green-drape context around each centroid. Scenes with
no candidate below the configured 0.70 fraction finish without CPU vision.
Candidate scenes send a boundary crop to Ministral with a 16-token boolean JSON
schema. A detector-confirmed off-drape candidate remains actionable if the VLM
times out or disagrees; sterile workflow policy is not delegated solely to a
small generative model.

## Troubleshooting

**Port `8082` cannot bind**

The previous `llama-server` may still be releasing the socket. Confirm with
`ss -ltnp | grep ':8082'` before restarting.

**Vision request exceeds 180 seconds**

Confirm `--parallel 1`, `--image-max-tokens 256`, and the detector-guided crop
path. A full 1024-token image pass exceeded five minutes on this device.

**The VLM always returns a text answer but ignores the image**

Do not point `VLM_BASE_URL` at port `8001`. Genie serves the text/tool decoder;
the projector-backed `llama-server` is on `8082`.

**Edge Impulse import asks for PyAudio**

Do not import the package root in an image-only smoke test. Import
`apps.detector.inference` and call `detect()`; it bypasses the unused audio
module.

**The wrong detector binary runs**

On the EVK, `default_model_path()` must select
`models/modelfile.aarch64.eim`. Check `uname -m`, the file checksum, and any
`EI_MODEL_PATH` override.

**Stop services**

`./start.sh stop` stops the dashboard and EMR. The externally managed model
processes can be stopped with their recorded PIDs or the following targeted
commands:

```bash
pkill -f 'shipping_agent.genie_cpp_adapter'
pkill -f 'GenieAPIService.*8911'
pkill -f 'llama-server.*--port 8082'
```