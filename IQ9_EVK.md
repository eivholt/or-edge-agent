# Build An On-Device Surgical Logistics Agent On The IQ9

This tutorial shows edge AI developers how to build and run a complete agentic
application on a Qualcomm IQ9 IQ9075 evaluation kit. The application is a
synthetic operating-room logistics demo: it observes a room prepared for
surgery, compares the instruments it can see with a synthetic case record, and
requests operational follow-up when supplies are missing or an instrument may
be outside the sterile work area.

No medical background is required. The **sterile drape** is the green covering
that defines the clean work surface around a patient. An instrument beyond its
boundary may need a person to review the scene. The **electronic medical
record** in this demo is a local synthetic service that supplies the expected
instrument list; it contains no real patient data. The application demonstrates
logistics workflow only. It does not make diagnoses, recommend treatment, clear
a surgical case, or act as a clinical alarm.

The architecture highlights three reusable edge-agent patterns:

1. **Object detection for inventory.** An Edge Impulse FOMO model detects and
  counts instruments before deterministic quantity reconciliation.
2. **One local multimodal large language model in two roles.** Ministral powers
   the tool-using agent through a text endpoint. The agent can also call a
  projector-backed Ministral endpoint to inspect detector-centered image
  segments. The detector selects context; only the VLM classifies the support
  surface. Deterministic application code retains the final workflow policy.
3. **A visual, on-device execution trace.** The dashboard shows the input image,
   detector results, case-record lookup, supply comparison, visual inspection,
   tool actions, and final status light as the workflow runs on the device.

The sections below assemble those parts, run the application, and verify the
full detector-to-agent path. They use the already-exported Ministral Qualcomm AI
Runtime bundle. Do not rebuild or re-export the text model when these artifacts
are present.

The configuration below was validated on:

- Ubuntu AArch64, kernel `6.8.0-1077-qcom`
- Python `3.12.3`
- Qualcomm AI Runtime `2.47.0.260601`, IQ9075, Hexagon v73
- Ministral 3B Q4 text model on the Qualcomm Hexagon Tensor Processor
- Ministral 3B Q4 decoder plus official BF16 vision projector on CPU
- Edge Impulse float32 AArch64 runner

## 1. Verify The Existing Qualcomm AI Runtime Setup

Complete the `qai-nemotron` evaluation-kit setup first. This application reuses
its Qualcomm AI Runtime environment, Genie service, native tool-call adapter,
and exported model bundle. The required paths are:

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

Check them on the evaluation kit:

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

Install Git Large File Storage before cloning because both detector runners are
stored as large-file objects.

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
ssh ubuntu@<device-ip> 'mkdir -p ~/models/ministral3_vlm'
rsync -ah --progress \
  Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf \
  ubuntu@<device-ip>:models/ministral3_vlm/
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
VLM_TIMEOUT_SECONDS=600
VLM_SEGMENT_RADIUS=64
VLM_SEGMENT_IMAGE_SIZE=224

EMR_BASE_URL=http://127.0.0.1:9000
OPENAI_API_KEY=local-dev-key
```

The split is deliberate. Port `8001` provides native tool calls through the
Qualcomm Hexagon Tensor Processor. Port `8082` provides actual image processing
through the Ministral projector.

## 5. Start The Agent's Text Model

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

CPU vision is the expensive path. The application crops a square around every
FOMO centroid and sends each crop to the VLM independently. A radius of 64 in
the detector's 320x320 coordinate system produces a 410x410 context from these
1024x1024 fixtures; the VLM copy is normalized to 224x224 while the dashboard
keeps the larger crop for inspection.

Use one server slot, a 4096-token context, and a fixed 64-token image budget for
the reproducible all-case benchmark below. Disable prompt and slot reuse because
cached visual tokens can be reused across different images. Disable hidden
reasoning so the 16-token guided JSON response is actually emitted:

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
  --image-min-tokens 64 \
  --image-max-tokens 64 \
  --no-cache-prompt \
  --slot-prompt-similarity 0 \
  --reasoning off \
  --reasoning-budget 0 \
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

### How The Image Encoder Reaches The Language Model

The JPEG or PNG does not become ordinary text and the language decoder does not
read pixels directly. The multimodal path has four stages:

1. `llama.cpp` decodes and resamples the image according to the configured
  visual-token budget. The resulting patch grid controls how much spatial
  detail survives.
2. Ministral's vision encoder transforms those patches into continuous visual
  feature vectors. Nearby patches carry local appearance and position, but
  this stage does not emit words such as `scissors` or `green`.
3. The BF16 multimodal projector maps the vision encoder's feature dimension
  into the language decoder's embedding dimension. These projected vectors are
  inserted into the prompt as visual tokens beside the user's text tokens.
4. The Q4 Ministral decoder attends to both token types and autoregressively
  emits the guided JSON answer. In this application the only accepted decision
  value is `{ "answer": true | false }`.

`--image-min-tokens` and `--image-max-tokens` therefore control encoder detail,
not the 16-token output limit. More visual tokens preserve a denser view, but
they also make CPU prompt evaluation much slower. Resizing a crop to 224x224 did
not materially lower latency while the server remained fixed at 256 visual
tokens: the server still produced about 857 prompt tokens and took about 116
seconds. Lowering the server budget to 64 visual tokens reduced the prompt to
about 657 tokens and the call to about 39 seconds.

The files also have different precisions. The vision projector is the official
BF16 artifact; the autoregressive decoder is Q4. It is inaccurate to describe
the entire IQ9 vision stack as Q4.

## 7. Start The Synthetic Case Service And Dashboard

The model processes are externally managed on the evaluation kit, so use `app`
mode:

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
http://<device-ip>:8000
```

Keep the dashboard visible when you run a scenario. It presents each on-device
step in order: object detections, expected supplies from the synthetic case
service, quantity differences, every targeted visual segment, agent tool calls,
and the resulting green, yellow, or red status light. The Local VLM node shows
`processed / total`, the current segment, and elapsed time. A yellow crop border
means inference is running, green means the VLM classified the instrument as on
the sterile drape, and red means the VLM classified it as on bare metal.

### Follow One Run In The Dashboard

The following screenshots are from one offline **Instrument Out of Zone** run
on the evaluation kit. No cloud service participated. The complete run took
6 minutes 26 seconds on the validated configuration. Operation times come from
the dashboard event log; cumulative times are measured from the detector event
and rounded to the nearest second. Generative-model latency varies between
runs, so treat these numbers as an observed trace rather than fixed targets.

#### 1. Supply A Camera Frame

![Prepared camera frame showing instruments on and beside a green sterile drape](resources/instrument-out-of-zone-01-video-input.png)

*Elapsed: 0 seconds.* A prepared image stands in for a live camera frame in this
demo. It shows eight instruments around a green sterile drape. The scissors at
the lower right are on the uncovered metal surface. The application receives
only the image, synthetic case identifier, and room identifier; the scenario
does not contain expected detections or the answer.

#### 2. Detect And Locate Instruments

![Edge Impulse object detector output with eight labeled instruments](resources/instrument-out-of-zone-02-object-detection.png)

*Operation time: 20 milliseconds; cumulative elapsed: less than 1 second.* The
Edge Impulse FOMO object-detection model finds two tweezers, two scalpels, three
sponges, and one pair of scissors. These counts become the agent's observed
supplies. The boxes also gate visual inspection by defining the small image
segments sent to the slower multimodal model; detector geometry does not decide
whether an instrument is inside the sterile area.

#### 3. Retrieve The Synthetic Case Record

![Synthetic case record listing the expected instrument quantities](resources/instrument-out-of-zone-03-case-record.png)

*Service time: 42 milliseconds; cumulative elapsed: 9 seconds.* Ministral's
first tool call retrieves synthetic case `CASE-1045`. The named procedure is a
type of gallbladder-removal surgery, but its name does not drive any action.
Only the required instrument quantities are used in the supply comparison.

#### 4. Reconcile Detected And Required Counts

![Reconciliation node showing that all required instruments are present](resources/instrument-out-of-zone-04-reconciliation.png)

*Tool time: 2 milliseconds; cumulative elapsed: 16 seconds.* A deterministic
Pydantic AI tool compares the detector counts with the synthetic case record.
Every required instrument is present and there are no unexpected items, so the
agent does not request resupply. This quantity check contains no language-model
or network logic.

#### 5. Inspect Detector-Centered Image Segments

![Local Ministral visual inspector showing seven inside segments and one outside segment](resources/instrument-out-of-zone-05-visual-inspection.png)

*Operation time: 5 minutes 9 seconds; cumulative completion: 5 minutes
35 seconds.* The agent calls the local visual inspector, which sends each of the
eight detector-centered segments through the multimodal Ministral endpoint.
The segments run serially on the CPU and take 38-39 seconds each with the
64-visual-token configuration. Seven return `false` to the question of whether
the instrument is outside the drape. Segment 8 returns `true`, so the combined
scene verdict flags the scissors. This boolean is produced by the multimodal
model, not by pixel color, detector coordinates, or instrument class.

#### 6. Complete The Agent Workflow

![Local Ministral agent summary with five tool calls and five model iterations](resources/instrument-out-of-zone-06-agent.png)

*Agent time: 6 minutes 25 seconds, including all nested tool calls.* The local
Ministral 3B model is orchestrated by Pydantic AI and prompted to gather the
case, compare supplies, inspect the scene, perform every applicable operational
action, and set exactly one status light. In five model iterations it calls
`get_case`, `check_supplies`, `inspect_scene`, `set_stacklight`, and
`create_task`. After the visual result, it selects the bounded sterile-area
workflow: red status plus one human-review task, with no resupply request.

#### 7. Create A Human-Review Task

![Task queue containing one high-priority human-review task](resources/instrument-out-of-zone-07-review-task.png)

*Cumulative elapsed: 6 minutes 5 seconds.* The task tool records one
high-priority request for a person to review scissors segment 8. The action is
operational and deliberately bounded: the application does not diagnose a
problem or decide whether surgery may proceed.

#### 8. Set The Physical Status Light

![Stack light node showing red for the detected sterile-area violation](resources/instrument-out-of-zone-08-stack-light.png)

*Cumulative elapsed: 6 minutes 5 seconds.* The agent sets the room's simulated
stack light to red and supplies the reason shown beneath it. The light is an
operational demo output. It is not a clinical alarm or case-clearance system.

#### 9. Validate The Tool Calls

![Validation node showing that all safety checks passed](resources/instrument-out-of-zone-09-validation.png)

*Total elapsed: 6 minutes 26 seconds.* Deterministic validation runs after the
agent finishes. It checks the tool allowlist and validates task, resupply, and
light arguments. This run passes all checks: it contains one allowed
human-review task, no supply action, and exactly one valid red-light action.

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

Run the five detector-to-agent integration scenarios only when the required
model services are available:

```bash
.venv/bin/python -m pytest tests/test_integration.py -q --tb=short
```

The reliability test repeats each scenario three times. Every detected
instrument invokes CPU vision, so this suite is not an acceptable quick smoke
test on the current CPU VLM path. Use the service-independent tests for routine
checks and treat the measured VLM benchmark below as the current acceptance
evidence.

## 9. Expected Performance And Policy

The validated device behavior is:

- FOMO detection: approximately 12-22 ms per 1024x1024 fixture
- Hexagon Tensor Processor text/tool request: approximately 7-30 seconds
  depending on context
- 64-token, 224x224 centroid segment: approximately 38.5-39.0 seconds
- 256-token, 224x224 centroid segment: approximately 116.2-116.5 seconds

FOMO supplies instrument counts for reconciliation. It does not decide whether
the sterile zone is clear. Detector boxes are used only to center crops. Every
crop receives its own 16-token guided boolean request, and the scene verdict is
the logical OR of those VLM booleans. No pixel-color check, detector coordinate,
or instrument class supplies a sterile-zone verdict.

### Full-Frame Results

The known positive fixture is `instrument_out_of_zone`; a correct answer is
`true`.

| Runtime and input | Visual/prompt tokens | Time | Answer | Result |
| --- | ---: | ---: | --- | --- |
| RTX 5090, BF16 vLLM, full frame | backend default | 0.458 s | `true` | correct |
| IQ9, Q4 decoder + BF16 projector, full frame | 1024 visual | 446.3 s | `false` | false negative |
| IQ9, Q4 decoder + BF16 projector, full frame | 2048 visual / 2778 prompt | 945.3 s | `false` | false negative |

Doubling the IQ9 image budget more than doubled latency and did not repair the
miss. The RTX BF16 result does not prove that quantization is the sole cause.
The host also used vLLM, a different image preprocessing path, a longer prompt
and output schema, and a backend-selected visual-token budget.

Historical host results have two distinct meanings. An earlier full-application
run on the BF16 vLLM host reported all five scenarios passing three consecutive
runs, or 15/15. That run verified the tool workflow, including calls to
`inspect_scene` and `set_stacklight`, against the policy and fixtures in use at
the time. Later host runs also showed BF16 producing the positive sterile-zone
result and red-light workflow, but the two positive integration scenarios were
not consistently stable across every subsequent suite run.

Separately, the committed standalone VLM snapshot in
`tests/vlm_benchmark_results.json` is 6/7: it records a 0.393-second false
negative for `sterile_zone_ambiguity`. That later snapshot must not be described
as disproving the earlier 15/15 application acceptance, nor should the earlier
application run be presented as a repeatable 7/7 standalone VLM benchmark. The
prompt, fixture set, policy assertions, and benchmark surface changed between
those runs.

A controlled BF16-versus-Q4 test through the same `llama.cpp` build, projector,
prompt, and image budget would be required to isolate quantization. The
defensible conclusion is that Q4 may reduce the margin of an already fragile 3B
spatial reasoner, not that Q4 alone caused the failure.

### Detector-Centered Crop Results

Crop radii 48, 64, and 80 all detected the out-of-zone scissors with the tuned
support-surface prompt at 64 visual tokens. Radius 64 was selected because it
retains more surrounding edge context than 48 while keeping the instrument
larger than radius 80 after normalization.

The complete radius-64 run processed all 42 detections without short-circuiting:

| Scenario | Segments | Ground truth | 64-token verdict | Time | Correct |
| --- | ---: | --- | --- | ---: | --- |
| `all_present` | 11 | `false` | `false` | 424.0 s | yes |
| `instrument_out_of_zone` | 8 | `true` | `true` | 309.2 s | yes |
| `missing_scissors` | 9 | `false` | `true` | 348.9 s | no |
| `missing_something` | 7 | `false` | `true` | 270.8 s | no |
| `sterile_zone_ambiguity` | 7 | `true` | `false` | 270.4 s | no |

The total was 1624.1 seconds (27 minutes 4 seconds), or **2/5 cases
correct**. At 16 visual tokens, calls fell to 17.7 seconds but a controlled
out-of-zone crop became a false negative. At 256 visual tokens, the three crop
decisions responsible for the 64-token case failures were all corrected, but
each took 116.2-116.5 seconds. Applying that cost to all 42 segments is roughly
81 minutes, or 9-21 minutes per case.

The practical conclusion is negative: centroid cropping improves observability
and can recover the obvious `instrument_out_of_zone` case, but no tested token
budget is both accurate and fast enough on the IQ9 CPU vision path. The 64-token
profile is useful for reproducing the experiment and dashboard progress; it is
not a validated sterile-zone classifier. The 256-token probes show that more
local visual detail can repair the observed errors, but their performance is
not operationally acceptable. Do not represent either profile as clinical or
production acceptance.

## Troubleshooting

**Port `8082` cannot bind**

The previous `llama-server` or a foreground SSH launcher may still own the
socket. Confirm with `ss -ltnp | grep ':8082'`, terminate the listed process,
and wait until the listener is gone before restarting.

**Vision request exceeds 600 seconds**

Confirm `--parallel 1`, the intended image-token budget, and
`VLM_TIMEOUT_SECONDS=600`. The reproducible crop benchmark uses 64 image tokens;
the higher-detail diagnostic profile uses 256. Full-frame 1024- and 2048-token
requests are retained as measured experiments, not recommended runtime settings.

**A different image appears to reuse the previous visual result**

Start `llama-server` with `--no-cache-prompt` and
`--slot-prompt-similarity 0`. Default longest-common-prefix slot reuse can
retain visual prompt tokens between requests even when the image changes.

**Guided JSON returns empty content**

Start `llama-server` with `--reasoning off --reasoning-budget 0`. Otherwise the
model can consume the short completion budget in `reasoning_content` before it
emits the required JSON object.

**The multimodal model returns text but ignores the image**

Do not point `VLM_BASE_URL` at port `8001`. Genie serves the text/tool decoder;
the projector-backed `llama-server` is on `8082`.

**Edge Impulse import asks for PyAudio**

Do not import the package root in an image-only smoke test. Import
`apps.detector.inference` and call `detect()`; it bypasses the unused audio
module.

**The wrong detector binary runs**

On the evaluation kit, `default_model_path()` must select
`models/modelfile.aarch64.eim`. Check `uname -m`, the file checksum, and any
`EI_MODEL_PATH` override.

**Stop services**

`./start.sh stop` stops the dashboard and synthetic case service. The externally
managed model processes can be stopped with their recorded process identifiers
or the following targeted commands:

```bash
pkill -f 'shipping_agent.genie_cpp_adapter'
pkill -f 'GenieAPIService.*8911'
pkill -f 'llama-server.*--port 8082'
```