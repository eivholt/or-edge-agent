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
  --port 8001 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --api-key local-dev-key


  vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --allowed-local-media-path /home/eivho/repos-native/or-edge-agent/data \
  --api-key local-dev-key

curl http://localhost:8001/v1/chat/completions \
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

python apps/agent/run_fixture.py scenarios/missing_suction_tip.json


ip addr

netsh interface portproxy add v4tov4 `
listenaddress=0.0.0.0 listenport=2222 `
connectaddress=172.24.98.15 connectport=22

sudo service ssh start
sudo service ssh status

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker