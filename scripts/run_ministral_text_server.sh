#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QAIRT_ROOT="${QAIRT_ROOT:-$HOME/qairt-2.47.0.260601}"
SERVICE_BIN="${SERVICE_BIN:-$HOME/src/qai-appbuilder-full/samples/genie/c++/Service/GenieService_v2.1.5_qnnunknown/GenieAPIService}"
BUNDLE="${BUNDLE:-$HOME/ministral_q4_genie_export}"
CONFIG_FILE="${CONFIG_FILE:-genie_config.agent.json}"
CPP_PORT="${CPP_PORT:-8911}"
PORT="${PORT:-8001}"
TARGET="aarch64-oe-linux-gcc11.2"

for path in "$SERVICE_BIN" "$BUNDLE/$CONFIG_FILE" "$BUNDLE/prompt.json"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

SERVICE_DIR="$(dirname "$SERVICE_BIN")"
export PATH="$QAIRT_ROOT/bin:$QAIRT_ROOT/bin/$TARGET:$PATH"
export LD_LIBRARY_PATH="$SERVICE_DIR:$QAIRT_ROOT/lib/$TARGET:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$QAIRT_ROOT/lib/hexagon-v73/unsigned;${ADSP_LIBRARY_PATH:-}"

"$SERVICE_BIN" \
  --config_file "$BUNDLE/$CONFIG_FILE" \
  --load_model \
  --num_response 0 \
  --min_output_num 512 \
  --port "$CPP_PORT" &
service_pid=$!

cleanup() {
  kill "$service_pid" 2>/dev/null || true
  wait "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if curl --silent --fail "http://127.0.0.1:$CPP_PORT/" >/dev/null; then
    break
  fi
  if ! kill -0 "$service_pid" 2>/dev/null; then
    wait "$service_pid"
    exit 1
  fi
  sleep 1
done

curl --silent --fail "http://127.0.0.1:$CPP_PORT/" >/dev/null
python3 "$ROOT/scripts/genie_cpp_adapter.py" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --upstream-url "http://127.0.0.1:$CPP_PORT/v1" \
  --upstream-model "$(basename "$BUNDLE")" \
  --model-name ministral3-3b-q4
