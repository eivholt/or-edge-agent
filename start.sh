#!/usr/bin/env bash
# ── OR Edge Agent — Service Launcher ──────────────────────────────────
# Usage:
#   ./start.sh              Start all services (vLLM + EMR + Dashboard)
#   ./start.sh llm          Start vLLM only
#   ./start.sh app          Start EMR + Dashboard (model managed externally)
#   ./start.sh emr          Start EMR API only
#   ./start.sh dashboard    Start Dashboard only
#   ./start.sh agent [file] Run agent fixture (default: missing_scissors)
#   ./start.sh stop         Stop all services
#
# The dashboard auto-opens in Chrome once the server is ready.
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")"

# ── Config ────────────────────────────────────────────────────────────
VLLM_PORT=8081
EMR_PORT=9000
DASH_PORT=8000
DASH_URL="http://localhost:${DASH_PORT}"

VLLM_MODEL="mistralai/Ministral-3-3B-Instruct-2512-BF16"
VLLM_VENV=".venv-vllm"
APP_VENV=".venv"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ── Helpers ───────────────────────────────────────────────────────────

_color()  { printf "\033[%sm%s\033[0m\n" "$1" "$2"; }
info()    { _color "1;34" "▸ $1"; }
ok()      { _color "1;32" "✓ $1"; }
warn()    { _color "1;33" "⚠ $1"; }
err()     { _color "1;31" "✗ $1"; }

port_busy() { fuser "$1/tcp" >/dev/null 2>&1; }

kill_port() {
    if port_busy "$1"; then
        warn "Killing process on port $1"
        # Get the PID and kill the whole process group so child workers die too
        local pid
        pid=$(fuser "$1/tcp" 2>/dev/null | awk '{print $1}')
        if [[ -n "${pid:-}" ]]; then
            local pgid
            pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
            if [[ -n "${pgid:-}" && "$pgid" != "1" ]]; then
                kill -- -"$pgid" 2>/dev/null || true
                sleep 1
                kill -9 -- -"$pgid" 2>/dev/null || true
            else
                kill "$pid" 2>/dev/null || true
                sleep 1
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        fuser -k "$1/tcp" >/dev/null 2>&1 || true
        sleep 1
    fi
}

wait_for_port() {
    local port=$1 label=$2 timeout=${3:-120}
    info "Waiting for $label on :${port} (timeout ${timeout}s)..."
    for ((i=0; i<timeout; i++)); do
        if port_busy "$port"; then
            ok "$label is up on :${port}"
            return 0
        fi
        sleep 1
    done
    err "$label failed to start within ${timeout}s"
    return 1
}

open_browser() {
    local url=$1
    if [[ "${OPEN_BROWSER:-1}" == "0" ]]; then
        info "Dashboard: $url"
        return
    fi
    if command -v wslview &>/dev/null; then
        wslview "$url" 2>/dev/null &
    elif command -v xdg-open &>/dev/null; then
        xdg-open "$url" 2>/dev/null &
    elif [[ -x /mnt/c/Windows/System32/cmd.exe ]]; then
        /mnt/c/Windows/System32/cmd.exe /c start "$url" 2>/dev/null &
    else
        info "Open in browser: $url"
    fi
}

# ── Service starters ─────────────────────────────────────────────────

start_llm() {
    if port_busy "$VLLM_PORT"; then
        ok "vLLM already running on :${VLLM_PORT}"
        return 0
    fi
    info "Starting vLLM (${VLLM_MODEL})..."
    source "${VLLM_VENV}/bin/activate"
    nohup vllm serve "$VLLM_MODEL" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --dtype auto \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.48 \
        --enable-auto-tool-choice \
        --tool-call-parser mistral \
        > "${LOG_DIR}/vllm.log" 2>&1 &
    deactivate 2>/dev/null || true
    wait_for_port "$VLLM_PORT" "vLLM" 120
}

start_emr() {
    if port_busy "$EMR_PORT"; then
        ok "EMR API already running on :${EMR_PORT}"
        return 0
    fi
    info "Starting EMR API..."
    source "${APP_VENV}/bin/activate"
    nohup uvicorn synthetic_emr.api:app \
        --host 0.0.0.0 \
        --port "$EMR_PORT" \
        > "${LOG_DIR}/emr.log" 2>&1 &
    deactivate 2>/dev/null || true
    wait_for_port "$EMR_PORT" "EMR API" 15
}

start_dashboard() {
    if port_busy "$DASH_PORT"; then
        ok "Dashboard already running on :${DASH_PORT}"
        return 0
    fi
    info "Starting Dashboard..."
    source "${APP_VENV}/bin/activate"
    nohup uvicorn apps.dashboard.server:app \
        --host 0.0.0.0 \
        --port "$DASH_PORT" \
        > "${LOG_DIR}/dashboard.log" 2>&1 &
    deactivate 2>/dev/null || true
    wait_for_port "$DASH_PORT" "Dashboard" 15
    open_browser "$DASH_URL"
}

run_agent() {
    local scenario="${1:-scenarios/missing_scissors.json}"
    info "Running agent fixture: ${scenario}"
    source "${APP_VENV}/bin/activate"
    python -m apps.agent.run_fixture "$scenario"
}

stop_all() {
    info "Stopping all services..."
    for port in "$VLLM_PORT" "$EMR_PORT" "$DASH_PORT"; do
        kill_port "$port"
    done
    # vLLM spawns child processes (EngineCore, resource trackers) that don't hold the port
    if pgrep -f ".venv-vllm/bin/python" >/dev/null 2>&1; then
        warn "Killing remaining vLLM processes"
        pkill -f ".venv-vllm/bin/python" 2>/dev/null || true
        sleep 1
        pkill -9 -f ".venv-vllm/bin/python" 2>/dev/null || true
    fi
    ok "All services stopped"
}

show_status() {
    echo ""
    _color "1;37" "── OR Edge Agent Services ──"
    for pair in "vLLM:${VLLM_PORT}" "EMR API:${EMR_PORT}" "Dashboard:${DASH_PORT}"; do
        label="${pair%%:*}"
        port="${pair##*:}"
        if port_busy "$port"; then
            ok "$label  :${port}"
        else
            err "$label  :${port}  (not running)"
        fi
    done
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────

case "${1:-all}" in
    llm|vllm)
        start_llm
        ;;
    app)
        start_emr
        start_dashboard
        ;;
    emr)
        start_emr
        ;;
    dash|dashboard)
        kill_port "$DASH_PORT"
        start_dashboard
        ;;
    agent)
        run_agent "${2:-}"
        ;;
    stop)
        stop_all
        ;;
    status)
        show_status
        ;;
    all)
        info "Starting all services (parallel)..."
        echo ""
        # Launch all three processes (nohup) without waiting
        _launch_llm() {
            if port_busy "$VLLM_PORT"; then ok "vLLM already running on :${VLLM_PORT}"; return 0; fi
            info "Starting vLLM (${VLLM_MODEL})..."
            source "${VLLM_VENV}/bin/activate"
            nohup vllm serve "$VLLM_MODEL" \
                --host 0.0.0.0 --port "$VLLM_PORT" --dtype auto \
                --max-model-len 8192 --gpu-memory-utilization 0.48 \
                --enable-auto-tool-choice --tool-call-parser mistral \
                > "${LOG_DIR}/vllm.log" 2>&1 &
            deactivate 2>/dev/null || true
        }
        _launch_emr() {
            if port_busy "$EMR_PORT"; then ok "EMR API already running on :${EMR_PORT}"; return 0; fi
            info "Starting EMR API..."
            source "${APP_VENV}/bin/activate"
            nohup uvicorn synthetic_emr.api:app \
                --host 0.0.0.0 --port "$EMR_PORT" \
                > "${LOG_DIR}/emr.log" 2>&1 &
            deactivate 2>/dev/null || true
        }
        _launch_dash() {
            if port_busy "$DASH_PORT"; then ok "Dashboard already running on :${DASH_PORT}"; return 0; fi
            info "Starting Dashboard..."
            source "${APP_VENV}/bin/activate"
            nohup uvicorn apps.dashboard.server:app \
                --host 0.0.0.0 --port "$DASH_PORT" \
                > "${LOG_DIR}/dashboard.log" 2>&1 &
            deactivate 2>/dev/null || true
        }
        _launch_llm
        _launch_emr
        _launch_dash
        # Wait for fast services first, then vLLM
        wait_for_port "$EMR_PORT"  "EMR API"   15 &
        wait_for_port "$DASH_PORT" "Dashboard"  15 &
        wait_for_port "$VLLM_PORT" "vLLM"      120 &
        wait  # wait for all three background waits
        echo ""
        show_status
        open_browser "$DASH_URL"
        ok "Dashboard: ${DASH_URL}"
        ;;
    *)
        echo "Usage: $0 {all|llm|app|emr|dashboard|agent [scenario]|stop|status}"
        exit 1
        ;;
esac
