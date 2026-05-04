#!/usr/bin/env bash
# Training deployment script for SFU embedding model
#
# Commands:
#   ./scripts/run_training.sh setup      Install training dependencies
#   ./scripts/run_training.sh generate   Generate training data (run once)
#   ./scripts/run_training.sh start      Start fresh training in background
#   ./scripts/run_training.sh resume     Resume from latest checkpoint
#   ./scripts/run_training.sh stop       Gracefully stop (saves checkpoint first)
#   ./scripts/run_training.sh status     Show current training progress
#   ./scripts/run_training.sh logs       Tail the live training log
#   ./scripts/run_training.sh attach     Attach to a running training session (tmux)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PROJECT_DIR}/.venv/bin/python3"
PIP="$PYTHON -m pip"
TMUX_SESSION="sfu-training"
LOG_FILE="${PROJECT_DIR}/logs/training.log"
PID_FILE="${PROJECT_DIR}/logs/training.pid"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_embedding_model.py"
GEN_SCRIPT="${SCRIPT_DIR}/generate_sfu_training_data.py"

# ── Default training hyperparameters ─────────────────────────────────────────
DATA_TRAIN="${PROJECT_DIR}/data/splits/train.jsonl"
DATA_VAL="${PROJECT_DIR}/data/splits/val.jsonl"
OUTPUT_DIR="${PROJECT_DIR}/models/sfu-academic-embed-v3"
EPOCHS=6
BATCH_SIZE=32
GRAD_ACCUM=1
LR=2e-5
MAX_SEQ_LENGTH=512
SAVE_STEPS=100
EVAL_STEPS=500
FP16_FLAG=""

# STATUS_FILE is derived from OUTPUT_DIR so changing the model name updates both.
STATUS_FILE="${OUTPUT_DIR}/checkpoints/training_status.json"

# Detect GPU and enable fp16 automatically
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null 2>&1; then
    FP16_FLAG="--fp16"
fi

# ─────────────────────────────────────────────────────────────────────────────
_require_python() {
    if [[ ! -x "$PYTHON" ]]; then
        echo "ERROR: Python venv not found at $PYTHON"
        echo "       Run: ./scripts/run_training.sh setup"
        exit 1
    fi
}

_check_torch() {
    "$PYTHON" -c "import torch" 2>/dev/null
}

_is_running() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

# ─────────────────────────────────────────────────────────────────────────────
cmd_setup() {
    echo "=== Setting up training environment ==="
    _require_python

    if _check_torch; then
        echo "  torch already installed"
    else
        echo "  Installing PyTorch (CUDA 12.4 build)..."
        $PIP install --quiet \
            torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu124
    fi

    echo "  Installing training dependencies..."
    $PIP install --quiet \
        sentence-transformers \
        transformers \
        accelerate \
        tqdm

    echo "  Verifying..."
    "$PYTHON" -c "
import torch, sentence_transformers
print(f'  torch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  sentence-transformers {sentence_transformers.__version__}')
print('Setup complete.')
"
}

cmd_generate() {
    _require_python
    if ! _check_torch; then
        echo "ERROR: torch not installed. Run: ./scripts/run_training.sh setup"
        exit 1
    fi

    SPLITS_DIR="${PROJECT_DIR}/data/splits"
    if [[ -f "${SPLITS_DIR}/train.jsonl" && -f "${SPLITS_DIR}/val.jsonl" ]]; then
        TRAIN_COUNT=$(wc -l < "${SPLITS_DIR}/train.jsonl")
        VAL_COUNT=$(wc -l < "${SPLITS_DIR}/val.jsonl")
        echo "Training data already exists: ${TRAIN_COUNT} train / ${VAL_COUNT} val samples"
        echo "  To regenerate, delete ${SPLITS_DIR}/ and re-run this command."
        return 0
    fi

    echo "=== Generating SFU training data ==="
    cd "$PROJECT_DIR"
    "$PYTHON" "$GEN_SCRIPT" --resume
    echo "Splitting into train/val..."
    mkdir -p "$SPLITS_DIR"
    "$PYTHON" - <<'EOF'
import json, random
from pathlib import Path

src = Path("data/sfu_training_triplets.jsonl")
splits = Path("data/splits")
splits.mkdir(exist_ok=True)

lines = [l for l in src.read_text().splitlines() if l.strip()]
random.seed(42)
random.shuffle(lines)
cut = int(len(lines) * 0.9)
(splits / "train.jsonl").write_text("\n".join(lines[:cut]) + "\n")
(splits / "val.jsonl").write_text("\n".join(lines[cut:]) + "\n")
print(f"Split: {cut} train / {len(lines)-cut} val")
EOF
    echo "Data generation complete."
}

_start_training() {
    local extra_args="$*"
    _require_python
    if ! _check_torch; then
        echo "torch not installed. Running setup first..."
        cmd_setup
    fi

    if [[ ! -f "$DATA_TRAIN" ]]; then
        echo "Training data not found at $DATA_TRAIN"
        echo "Run: ./scripts/run_training.sh generate"
        exit 1
    fi

    if _is_running; then
        echo "Training is already running (PID $(cat "$PID_FILE"))"
        echo "  Use: ./scripts/run_training.sh status"
        echo "  Use: ./scripts/run_training.sh stop    to stop it first"
        exit 1
    fi

    mkdir -p "${PROJECT_DIR}/logs"

    VAL_ARG=""
    [[ -f "$DATA_VAL" ]] && VAL_ARG="--val-data $DATA_VAL"

    local CMD="$PYTHON $TRAIN_SCRIPT \
        --data $DATA_TRAIN \
        $VAL_ARG \
        --output $OUTPUT_DIR \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --gradient-accumulation $GRAD_ACCUM \
        --learning-rate $LR \
        --max-seq-length $MAX_SEQ_LENGTH \
        --save-steps $SAVE_STEPS \
        --eval-steps $EVAL_STEPS \
        $FP16_FLAG \
        $extra_args"

    echo "=== Starting training ==="
    echo "  Log:    $LOG_FILE"
    echo "  Status: $STATUS_FILE"
    echo ""

    # Use tmux if available (allows attach); otherwise nohup
    if command -v tmux &>/dev/null; then
        tmux new-session -d -s "$TMUX_SESSION" \
            "cd '$PROJECT_DIR' && $CMD 2>&1 | tee '$LOG_FILE'; echo 'Training ended'; read"
        echo "Started in tmux session '$TMUX_SESSION'"
        echo "  Attach:    tmux attach -t $TMUX_SESSION"
        echo "  Detach:    Ctrl-B, D"
        # Grab PID of the python process inside tmux
        sleep 1
        tmux list-panes -t "$TMUX_SESSION" -F '#{pane_pid}' | head -1 > "$PID_FILE" || true
    else
        nohup bash -c "cd '$PROJECT_DIR' && $CMD" >> "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        echo "Started in background (PID $(cat "$PID_FILE"))"
        echo "  Logs:   tail -f $LOG_FILE"
    fi

    echo "  Stop:   ./scripts/run_training.sh stop"
    echo "  Status: ./scripts/run_training.sh status"
}

cmd_start() {
    echo "Starting fresh training run..."
    _start_training
}

cmd_resume() {
    echo "Resuming training from latest checkpoint..."
    _start_training "--resume"
}

cmd_stop() {
    if ! _is_running; then
        echo "No training process is running."
        [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid="$(cat "$PID_FILE")"
    echo "Sending SIGTERM to PID $pid (graceful stop — will save checkpoint)..."
    kill -TERM "$pid" 2>/dev/null || true

    # Wait up to 120 seconds for graceful exit
    local waited=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 2
        waited=$((waited + 2))
        if [[ $waited -ge 120 ]]; then
            echo "Still running after 120s — sending SIGKILL..."
            kill -KILL "$pid" 2>/dev/null || true
            break
        fi
        echo "  Waiting for checkpoint save... (${waited}s)"
    done

    rm -f "$PID_FILE"
    # Kill tmux session if it exists
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    echo "Training stopped."
    cmd_status 2>/dev/null || true
}

cmd_status() {
    echo "=== Training Status ==="

    if _is_running; then
        echo "  State:   RUNNING (PID $(cat "$PID_FILE"))"
    else
        echo "  State:   STOPPED"
    fi

    if [[ -f "$STATUS_FILE" ]]; then
        python3 - "$STATUS_FILE" <<'EOF'
import json, sys
s = json.loads(open(sys.argv[1]).read())
print(f"  Phase:   {s.get('phase','?')}")
print(f"  Epoch:   {s.get('epoch',0)+1} / {s.get('total_epochs','?')}")
print(f"  Step:    {s.get('global_step',0)} / {s.get('total_steps','?')}  ({s.get('progress_pct',0):.1f}%)")
if s.get('loss') is not None:
    print(f"  Loss:    {s['loss']:.6f}")
print(f"  BestVal: {s.get('best_val_score',0):.4f}")
print(f"  Updated: {s.get('timestamp','?')}")
EOF
    else
        echo "  No status file yet — training hasn't started or checkpoint dir not created."
    fi

    # Show last checkpoint
    CKPT_STATE="${OUTPUT_DIR}/checkpoints/training_state.json"
    if [[ -f "$CKPT_STATE" ]]; then
        echo ""
        echo "  Last checkpoint:"
        python3 - "$CKPT_STATE" <<'EOF'
import json, sys
s = json.loads(open(sys.argv[1]).read())
print(f"    Label:  {s.get('checkpoint_label','?')}")
print(f"    Step:   {s.get('global_step',0)}")
print(f"    Epoch:  {s.get('epoch',0)}")
print(f"    Time:   {s.get('timestamp','?')}")
EOF
    fi
}

cmd_logs() {
    if [[ ! -f "$LOG_FILE" ]]; then
        echo "No log file found at $LOG_FILE"
        echo "Training has not been started yet."
        exit 1
    fi
    echo "=== Tailing $LOG_FILE (Ctrl-C to stop) ==="
    tail -f "$LOG_FILE"
}

cmd_attach() {
    if command -v tmux &>/dev/null && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "Attaching to tmux session '$TMUX_SESSION'  (Ctrl-B, D to detach)"
        tmux attach -t "$TMUX_SESSION"
    else
        echo "No active tmux session — falling back to log tail:"
        cmd_logs
    fi
}

cmd_help() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# //'
    echo ""
    echo "GPU detected: $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
    echo "FP16:         ${FP16_FLAG:-(disabled)}"
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
case "${1:-help}" in
    setup)    cmd_setup ;;
    generate) cmd_generate ;;
    start)    cmd_start ;;
    resume)   cmd_resume ;;
    stop)     cmd_stop ;;
    status)   cmd_status ;;
    logs)     cmd_logs ;;
    attach)   cmd_attach ;;
    help|--help|-h) cmd_help ;;
    *) echo "Unknown command: $1"; cmd_help; exit 1 ;;
esac
