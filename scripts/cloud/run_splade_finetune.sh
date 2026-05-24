#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Q3.3 — Cost-safe cloud orchestrator for the SPLADE fine-tune.
#
# What this does (and, crucially, what it GUARANTEES):
#   1. Computes max_cost = hourly_rate * hard_deadline_hours and REFUSES to
#      launch if it exceeds --max-cost (default $20). The cost ceiling + hard
#      deadline are printed prominently before any provisioning.
#   2. Provisions a DigitalOcean GPU droplet (gpu-l40sx1-48gb @ $1.57/hr) in an
#      explicit GPU-enabled region (sizes API returns empty regions[], so the
#      region MUST be passed explicitly — nyc2/tor1/atl1).
#   3. Installs a SELF-DESTRUCT safeguard ON the droplet via cloud-init: an
#      `at`/`shutdown` job + a doctl self-delete so that even if THIS
#      orchestrator loses its connection, the droplet still tears itself down at
#      the hard deadline.
#   4. Runs the trainer with an inner --max-runtime-min soft budget, periodically
#      rsync-pulls checkpoints to the ALWAYS-ON local box, and pulls the final
#      model on success.
#   5. ALWAYS destroys the droplet on success, timeout, error, or SIGINT/SIGTERM
#      via a shell `trap` (the bash equivalent of finally) — `doctl compute
#      droplet delete -f`. Teardown is idempotent and runs exactly once.
#
# THIS SCRIPT DOES NOT SPEND MONEY BY DEFAULT. Run with --dry-run (default for
# review) to print the full plan with NO doctl calls. Pass --launch to actually
# provision (requires doctl installed + a Write-scope DO token).
#
# Remaining run-time prerequisites (NOT installed here — prep only):
#   * doctl installed + authed (doctl auth init, or DIGITALOCEAN_ACCESS_TOKEN)
#   * the DO token must have WRITE scope (the .env token's scope is unconfirmed)
#   * an SSH key fingerprint registered with DO (--ssh-key-id)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIZE="gpu-l40sx1-48gb"
HOURLY_RATE="1.57"            # USD/hr for gpu-l40sx1-48gb
REGION="tor1"                 # GPU-enabled region (nyc2/tor1/atl1); sizes API regions[] is empty
IMAGE="gpu-h100x1-base"       # DO GPU base image (CUDA preinstalled); overridable
HARD_DEADLINE_HOURS="11"      # absolute kill — slightly above the 600-min soft budget
INNER_RUNTIME_MIN="600"       # trainer --max-runtime-min (soft; checkpoint + exit)
MAX_COST="20"                 # refuse to launch if rate*deadline exceeds this
CHECKPOINT_EVERY_MIN="15"
PULL_EVERY_MIN="5"            # how often the orchestrator rsync-pulls checkpoints
DROPLET_NAME="sfu-splade-ft-$(date +%Y%m%d-%H%M%S)"
SSH_KEY_ID=""                 # DO SSH key fingerprint/ID (required for --launch)
SSH_USER="root"
REMOTE_REPO="/root/sfu-library-mcp-training"
LOCAL_CKPT_DIR="${REPO_ROOT}/models/sfu-splade-v1/checkpoints"
TRAIN_DATA="data/training/hard_negatives_triplets.jsonl"
BATCH_SIZE="48"
GRAD_ACCUM="1"
EPOCHS="3"
EXTRA_TRAIN_ARGS=""
DRY_RUN="1"                   # default to dry-run; --launch flips this off
RESUME="0"                    # --resume: rsync-push local checkpoints up first

usage() {
  cat <<EOF
Usage: $0 [--dry-run | --launch] [options]

  --dry-run                 Print the full plan, NO doctl calls (default)
  --launch                  Actually provision + run + always destroy
  --resume                  Push local checkpoints up and resume training
  --size NAME               DO size slug (default: ${SIZE})
  --region SLUG             GPU-enabled region (default: ${REGION})
  --image SLUG              Droplet image (default: ${IMAGE})
  --hourly-rate USD         Size hourly rate (default: ${HOURLY_RATE})
  --hard-deadline-hours N   Absolute kill deadline (default: ${HARD_DEADLINE_HOURS})
  --inner-runtime-min N     Trainer soft budget (default: ${INNER_RUNTIME_MIN})
  --max-cost USD            Refuse to launch above this (default: ${MAX_COST})
  --ssh-key-id ID           DO SSH key fingerprint/ID (required for --launch)
  --batch-size N            (default: ${BATCH_SIZE})
  --grad-accum N            (default: ${GRAD_ACCUM})
  --epochs N                (default: ${EPOCHS})
  --extra-train-args "..."  Passed verbatim to finetune_splade.py
  -h, --help                This help
EOF
}

# ── Arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN="1"; shift ;;
    --launch) DRY_RUN="0"; shift ;;
    --resume) RESUME="1"; shift ;;
    --size) SIZE="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --hourly-rate) HOURLY_RATE="$2"; shift 2 ;;
    --hard-deadline-hours) HARD_DEADLINE_HOURS="$2"; shift 2 ;;
    --inner-runtime-min) INNER_RUNTIME_MIN="$2"; shift 2 ;;
    --max-cost) MAX_COST="$2"; shift 2 ;;
    --ssh-key-id) SSH_KEY_ID="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --extra-train-args) EXTRA_TRAIN_ARGS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# ── Load DO token from .env (matches house _load_dotenv behaviour) ───────────
if [[ -f "${REPO_ROOT}/.env" ]]; then
  # shellcheck disable=SC2046  # we want word-splitting of the grep output here
  while IFS='=' read -r k v; do
    [[ -z "$k" || "$k" == \#* ]] && continue
    if [[ "$k" == "DIGITALOCEAN_ACCESS_TOKEN" ]]; then
      v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"
      export DIGITALOCEAN_ACCESS_TOKEN="$v"
    fi
  done < "${REPO_ROOT}/.env"
fi

# ── Cost ceiling math (this is the money guard) ──────────────────────────────
# max_cost = hourly_rate * hard_deadline_hours. Refuse if > --max-cost.
MAX_RUN_COST="$(awk -v r="${HOURLY_RATE}" -v h="${HARD_DEADLINE_HOURS}" \
  'BEGIN { printf "%.2f", r * h }')"
OVER_CEILING="$(awk -v c="${MAX_RUN_COST}" -v m="${MAX_COST}" \
  'BEGIN { print (c > m) ? 1 : 0 }')"

DEADLINE_EPOCH="$(awk -v h="${HARD_DEADLINE_HOURS}" 'BEGIN { printf "%d", systime() + h*3600 }' 2>/dev/null \
  || date -u -d "+${HARD_DEADLINE_HOURS} hours" +%s)"
DEADLINE_HUMAN="$(date -u -d "@${DEADLINE_EPOCH}" +'%Y-%m-%d %H:%M UTC' 2>/dev/null || echo "+${HARD_DEADLINE_HOURS}h")"

print_plan() {
  cat <<EOF
══════════════════════════════════════════════════════════════════════
  SPLADE FINE-TUNE — CLOUD COST-SAFETY PLAN
══════════════════════════════════════════════════════════════════════
  mode                : $([[ "$DRY_RUN" == "1" ]] && echo "DRY-RUN (no doctl, no spend)" || echo "LAUNCH (will provision + bill)")
  droplet name        : ${DROPLET_NAME}
  size                : ${SIZE}   (\$${HOURLY_RATE}/hr)
  region              : ${REGION}   (explicit — sizes API regions[] is empty for GPU)
  image               : ${IMAGE}

  ── HARD MONEY CAPS ──────────────────────────────────────────────────
  hourly rate         : \$${HOURLY_RATE}/hr
  hard deadline       : ${HARD_DEADLINE_HOURS} h   (absolute kill @ ${DEADLINE_HUMAN})
  >>> MAX COST CEILING: \$${MAX_RUN_COST}  (= ${HOURLY_RATE} * ${HARD_DEADLINE_HOURS})
  --max-cost limit    : \$${MAX_COST}
  inner soft budget   : ${INNER_RUNTIME_MIN} min  (trainer checkpoints + exits)

  ── TEARDOWN GUARANTEES ──────────────────────────────────────────────
  orchestrator trap   : doctl compute droplet delete -f  on EXIT/ERR/INT/TERM
  on-droplet failsafe : cloud-init schedules 'shutdown -h' + doctl self-delete
                        at +${HARD_DEADLINE_HOURS}h — fires even if this box dies

  ── CHECKPOINT ROUND-TRIP ────────────────────────────────────────────
  checkpoint cadence  : every ${CHECKPOINT_EVERY_MIN} min (on droplet)
  pull cadence        : rsync pull every ${PULL_EVERY_MIN} min -> ${LOCAL_CKPT_DIR}
  resume push         : $([[ "$RESUME" == "1" ]] && echo "yes (rsync local -> droplet, --resume-from)" || echo "no")
  verify-on-arrival   : checkpoint_manifest.json sha256+size re-checked locally

  ── TRAINING ─────────────────────────────────────────────────────────
  train data          : ${TRAIN_DATA}
  batch/accum/epochs  : ${BATCH_SIZE} / ${GRAD_ACCUM} / ${EPOCHS}
  extra train args    : ${EXTRA_TRAIN_ARGS:-(none)}
══════════════════════════════════════════════════════════════════════
EOF
}

print_plan

# ── Refuse to launch over the cost ceiling ───────────────────────────────────
if [[ "${OVER_CEILING}" == "1" ]]; then
  echo "REFUSING: max cost \$${MAX_RUN_COST} exceeds --max-cost \$${MAX_COST}." >&2
  echo "Lower --hard-deadline-hours or raise --max-cost deliberately." >&2
  exit 3
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY-RUN: plan validated, cost ceiling within limit. No doctl calls made."
  log "To actually run (once doctl + a Write-scope token + --ssh-key-id are set):"
  echo "    $0 --launch --ssh-key-id <DO_SSH_KEY_FINGERPRINT>"
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Everything below this line only runs under --launch.
# ─────────────────────────────────────────────────────────────────────────────

command -v doctl >/dev/null 2>&1 || { echo "doctl not installed. Install it first (prep step)." >&2; exit 4; }
command -v rsync >/dev/null 2>&1 || { echo "rsync not installed." >&2; exit 4; }
[[ -n "${DIGITALOCEAN_ACCESS_TOKEN:-}" ]] || { echo "DIGITALOCEAN_ACCESS_TOKEN not set (.env)." >&2; exit 4; }
[[ -n "${SSH_KEY_ID}" ]] || { echo "--ssh-key-id is required for --launch." >&2; exit 4; }

DROPLET_ID=""
TORN_DOWN="0"

teardown() {
  # The money guard. Runs on EXIT (any reason), idempotently, exactly once.
  local rc=$?
  if [[ "${TORN_DOWN}" == "1" ]]; then return; fi
  TORN_DOWN="1"
  if [[ -n "${DROPLET_ID}" ]]; then
    log "TEARDOWN: destroying droplet ${DROPLET_ID} (exit code ${rc})..."
    # Retry a few times — a destroy must not be skipped on a transient API blip.
    for attempt in 1 2 3 4 5; do
      if doctl compute droplet delete -f "${DROPLET_ID}" 2>/dev/null; then
        log "TEARDOWN: droplet ${DROPLET_ID} destroyed."
        break
      fi
      log "TEARDOWN: delete attempt ${attempt} failed; retrying in 10s..."
      sleep 10
    done
    # Final confirmation; warn loudly if it somehow still exists.
    if doctl compute droplet get "${DROPLET_ID}" >/dev/null 2>&1; then
      echo "!!! WARNING: droplet ${DROPLET_ID} may still exist — VERIFY MANUALLY:" >&2
      echo "    doctl compute droplet delete -f ${DROPLET_ID}" >&2
    fi
  else
    log "TEARDOWN: no droplet was created; nothing to destroy."
  fi
  exit "${rc}"
}
trap teardown EXIT INT TERM

# ── cloud-init: on-droplet self-destruct safeguard ───────────────────────────
# Even if this orchestrator dies, the droplet shuts itself down (and tries to
# self-delete via doctl) at the hard deadline. shutdown alone stops billing for
# compute on most plans, but we ALSO self-delete to release the resource fully.
CLOUD_INIT="$(mktemp)"
cat > "${CLOUD_INIT}" <<EOF
#cloud-config
write_files:
  - path: /root/self_destruct.sh
    permissions: '0755'
    content: |
      #!/usr/bin/env bash
      # On-droplet failsafe: hard power-off at the deadline, then self-delete.
      export DIGITALOCEAN_ACCESS_TOKEN="${DIGITALOCEAN_ACCESS_TOKEN}"
      ID=\$(curl -s http://169.254.169.254/metadata/v1/id || true)
      if command -v doctl >/dev/null 2>&1 && [ -n "\$ID" ]; then
        doctl compute droplet delete -f "\$ID" || true
      fi
      shutdown -h now
runcmd:
  - apt-get update -y || true
  - apt-get install -y at rsync || true
  - systemctl enable --now atd || true
  # Schedule the absolute self-destruct ${HARD_DEADLINE_HOURS}h from boot.
  - echo "/root/self_destruct.sh" | at now + ${HARD_DEADLINE_HOURS} hours || \
      ( sleep ${HARD_DEADLINE_HOURS}h && /root/self_destruct.sh ) &
EOF

log "Creating droplet ${DROPLET_NAME} (${SIZE} @ ${REGION})..."
DROPLET_ID="$(doctl compute droplet create "${DROPLET_NAME}" \
  --size "${SIZE}" --region "${REGION}" --image "${IMAGE}" \
  --ssh-keys "${SSH_KEY_ID}" \
  --user-data-file "${CLOUD_INIT}" \
  --wait --no-header --format ID)"
rm -f "${CLOUD_INIT}"
log "Droplet created: ID=${DROPLET_ID}"

DROPLET_IP="$(doctl compute droplet get "${DROPLET_ID}" --no-header --format PublicIPv4)"
log "Droplet IP: ${DROPLET_IP}"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
ssh_do() { ssh ${SSH_OPTS} "${SSH_USER}@${DROPLET_IP}" "$@"; }

# Wait for SSH.
log "Waiting for SSH..."
for _ in $(seq 1 60); do
  if ssh_do true 2>/dev/null; then break; fi
  sleep 10
done

mkdir -p "${LOCAL_CKPT_DIR}"

# ── Resume: push the latest local checkpoint up before starting ──────────────
RESUME_ARG=""
if [[ "${RESUME}" == "1" && -d "${LOCAL_CKPT_DIR}/latest_ckpt" ]]; then
  log "RESUME: pushing local checkpoint up to droplet..."
  ssh_do "mkdir -p ${REMOTE_REPO}/models/sfu-splade-v1/checkpoints"
  rsync -az -e "ssh ${SSH_OPTS}" "${LOCAL_CKPT_DIR}/" \
    "${SSH_USER}@${DROPLET_IP}:${REMOTE_REPO}/models/sfu-splade-v1/checkpoints/"
  RESUME_ARG="--resume-from ${REMOTE_REPO}/models/sfu-splade-v1/checkpoints/latest_ckpt"
fi

# ── Background rsync puller (verify-on-arrival happens locally on resume) ─────
PULL_PID=""
start_puller() {
  (
    while true; do
      rsync -az -e "ssh ${SSH_OPTS}" \
        "${SSH_USER}@${DROPLET_IP}:${REMOTE_REPO}/models/sfu-splade-v1/checkpoints/" \
        "${LOCAL_CKPT_DIR}/" 2>/dev/null || true
      sleep "$(( PULL_EVERY_MIN * 60 ))"
    done
  ) &
  PULL_PID=$!
  log "Checkpoint puller started (PID ${PULL_PID}, every ${PULL_EVERY_MIN} min)."
}
stop_puller() { [[ -n "${PULL_PID}" ]] && kill "${PULL_PID}" 2>/dev/null || true; }
start_puller

# ── Run the trainer under timeout as a belt-and-suspenders hard cap ──────────
HARD_DEADLINE_SECS="$(awk -v h="${HARD_DEADLINE_HOURS}" 'BEGIN { printf "%d", h*3600 }')"
log "Launching trainer (inner soft budget ${INNER_RUNTIME_MIN} min, hard timeout ${HARD_DEADLINE_HOURS}h)..."
set +e
timeout "${HARD_DEADLINE_SECS}" ssh ${SSH_OPTS} "${SSH_USER}@${DROPLET_IP}" bash -lc "'
  set -e
  cd ${REMOTE_REPO}
  export HF_HUB_OFFLINE=1
  .venv/bin/python3 scripts/finetune_splade.py \
    --train-data ${TRAIN_DATA} \
    --output models/sfu-splade-v1 \
    --batch-size ${BATCH_SIZE} --grad-accum ${GRAD_ACCUM} --epochs ${EPOCHS} \
    --max-runtime-min ${INNER_RUNTIME_MIN} \
    --checkpoint-every-min ${CHECKPOINT_EVERY_MIN} \
    ${RESUME_ARG} ${EXTRA_TRAIN_ARGS}
'"
TRAIN_RC=$?
set -e
log "Trainer exited with code ${TRAIN_RC}."

stop_puller

# ── Final pull (checkpoints + final model) on success ────────────────────────
log "Final rsync pull of checkpoints + model..."
rsync -az -e "ssh ${SSH_OPTS}" \
  "${SSH_USER}@${DROPLET_IP}:${REMOTE_REPO}/models/sfu-splade-v1/" \
  "${REPO_ROOT}/models/sfu-splade-v1/" 2>/dev/null || true

# Verify-on-arrival: refuse to trust a half-pulled latest checkpoint.
if [[ -d "${LOCAL_CKPT_DIR}/latest_ckpt" ]]; then
  log "Verifying pulled checkpoint integrity..."
  "${REPO_ROOT}/.venv/bin/python3" - "${LOCAL_CKPT_DIR}/latest_ckpt" <<'PY' || \
    log "WARNING: pulled checkpoint failed verification (may be mid-write)."
import sys
sys.path.insert(0, "scripts")
from finetune_splade import verify_checkpoint
from pathlib import Path
ok = verify_checkpoint(Path(sys.argv[1]))
sys.exit(0 if ok else 1)
PY
fi

log "Done. Teardown trap will now destroy the droplet."
exit "${TRAIN_RC}"
