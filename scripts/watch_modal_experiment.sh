#!/bin/bash
# Watch a credit-assignment Modal experiment by config nickname.
# Usage: watch_modal_experiment.sh <nickname> [cycles] [sleep_secs] [state_dir]
# Prints a summary and exits when orchestration progress changes, the app
# stops, or the polling window ends (so a supervisor gets one notification
# per state change).
set -u
NICK="${1:?usage: watch_modal_experiment.sh <nickname> [cycles] [sleep_secs] [state_dir]}"
CYCLES="${2:-9}"
SLEEP_SECS="${3:-300}"
STATE_DIR="${4:-/tmp/credit_assignment_watch}"
mkdir -p "$STATE_DIR"
VOL="credit-assignment-qwen25-$(echo "$NICK" | tr '_' '-')"
REMOTE="small_files/qwen25/$NICK/orchestration_progress.json"
STATE="$STATE_DIR/monitor_state_$NICK.txt"
CUR="$STATE_DIR/progress_current_$NICK.json"

summarize() {
  python3.12 - "$CUR" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("PARSE_FAIL", e); raise SystemExit
va = d.get("validation_accuracies", {})
tra = d.get("training_rollout_accuracies", {})
sig = f"{d.get('status')}|epoch={d.get('epoch')}|val_n={len(va)}|tra_n={len(tra)}"
print("SIGNATURE:", sig)
print("status:", d.get("status"), "epoch:", d.get("epoch"))
for k in sorted(va, key=int):
    print(f"  val[{k}]: overall={va[k][0]:.4f} deepmath={va[k][1]:.4f} math={va[k][2]:.4f} numinamath={va[k][3]:.4f}")
for k in sorted(tra, key=int):
    print(f"  train_rollout[{k}]: overall={tra[k][0]:.4f} deepmath={tra[k][1]:.4f} math={tra[k][2]:.4f} numinamath={tra[k][3]:.4f}")
tt = d.get("training_throughputs", {})
if tt:
    print("  training_throughputs:", {k: round(v,2) for k,v in tt.items()})
EOF
}

for i in $(seq 1 "$CYCLES"); do
  if modal volume get --force "$VOL" "$REMOTE" "$CUR" >/dev/null 2>&1; then
    NEW_SIG=$(summarize | head -1)
    OLD_SIG=$(cat "$STATE" 2>/dev/null || echo "none")
    if [ "$NEW_SIG" != "$OLD_SIG" ]; then
      echo "$NEW_SIG" > "$STATE"
      echo "=== PROGRESS CHANGED (cycle $i, $NICK) ==="
      summarize
      exit 0
    fi
  else
    echo "cycle $i: progress file not yet available ($NICK still building/booting)"
  fi
  APP_STATE=$(modal app list --json 2>/dev/null | python3.12 -c "
import json,sys
try: apps=json.load(sys.stdin)
except Exception: apps=[]
for a in apps:
    if '$NICK' in (a.get('description') or ''):
        print(a.get('state','')); break
else: print('absent')
" 2>/dev/null)
  case "$APP_STATE" in
    *ephemeral*|*deployed*|*running*|"") : ;;
    *)
      echo "=== APP NOT RUNNING ($NICK state: $APP_STATE) ==="
      [ -f "$CUR" ] && summarize
      exit 0
      ;;
  esac
  [ "$i" -lt "$CYCLES" ] && sleep "$SLEEP_SECS"
done
echo "=== NO CHANGE within $CYCLES cycles ($NICK) ==="
[ -f "$CUR" ] && summarize
exit 0
