#!/usr/bin/env bash
set -euo pipefail

SWEEP_ID="${1:?Usage: ./run_agents.sh SWEEP_ID N}"
N="${2:-4}"

mkdir -p logs

pids=()

for i in $(seq 1 "$N"); do
    echo "Starting agent $i/$N"

    wandb agent "$SWEEP_ID" \
        > "logs/agent_$i.log" 2>&1 &

    pids+=("$!")
done

trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM

wait
