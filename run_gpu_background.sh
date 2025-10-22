#!/usr/bin/env bash
set -euo pipefail

# run_gpu_background.sh
# Usage: ./run_gpu_background.sh [namd_input.namd] [stdout.log] [stderr.log] [pidfile]
# Defaults: stmv.namd test.out test.err run_namd_gpu.pid

INPUT=${1:-stmv.namd}
STDOUT=${2:-test.out}
STDERR=${3:-test.err}
PIDFILE=${4:-run_namd_gpu.pid}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running from: ${PWD}"
echo "NAMD input: ${INPUT}"
echo "Stdout -> ${STDOUT}"
echo "Stderr -> ${STDERR}"
echo "PID file -> ${PIDFILE}"

# Try to load modules if the environment supports module command
if command -v module >/dev/null 2>&1; then
  echo "Loading required modules..."
  module load cuda/12.6
  module load nvmath/12.6.0
  module load openmpi/5.0.5
  module load ucx/1.18.0
  module load namd-gpu/3.0.2
else
  echo "Warning: 'module' command not found. Make sure required modules are loaded in your environment." >&2
fi

# Basic GPU availability check (optional)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU status (nvidia-smi):"
  nvidia-smi || true
fi

# Build the command
CMD=(run_namd_gpu "${INPUT}" "${STDOUT}")

echo "Starting: ${CMD[*]}"

# Start with nohup so it survives logout. Redirect output to the chosen files.
nohup "${CMD[@]}" >"${STDOUT}" 2>"${STDERR}" &

PID=$!
echo "${PID}" >"${PIDFILE}"

# Detach
disown ${PID} || true

echo "Started run_namd_gpu with PID ${PID}."
echo "Follow logs with: tail -f ${STDOUT} ${STDERR}"
echo "To stop: kill ${PID} or kill \$(cat ${PIDFILE})"

exit 0
