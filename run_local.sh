#!/bin/bash

# Local run script emulating the Slurm runjob
# Usage: bash run_local.sh

set -euo pipefail

LOG_OUT=gpu_test.out
LOG_ERR=gpu_test.err

# Redirect stdout/stderr to files while also showing on terminal
# We'll use exec to redirect file descriptors
exec 1> >(tee -a "$LOG_OUT")
exec 2> >(tee -a "$LOG_ERR" >&2)

echo "Starting local run at $(date)"

# Try to load modules if `module` command exists; otherwise warn
if command -v module >/dev/null 2>&1; then
  echo "Loading modules..."
  module load cuda/12.6 || true
  module load nvmath/12.6.0 || true
  module load openmpi/5.0.5 || true
  module load ucx/1.18.0 || true
  module load namd-gpu/3.0.2 || true
else
  echo "module command not found; please ensure required software is available in PATH or load modules manually"
fi

# Show environment for debugging
echo "Loaded modules:" 
module list 2>&1 || true

# Print GPU status (if available)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi output:" 
  nvidia-smi
fi

# Run the command (use full paths if needed)
CMD=(run_namd_gpu stmv.namd test.out)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
EXIT_CODE=$?

echo "Command exited with code: $EXIT_CODE"

echo "Finished local run at $(date)"
exit $EXIT_CODE
