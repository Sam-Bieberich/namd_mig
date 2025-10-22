#!/usr/bin/env bash
# run_namd_local.sh
#
# Run NAMD3-GPU inside an existing interactive SLURM allocation,
# in the background, with CPU pinning and GPU selection.
# Usage: ./run_namd_local.sh <input.namd> <output.log>

set -euo pipefail

# --- Modules (match your site template) ---
module load cuda/12.6
module load nvmath/12.6.0
module load openmpi/5.0.5
module load ucx/1.18.0
module load namd-gpu/3.0.2

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <input.namd> <output.log>"
  exit 1
fi

infile="$1"
outfile="$2"

# --- Binding defaults (can be overridden via env) ---
# Reserve core 0 for comm thread; pin workers to 1-71 (example: 72 logical CPUs)
: "${PEMAP:=1-71}"      # worker thread core map
: "${COMMAP:=0}"        # comm thread core
: "${PPN:=71}"          # number of worker threads
: "${DEVICES:=0}"       # which GPU(s) to use, NAMD format, e.g., "0" or "0,1"

# --- Choose launcher: use srun inside an allocation, else run binary directly ---
use_srun=false
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  use_srun=true
fi

# --- Locate namd3 ---
if [[ -n "${TACC_NAMD_GPU_BIN:-}" && -x "${TACC_NAMD_GPU_BIN}/namd3" ]]; then
  namd_exec="${TACC_NAMD_GPU_BIN}/namd3"
elif command -v namd3 >/dev/null 2>&1 ; then
  namd_exec="$(command -v namd3)"
else
  echo "ERROR: Cannot find 'namd3'. Ensure the namd-gpu module is loaded."
  exit 2
fi

# --- Build command ---
cmd=( "$namd_exec" +ppn "$PPN" +pemap "$PEMAP" +commap "$COMMAP" +devices "$DEVICES" "$infile" )

# --- Launch in background with unbuffered output redirected to file ---
# We prefer srun inside allocation to get proper cgroup binding.
if $use_srun; then
  # --mpi=pmi2 matches your site template; change to --mpi=pmix if your site requires.
  # --unbuffered helps flush output promptly to the log.
  srun --mpi=pmi2 --unbuffered --ntasks=1 "${cmd[@]}" >> "$outfile" 2>&1 &
else
  # Fallback if for some reason you're not in an allocation (rare in your case).
  "${cmd[@]}" >> "$outfile" 2>&1 &
fi

pid=$!
echo "Started NAMD (PID $pid). Logging to: $outfile"
echo
echo "Useful commands:"
echo "  tail -f $outfile"
echo "  jobs -l"
echo "  disown -h %<job>   # prevent SIGHUP if you close the shell"
echo "  kill $pid          # stop the run (or 'kill %<job>')"
``

