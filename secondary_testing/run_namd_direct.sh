#!/usr/bin/env bash
# run_namd_direct.sh
# Run NAMD without srun/MPI (direct execution with cgroup isolation)
# Use this if UCX is causing problems
#
# Usage:
#   ./run_namd_direct.sh --mig N <input.namd> <output.log>

set -euo pipefail

# --- Load modules ---
module load cuda/12.6
module load nvmath/12.6.0
module load namd-gpu/3.0.2

# --- Configuration ---
: "${TOTAL_CPUS:=72}"
: "${CORES_PER_MIG:=10}"
: "${CGROUP_BASE:=/sys/fs/cgroup/mig}"

# --- Parse arguments ---
mig_index=""
args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mig)
      [[ $# -ge 2 ]] || { echo "ERROR: --mig requires an index"; exit 2; }
      mig_index="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./run_namd_direct.sh --mig N <input.namd> <output.log>

This version runs NAMD directly without srun/MPI.
Use this if you're having UCX initialization issues.

Arguments:
  --mig N         MIG instance number (0-6)
  input.namd      NAMD configuration file
  output.log      Output log file

Examples:
  # Single run
  ./run_namd_direct.sh --mig 0 stmv.namd test0.out

  # Multiple concurrent runs
  ./run_namd_direct.sh --mig 0 stmv.namd test0.out &
  ./run_namd_direct.sh --mig 1 stmv.namd test1.out &
USAGE
      exit 0
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$mig_index" ]] || [[ ${#args[@]} -ne 2 ]]; then
  echo "Usage: $0 --mig N <input.namd> <output.log>"
  exit 1
fi

if ! [[ "$mig_index" =~ ^[0-6]$ ]]; then
  echo "ERROR: MIG index must be 0-6"
  exit 1
fi

infile="${args[0]}"
outfile="${args[1]}"

mkdir -p "$(dirname -- "$outfile")"
: > "$outfile"

# --- Locate NAMD ---
if [[ -n "${TACC_NAMD_GPU_BIN:-}" && -x "${TACC_NAMD_GPU_BIN}/namd3" ]]; then
  namd_exec="${TACC_NAMD_GPU_BIN}/namd3"
elif command -v namd3 >/dev/null 2>&1; then
  namd_exec="$(command -v namd3)"
else
  echo "ERROR: 'namd3' not found" | tee -a "$outfile"
  exit 2
fi

# --- Get MIG UUID ---
mapfile -t mig_uuids < <(nvidia-smi -L 2>/dev/null | grep "MIG" | grep -oP 'UUID: \K[^)]+')

if [[ ${#mig_uuids[@]} -eq 0 ]]; then
  echo "ERROR: No MIG devices found" | tee -a "$outfile"
  exit 2
fi

if (( mig_index >= ${#mig_uuids[@]} )); then
  echo "ERROR: MIG index $mig_index out of range" | tee -a "$outfile"
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${mig_uuids[$mig_index]}"

# --- CPU configuration ---
start_core=$((mig_index * CORES_PER_MIG))
end_core=$((start_core + CORES_PER_MIG - 1))

if (( end_core >= TOTAL_CPUS )); then
  end_core=$((TOTAL_CPUS - 1))
fi

# Charm++ settings: 1 comm thread + 9 worker threads
COMMAP="$start_core"
PEMAP="$((start_core + 1))-$end_core"
PPN=$((CORES_PER_MIG - 1))

# --- Verify cgroup ---
cgroup_path="$CGROUP_BASE/mig$mig_index"

if [[ ! -d "$cgroup_path" ]]; then
  echo "WARNING: Cgroup not found: $cgroup_path" | tee -a "$outfile"
  echo "         Will use taskset for CPU binding instead" | tee -a "$outfile"
  use_cgroup=false
else
  use_cgroup=true
  actual_cpus=$(cat "$cgroup_path/cpuset.cpus.effective" 2>/dev/null || echo "unknown")
  actual_mems=$(cat "$cgroup_path/cpuset.mems.effective" 2>/dev/null || echo "unknown")
fi

# --- Build NAMD command ---
cmd=(
  "$namd_exec"
  +ppn "$PPN"
  +pemap "$PEMAP"
  +commap "$COMMAP"
  +devices 0
  "$infile"
)

# --- Log startup ---
{
  echo "========================================"
  echo "NAMD Direct Run - MIG Instance $mig_index"
  echo "========================================"
  echo "Date: $(date '+%F %T')"
  echo "Host: $(hostname)"
  echo "Mode: Direct execution (no srun/MPI)"
  echo ""
  echo "GPU Configuration:"
  echo "  MIG Index: $mig_index"
  echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo ""
  echo "CPU Configuration:"
  echo "  Core range: $start_core-$end_core"
  echo "  PEMAP: $PEMAP"
  echo "  COMMAP: $COMMAP"
  echo "  PPN: $PPN"
  echo ""
  if $use_cgroup; then
    echo "Cgroup:"
    echo "  Path: $cgroup_path"
    echo "  CPUs: $actual_cpus"
    echo "  MEMs: $actual_mems"
  else
    echo "CPU Binding: taskset (cgroup not available)"
  fi
  echo ""
  echo "NAMD:"
  echo "  Executable: $namd_exec"
  echo "  Input: $infile"
  echo "  Output: $outfile"
  echo ""
  echo "Command: ${cmd[*]}"
  echo "========================================"
  echo ""
} | tee -a "$outfile"

# --- Launch NAMD ---
if $use_cgroup && command -v cgexec >/dev/null 2>&1; then
  # Use cgexec if available
  echo "Launching with cgexec..." | tee -a "$outfile"
  stdbuf -oL -eL cgexec -g "cpuset:$cgroup_path" "${cmd[@]}" >> "$outfile" 2>&1 &
  pid=$!
  
elif $use_cgroup; then
  # Manual cgroup assignment
  echo "Launching with manual cgroup assignment..." | tee -a "$outfile"
  "${cmd[@]}" >> "$outfile" 2>&1 &
  pid=$!
  
  sleep 0.5
  
  # Move to cgroup (try both v1 and v2 formats)
  if [[ -f "$cgroup_path/cgroup.procs" ]]; then
    echo "$pid" > "$cgroup_path/cgroup.procs" 2>/dev/null || \
      echo "WARNING: Could not move to cgroup (may lack permissions)" | tee -a "$outfile"
  elif [[ -f "$cgroup_path/tasks" ]]; then
    echo "$pid" > "$cgroup_path/tasks" 2>/dev/null || \
      echo "WARNING: Could not move to cgroup (may lack permissions)" | tee -a "$outfile"
  fi
  
else
  # Fallback: use taskset
  echo "Launching with taskset..." | tee -a "$outfile"
  taskset -c "$start_core-$end_core" "${cmd[@]}" >> "$outfile" 2>&1 &
  pid=$!
fi

# --- Report status ---
{
  echo "[$(date '+%F %T')] Launched NAMD"
  echo "  PID: $pid"
  if $use_cgroup; then
    echo "  Cgroup: $cgroup_path"
  else
    echo "  CPU binding: taskset -c $start_core-$end_core"
  fi
  echo "  Log: $outfile"
  echo ""
  echo "Monitor with:"
  echo "  tail -f $outfile"
  echo "  htop -p $pid"
  echo "  nvidia-smi dmon"
  echo ""
} | tee -a "$outfile"

echo "PID $pid"