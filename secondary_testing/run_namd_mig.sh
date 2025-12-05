#!/usr/bin/env bash
# run_namd_mig.sh
# Run NAMD3-GPU with MIG GPU isolation and CPU/memory cgroup partitioning
#
# Usage:
#   ./run_namd_mig.sh --mig N <input.namd> <output.log>
#
# Examples:
#   ./run_namd_mig.sh --mig 0 stmv.namd test0.out
#   ./run_namd_mig.sh --mig 2 stmv.namd test2.out &  # background
#
# Multiple concurrent runs:
#   SRUN_MPI=pmix ./run_namd_mig.sh --mig 0 stmv.namd test0.out &
#   SRUN_MPI=pmix ./run_namd_mig.sh --mig 1 stmv.namd test1.out &

set -euo pipefail

# --- Load required modules ---
module load cuda/12.6
module load nvmath/12.6.0
module load openmpi/5.0.5
module load ucx/1.18.0
module load namd-gpu/3.0.2

# --- Configuration defaults ---
: "${TOTAL_CPUS:=72}"
: "${MIG_SLOTS:=7}"
: "${CORES_PER_MIG:=10}"
: "${CGROUP_BASE:=/sys/fs/cgroup/mig}"

# UCX settings for single-node operation with MIG
export UCX_TLS="${UCX_TLS:-self,sm,cuda_copy,cuda_ipc}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export UCX_WARN_UNUSED_ENV_VARS="${UCX_WARN_UNUSED_ENV_VARS:-n}"

# Additional UCX settings that help with MIG
export UCX_IB_GPU_DIRECT_RDMA="${UCX_IB_GPU_DIRECT_RDMA:-no}"
export UCX_RNDV_SCHEME="${UCX_RNDV_SCHEME:-put_zcopy}"

# NAMD/Charm++ defaults (will be auto-configured per MIG)
: "${DEVICES:=0}"  # Single MIG device visible = device 0
: "${SRUN_MPI:=pmix}"  # pmix|pmi2 (OpenMPI 5 works well with pmix)
: "${SRUN_OVERLAP:=1}"  # Allow concurrent srun steps

# --- Parse command-line arguments ---
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
Usage: ./run_namd_mig.sh --mig N <input.namd> <output.log>

Arguments:
  --mig N         MIG instance number (0-6)
  input.namd      NAMD configuration file
  output.log      Output log file

Environment variables:
  CORES_PER_MIG   Cores per MIG (default: 10)
  SRUN_MPI        MPI type for srun (default: pmix)
  SRUN_OVERLAP    Allow concurrent runs (default: 1)

Examples:
  # Single run
  ./run_namd_mig.sh --mig 0 stmv.namd test0.out

  # Multiple concurrent runs
  ./run_namd_mig.sh --mig 0 stmv.namd test0.out &
  ./run_namd_mig.sh --mig 1 stmv.namd test1.out &
  ./run_namd_mig.sh --mig 2 stmv.namd test2.out &
USAGE
      exit 0
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

# Validate arguments
if [[ -z "$mig_index" ]] || [[ ${#args[@]} -ne 2 ]]; then
  echo "Usage: $0 --mig N <input.namd> <output.log>"
  exit 1
fi

if ! [[ "$mig_index" =~ ^[0-6]$ ]]; then
  echo "ERROR: MIG index must be 0-6, got: $mig_index"
  exit 1
fi

infile="${args[0]}"
outfile="${args[1]}"

# Create output directory and initialize log file
mkdir -p "$(dirname -- "$outfile")"
: > "$outfile"

# --- Locate NAMD executable ---
if [[ -n "${TACC_NAMD_GPU_BIN:-}" && -x "${TACC_NAMD_GPU_BIN}/namd3" ]]; then
  namd_exec="${TACC_NAMD_GPU_BIN}/namd3"
elif command -v namd3 >/dev/null 2>&1; then
  namd_exec="$(command -v namd3)"
else
  echo "ERROR: 'namd3' not found. Ensure namd-gpu module is loaded." | tee -a "$outfile"
  exit 2
fi

# --- Get MIG UUID for this index ---
mapfile -t mig_uuids < <(nvidia-smi -L 2>/dev/null | grep "MIG" | grep -oP 'UUID: \K[^)]+')

if [[ ${#mig_uuids[@]} -eq 0 ]]; then
  echo "ERROR: No MIG devices found. Run MIG setup first." | tee -a "$outfile"
  exit 2
fi

if (( mig_index >= ${#mig_uuids[@]} )); then
  echo "ERROR: MIG index $mig_index out of range (0-$((${#mig_uuids[@]}-1)))" | tee -a "$outfile"
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${mig_uuids[$mig_index]}"

# --- Calculate CPU pinning for this MIG instance ---
start_core=$((mig_index * CORES_PER_MIG))
end_core=$((start_core + CORES_PER_MIG - 1))

# Adjust if we exceed total CPUs
if (( end_core >= TOTAL_CPUS )); then
  end_core=$((TOTAL_CPUS - 1))
fi

# NAMD Charm++ thread mapping
# Reserve first core for comm thread, rest for workers
COMMAP="$start_core"
PEMAP="$((start_core + 1))-$end_core"
PPN=$((CORES_PER_MIG - 1))  # 9 worker threads (10 cores - 1 for comm)

# --- Verify cgroup exists ---
cgroup_path="$CGROUP_BASE/mig$mig_index"

if [[ ! -d "$cgroup_path" ]]; then
  echo "ERROR: Cgroup not found: $cgroup_path" | tee -a "$outfile"
  echo "       Run setup_mig_cgroups.sh first" | tee -a "$outfile"
  exit 2
fi

# Verify cgroup configuration
if [[ -f "$cgroup_path/cpuset.cpus.effective" ]]; then
  actual_cpus=$(cat "$cgroup_path/cpuset.cpus.effective")
  actual_mems=$(cat "$cgroup_path/cpuset.mems.effective")
else
  echo "WARNING: Cannot verify cgroup configuration" | tee -a "$outfile"
  actual_cpus="unknown"
  actual_mems="unknown"
fi

# --- Check SLURM allocation ---
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: Not inside a SLURM allocation" | tee -a "$outfile"
  echo "       Use: salloc -N 1 -n 1 -p <partition> -t <time>" | tee -a "$outfile"
  exit 2
fi

# --- Build NAMD command ---
cmd=(
  "$namd_exec"
  +ppn "$PPN"
  +pemap "$PEMAP"
  +commap "$COMMAP"
  +devices "$DEVICES"
  "$infile"
)

# --- Build srun arguments ---
srun_args=(
  --ntasks=1
  --unbuffered
  --cpu-bind=none
  --gpu-bind=none
  --cpus-per-task="$CORES_PER_MIG"
)

[[ "$SRUN_OVERLAP" == "1" ]] && srun_args+=(--overlap)

case "$SRUN_MPI" in
  pmix) srun_args+=(--mpi=pmix) ;;
  pmi2) srun_args+=(--mpi=pmi2) ;;
  none) ;;
  *) srun_args+=(--mpi=pmix) ;;
esac

export SLURM_CPU_BIND=none

# --- Log startup information ---
{
  echo "========================================"
  echo "NAMD Run - MIG Instance $mig_index"
  echo "========================================"
  echo "Date: $(date '+%F %T')"
  echo "Host: $(hostname)"
  echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
  echo ""
  echo "GPU Configuration:"
  echo "  MIG Index: $mig_index"
  echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo ""
  echo "CPU Configuration:"
  echo "  Core range: $start_core-$end_core ($CORES_PER_MIG cores)"
  echo "  PEMAP (workers): $PEMAP"
  echo "  COMMAP (comm): $COMMAP"
  echo "  PPN (threads): $PPN"
  echo ""
  echo "Cgroup:"
  echo "  Path: $cgroup_path"
  echo "  CPUs (effective): $actual_cpus"
  echo "  MEMs (effective): $actual_mems"
  echo ""
  echo "NAMD:"
  echo "  Executable: $namd_exec"
  echo "  Input: $infile"
  echo "  Output: $outfile"
  echo ""
  echo "Launcher: srun ${srun_args[*]}"
  echo "Command: ${cmd[*]}"
  echo "========================================"
  echo ""
} | tee -a "$outfile"

# --- Launch NAMD with cgroup isolation ---
# Method: Start with srun, then move process to cgroup

# Use cgexec if available (cleaner)
if command -v cgexec >/dev/null 2>&1; then
  stdbuf -oL -eL \
    cgexec -g "cpuset:$cgroup_path" \
    srun "${srun_args[@]}" "${cmd[@]}" \
    >> "$outfile" 2>&1 &
  pid=$!
else
  # Fallback: launch then move to cgroup
  stdbuf -oL -eL srun "${srun_args[@]}" "${cmd[@]}" >> "$outfile" 2>&1 &
  pid=$!
  
  # Give srun time to spawn children
  sleep 0.5
  
  # Move process and children to cgroup (requires sudo or proper permissions)
  if [[ -f "$cgroup_path/cgroup.procs" ]]; then
    echo "$pid" > "$cgroup_path/cgroup.procs" 2>/dev/null || \
      echo "WARNING: Could not move PID $pid to cgroup (may lack permissions)" | tee -a "$outfile"
    
    # Also move child processes
    for child_pid in $(pgrep -P "$pid" 2>/dev/null || true); do
      echo "$child_pid" > "$cgroup_path/cgroup.procs" 2>/dev/null || true
    done
  else
    echo "WARNING: Cannot find cgroup.procs in $cgroup_path" | tee -a "$outfile"
  fi
fi

# --- Report launch status ---
{
  echo "[$(date '+%F %T')] Launched NAMD"
  echo "  PID: $pid"
  echo "  Cgroup: $cgroup_path"
  echo "  Log: $outfile"
  echo ""
  echo "Monitor with:"
  echo "  tail -f $outfile"
  echo "  watch -n 1 'nvidia-smi | grep $mig_index'"
  echo "  htop -p $pid"
  echo ""
} | tee -a "$outfile"

echo "PID $pid"