#!/usr/bin/env bash
# Notes:
#   * UCX-enabled NAMD must run under a PMI/PMIx launcher (e.g. srun). Do NOT run directly. 
#   * This script is designed for interactive SLURM allocations on Vista GH nodes.
#   * It allows multiple background runs, each on its own MIG slice and CPU block.
#   * Uses cgroup CPU partitioning instead of taskset for resource isolation.

set -euo pipefail

# --- Modules (match site) ---
module load cuda/12.6
module load nvmath/12.6.0
module load openmpi/5.0.5
module load ucx/1.18.0
module load namd-gpu/3.0.2

# --- Cluster/UCX defaults (can be overridden in env) ---
# Keep UCX on-node; you may tighten/loosen as needed.
export UCX_TLS="${UCX_TLS:-self,sm,cuda_copy,cuda_ipc}"
# Optional UCX tweaks that often improve reliability; uncomment if needed:
# export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"  # common recommendation in CUDA workflows

# --- MIG-aware CPU partitioning defaults ---
: "${TOTAL_LOGICAL_CPUS:=72}"   # logical CPUs numbered 0..71
: "${MIG_SLOTS:=7}"             # number of MIG slices (GH200 1g.12gb -> 7)
: "${CORES_PER_MIG:=10}"        # cores per MIG partition (72/7 ≈ 10)
: "${AUTO_CPU_PIN:=1}"          # 1=derive +pemap/+commap/PPN from MIG index
: "${COMM_SHARES_CORE:=1}"      # 1=comm thread shares first core of block; 0=reserve one core

# Charm++ thread mapping (will be overridden if AUTO_CPU_PIN=1)
: "${PEMAP:=1-71}"
: "${COMMAP:=0}"
: "${PPN:=71}"

# With a single (MIG) device visible, NAMD uses +devices 0
: "${DEVICES:=0}"

# Force srun, since UCX+Charm++ needs PMI/PMIx KVS
: "${SRUN_MPI:=pmix}"           # pmix|pmi2|none (OpenMPI 5 pairs well with pmix)
: "${SRUN_OVERLAP:=1}"          # allow concurrent steps inside your allocation

# Cgroup base path (pre-existing MIG cgroups)
: "${CGROUP_MIG_BASE:=/sys/fs/cgroup/mig}"

# --- Parse options ---
mig_index=""
mig_uuid=""
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mig)       mig_index="${2:?}"; shift 2 ;;
    --mig-uuid)  mig_uuid="${2:?}"; shift 2 ;;
    -h|--help)
      cat <<'USAGE'
Usage:
  ./cgroup_mig.sh [--mig N | --mig-uuid UUID] <input.namd> <output.log>

Examples:
  ./cgroup_mig.sh --mig 2 stmv.namd test2.out
  ./cgroup_mig.sh --mig 0 stmv.namd test0.out   # can run concurrently

CPU Partitioning (10 cores per MIG):
  MIG 0: cores 0-9
  MIG 1: cores 10-19
  MIG 2: cores 20-29
  MIG 3: cores 30-39
  MIG 4: cores 40-49
  MIG 5: cores 50-59
  MIG 6: cores 60-69
USAGE
      exit 0;;
    *) args+=("$1"); shift ;;
  esac
done
[[ ${#args[@]} -eq 2 ]] || { echo "Usage: $0 [--mig N | --mig-uuid UUID] <input.namd> <output.log>"; exit 1; }
infile="${args[0]}"
outfile="${args[1]}"

mkdir -p "$(dirname -- "$outfile")"
: > "$outfile"

# --- Locate namd3 ---
if [[ -n "${TACC_NAMD_GPU_BIN:-}" && -x "${TACC_NAMD_GPU_BIN}/namd3" ]]; then
  namd_exec="${TACC_NAMD_GPU_BIN}/namd3"
elif command -v namd3 >/dev/null 2>&1; then
  namd_exec="$(command -v namd3)"
else
  echo "ERROR: 'namd3' not found; check that the namd-gpu module is loaded." | tee -a "$outfile"
  exit 2
fi

# --- Helper: list MIG UUIDs (0-based order) ---
list_mig_uuids() {
  nvidia-smi -L 2>/dev/null \
  | awk -F 'UUID: ' '/^[[:space:]]*MIG[[:space:]]/{ sub(/\).*/,"",$2); print $2 }'
}

# --- Resolve MIG UUID and index ---
if [[ -n "$mig_uuid" && -n "$mig_index" ]]; then
  echo "ERROR: Use either --mig or --mig-uuid (not both)." | tee -a "$outfile"; exit 2
fi

sel_index=""
if [[ -n "$mig_uuid" ]]; then
  export CUDA_VISIBLE_DEVICES="$mig_uuid"
  mapfile -t _uuids < <(list_mig_uuids || true)
  for i in "${!_uuids[@]}"; do [[ "${_uuids[$i]}" == "$mig_uuid" ]] && sel_index="$i" && break; done
elif [[ -n "$mig_index" ]]; then
  [[ "$mig_index" =~ ^[0-9]+$ ]] || { echo "ERROR: --mig needs a 0-based integer index."; exit 2; }
  mapfile -t mig_uuids < <(list_mig_uuids || true)
  {
    echo "Detected MIG devices (0-based indices):"
    if [[ ${#mig_uuids[@]} -eq 0 ]]; then
      echo "  <none found>"
    else
      for i in "${!mig_uuids[@]}"; do printf "  %d : %s\n" "$i" "${mig_uuids[$i]}"; done
    fi
  } | tee -a "$outfile"
  (( ${#mig_uuids[@]} > 0 )) || { echo "ERROR: Could not parse MIG UUIDs from nvidia-smi -L."; exit 2; }
  (( mig_index >= 0 && mig_index < ${#mig_uuids[@]} )) || { echo "ERROR: --mig $mig_index out of range (0..$(( ${#mig_uuids[@]}-1 )))."; exit 2; }
  export CUDA_VISIBLE_DEVICES="${mig_uuids[$mig_index]}"
  sel_index="$mig_index"
else
  echo "WARNING: No MIG selected; relying on job-level GPU binding." | tee -a "$outfile"
fi

# --- MIG-based CPU partition assignment (10 cores per MIG) ---
if [[ -z "${sel_index}" ]]; then
  echo "ERROR: MIG index not resolved; cannot assign CPU partition." | tee -a "$outfile"
  exit 2
fi

[[ "$sel_index" =~ ^[0-6]$ ]] || { echo "ERROR: MIG index must be 0-6, got $sel_index"; exit 2; }

# Calculate core range: MIG N gets cores [N*10, N*10+9]
start_core=$(( sel_index * CORES_PER_MIG ))
end_core=$(( start_core + CORES_PER_MIG - 1 ))

# Ensure we don't exceed total CPUs
(( end_core < TOTAL_LOGICAL_CPUS )) || end_core=$(( TOTAL_LOGICAL_CPUS - 1 ))

# --- Auto CPU pinning for Charm++ ---
if [[ "${AUTO_CPU_PIN}" == "1" ]]; then
  if [[ "${COMM_SHARES_CORE}" == "1" ]]; then
    PEMAP="${start_core}-${end_core}"
    COMMAP="${start_core}"
    PPN="${CORES_PER_MIG}"
  else
    (( CORES_PER_MIG > 1 )) || { echo "ERROR: CORES_PER_MIG=$CORES_PER_MIG too small to reserve a comm core."; exit 2; }
    PEMAP="$((start_core+1))-${end_core}"
    COMMAP="${start_core}"
    PPN="$((CORES_PER_MIG - 1))"
  fi

  {
    echo "CPU partition by MIG index:"
    echo "  MIG_SLOTS=$MIG_SLOTS  CORES_PER_MIG=$CORES_PER_MIG"
    echo "  MIG index=$sel_index  -> core range ${start_core}-${end_core}"
    echo "  COMM_SHARES_CORE=$COMM_SHARES_CORE  => COMMAP=$COMMAP  PEMAP=$PEMAP  PPN=$PPN"
  } | tee -a "$outfile"
fi

# --- Use pre-existing cgroup for this MIG partition ---
cgroup_path="${CGROUP_MIG_BASE}/mig${sel_index}"

# Verify cgroup exists
if [[ ! -d "$cgroup_path" ]]; then
  echo "ERROR: Cgroup not found: $cgroup_path" | tee -a "$outfile"
  echo "       Expected pre-existing cgroups at ${CGROUP_MIG_BASE}/mig{0..6}" | tee -a "$outfile"
  exit 2
fi

# Verify cpuset configuration (informational)
if [[ -f "${cgroup_path}/cpuset.cpus" ]]; then
  actual_cpus=$(cat "${cgroup_path}/cpuset.cpus" 2>/dev/null || echo "<unreadable>")
  {
    echo "Using pre-existing cgroup: $cgroup_path"
    echo "  cpuset.cpus: $actual_cpus"
  } | tee -a "$outfile"
elif [[ -f "${cgroup_path}/cpus" ]]; then
  actual_cpus=$(cat "${cgroup_path}/cpus" 2>/dev/null || echo "<unreadable>")
  {
    echo "Using pre-existing cgroup: $cgroup_path"
    echo "  cpus: $actual_cpus"
  } | tee -a "$outfile"
else
  echo "WARNING: Could not verify cpuset configuration in $cgroup_path" | tee -a "$outfile"
fi

# --- Must be inside a SLURM allocation for srun to work here ---
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: Not inside an interactive SLURM allocation. Use 'salloc' or your site's interactive queue." | tee -a "$outfile"
  exit 2
fi

# --- Build NAMD command (Charm++) ---
cmd=( "$namd_exec" +ppn "$PPN" +pemap "$PEMAP" +commap "$COMMAP" +devices "$DEVICES" "$infile" )

# --- srun step arguments (concurrency-friendly) ---
srun_args=( --ntasks=1 --unbuffered --cpu-bind=none --gpu-bind=none )
# Give each step its own CPU budget but do NOT let Slurm override our mapping
srun_args+=( --cpus-per-task="$PPN" )
# Allow multiple srun steps to overlap within the same job allocation
[[ "$SRUN_OVERLAP" == "1" ]] && srun_args+=( --overlap )
# Use PMIx by default with OpenMPI 5
case "$SRUN_MPI" in
  pmix) srun_args+=( --mpi=pmix ) ;;
  pmi2) srun_args+=( --mpi=pmi2 ) ;;
  none) : ;;  # rare
  *)    srun_args+=( --mpi=pmix ) ;;
esac

# Also ensure Slurm doesn't apply an implicit CPU binding policy
export SLURM_CPU_BIND=none

# --- Log header ---
{
  echo "[$(date '+%F %T')] Starting NAMD with cgroup CPU partitioning"
  echo " Host: $(hostname)"
  echo " SLURM_JOB_ID: ${SLURM_JOB_ID:-<none>}"
  echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
  echo " UCX_TLS: ${UCX_TLS:-<not set>}"
  echo " Cgroup: $cgroup_path"
  echo " Expected CPU range: ${start_core}-${end_core}"
  echo " PEMAP: $PEMAP   COMMAP: $COMMAP   PPN: $PPN   DEVICES: $DEVICES"
  echo " Launcher: srun ${srun_args[*]}"
  echo " Command: ${cmd[*]}"
  echo " Output:  $outfile"
} | tee -a "$outfile"

# --- Launch via systemd-run (or cgexec) to use pre-existing cgroup ---
# systemd-run is preferred for attaching to existing cgroups
if command -v systemd-run >/dev/null 2>&1; then
  # Use systemd-run with --scope to run in existing cgroup slice
  # Note: This may require adjustment based on your cgroup hierarchy
  stdbuf -oL -eL systemd-run --scope --slice="mig/${sel_index}" --quiet \
    srun "${srun_args[@]}" "${cmd[@]}" >> "$outfile" 2>&1 &
elif command -v cgexec >/dev/null 2>&1; then
  # Fallback to cgexec with absolute path
  stdbuf -oL -eL cgexec -g ":${cgroup_path}" srun "${srun_args[@]}" "${cmd[@]}" >> "$outfile" 2>&1 &
else
  # Manual fallback: launch then move PID to cgroup
  echo "WARNING: Neither systemd-run nor cgexec found; attempting manual cgroup assignment" | tee -a "$outfile"
  stdbuf -oL -eL srun "${srun_args[@]}" "${cmd[@]}" >> "$outfile" 2>&1 &
  pid=$!
  
  # Move process to cgroup (may require permissions)
  if [[ -f "${cgroup_path}/cgroup.procs" ]]; then
    echo "$pid" > "${cgroup_path}/cgroup.procs" 2>/dev/null || \
      echo "WARNING: Could not move PID $pid to cgroup (may need sudo)" | tee -a "$outfile"
  elif [[ -f "${cgroup_path}/tasks" ]]; then
    echo "$pid" > "${cgroup_path}/tasks" 2>/dev/null || \
      echo "WARNING: Could not move PID $pid to cgroup (may need sudo)" | tee -a "$outfile"
  fi
fi

pid=$!
echo "[$(date '+%F %T')] Launched in cgroup ${cgroup_path} (PID $pid). Use 'tail -f $outfile' to monitor." | tee -a "$outfile"
echo "PID $pid"
