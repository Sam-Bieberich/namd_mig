#!/usr/bin/env bash
# run_namd_mig.sh
#
# Run NAMD3-GPU inside an existing interactive SLURM allocation,
# with robust MIG selection (0-based only) and background execution.
#
# Usage:
#   ./run_namd_mig.sh [--mig N | --mig-uuid UUID] <input.namd> <output.log>
#
# Examples:
#   ./run_namd_mig.sh hanning_namd/stmv.namd test.out
#   ./run_namd_mig.sh --mig 6 hanning_namd/stmv.namd test6.out
#   ./run_namd_mig.sh --mig-uuid MIG-... hanning_namd/stmv.namd test.out
#
# Notes:
#   - Do NOT use 'sudo' on the cluster (root_squash + wiped env).
#   - With a single MIG UUID in CUDA_VISIBLE_DEVICES, use '+devices 0' for NAMD.
#   - To avoid UCX init failures in single-node runs, we default UCX_TLS to local transports.

set -euo pipefail

# --- Modules (match your site template) ---
module load cuda/12.6
module load nvmath/12.6.0
module load openmpi/5.0.5
module load ucx/1.18.0
module load namd-gpu/3.0.2

# --- Defaults (overridable via env) ---
: "${PEMAP:=1-71}"      # worker CPU core map (adjust to your node)
: "${COMMAP:=0}"        # comm thread core
: "${PPN:=71}"          # number of worker threads
: "${DEVICES:=0}"       # with a single MIG in CUDA_VISIBLE_DEVICES, +devices 0 selects it
: "${USE_SRUN:=auto}"   # auto|yes|no
: "${SRUN_MPI:=pmi2}"   # pmi2|pmix|none  (default pmi2 to match site template)

# Favor local UCX transports for single-node, single-rank runs (can be overridden by user)
export UCX_TLS="${UCX_TLS:-self,sm,cuda_copy,cuda_ipc}"

# --- Parse options ---
mig_index=""
mig_uuid=""

args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mig)  # 0-based index
      [[ $# -ge 2 ]] || { echo "ERROR: --mig requires an index"; exit 2; }
      mig_index="$2"; shift 2 ;;
    --mig-uuid)
      [[ $# -ge 2 ]] || { echo "ERROR: --mig-uuid requires a UUID"; exit 2; }
      mig_uuid="$2"; shift 2 ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# //'; exit 0 ;;
    *)
      args+=("$1"); shift ;;
  esac
done

# Expect exactly 2 positional args: input and output
if [[ ${#args[@]} -ne 2 ]]; then
  echo "Usage: $0 [--mig N | --mig-uuid UUID] <input.namd> <output.log>"
  exit 1
fi
infile="${args[0]}"
outfile="${args[1]}"

# Ensure output dir exists and create log file early
mkdir -p "$(dirname -- "$outfile")"
: > "$outfile"

# --- Find NAMD exec ---
if [[ -n "${TACC_NAMD_GPU_BIN:-}" && -x "${TACC_NAMD_GPU_BIN}/namd3" ]]; then
  namd_exec="${TACC_NAMD_GPU_BIN}/namd3"
elif command -v namd3 >/dev/null 2>&1 ; then
  namd_exec="$(command -v namd3)"
else
  echo "ERROR: Cannot find 'namd3'. Ensure the namd-gpu module is loaded." | tee -a "$outfile"
  exit 2
fi

# --- Helper: list MIG UUIDs, one per line (robust to spacing) ---
# Matches lines like:
#   '  MIG 1g.12gb     Device  6: (UUID: MIG-6715c64a-....)'
list_mig_uuids() {
  nvidia-smi -L 2>/dev/null \
  | awk -F 'UUID: ' '/^[[:space:]]*MIG[[:space:]]/{ sub(/\).*/,"",$2); print $2 }'
}

# --- MIG selection ---
if [[ -n "$mig_uuid" && -n "$mig_index" ]]; then
  echo "ERROR: Use either --mig or --mig-uuid, not both." | tee -a "$outfile"
  exit 2
fi

if [[ -n "$mig_uuid" ]]; then
  export CUDA_VISIBLE_DEVICES="$mig_uuid"
elif [[ -n "$mig_index" ]]; then
  [[ "$mig_index" =~ ^[0-9]+$ ]] || { echo "ERROR: --mig requires a non-negative integer index." | tee -a "$outfile"; exit 2; }

  mapfile -t mig_uuids < <(list_mig_uuids || true)

  {
    echo "Detected MIG devices (0-based indices):"
    if [[ ${#mig_uuids[@]} -eq 0 ]]; then
      echo "  <none found>"
    else
      for i in "${!mig_uuids[@]}"; do
        printf "  %d : %s\n" "$i" "${mig_uuids[$i]}"
      done
    fi
  } | tee -a "$outfile"

  if [[ ${#mig_uuids[@]} -eq 0 ]]; then
    echo "ERROR: Could not parse MIG UUIDs from 'nvidia-smi -L'." | tee -a "$outfile"
    echo "       Workaround: pass --mig-uuid <MIG-...> explicitly." | tee -a "$outfile"
    exit 2
  fi
  if (( mig_index < 0 || mig_index >= ${#mig_uuids[@]} )); then
    max_zero_based=$(( ${#mig_uuids[@]} - 1 ))
    echo "ERROR: --mig $mig_index is out of range (0..$max_zero_based)" | tee -a "$outfile"
    exit 2
  fi

  export CUDA_VISIBLE_DEVICES="${mig_uuids[$mig_index]}"
fi

# --- Decide launcher ---
use_srun=false
if [[ "${USE_SRUN}" == "yes" ]]; then
  use_srun=true
elif [[ "${USE_SRUN}" == "auto" && -n "${SLURM_JOB_ID:-}" ]]; then
  use_srun=true
fi

# --- Build NAMD command ---
cmd=( "$namd_exec" +ppn "$PPN" +pemap "$PEMAP" +commap "$COMMAP" +devices "$DEVICES" "$infile" )

# --- srun MPI arg selection (match site template by default) ---
srun_args=( --ntasks=1 --unbuffered )
case "$SRUN_MPI" in
  pmi2) srun_args+=( --mpi=pmi2 ) ;;
  pmix) srun_args+=( --mpi=pmix ) ;;
  none) : ;; # no --mpi
  *)    srun_args+=( --mpi=pmi2 ) ;; # default to pmi2
esac

# --- Log environment & command ---
{
  echo "[$(date '+%F %T')] Starting NAMD"
  echo " Host: $(hostname)"
  echo " SLURM_JOB_ID: ${SLURM_JOB_ID:-<none>}"
  echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
  echo " UCX_TLS: ${UCX_TLS:-<not set>}"
  echo " PEMAP: $PEMAP   COMMAP: $COMMAP   PPN: $PPN   DEVICES: $DEVICES"
  if $use_srun; then
    echo " Launcher: srun ${srun_args[*]}"
  else
    echo " Launcher: direct exec (no srun)"
  fi
  echo " Command: ${cmd[*]}"
  echo " Output:  $outfile"
} | tee -a "$outfile"

# --- Launch in background with line-buffered output ---
if $use_srun; then
  stdbuf -oL -eL srun "${srun_args[@]}" "${cmd[@]}" >> "$outfile" 2>&1 &
else
  stdbuf -oL -eL "${cmd[@]}" >> "$outfile" 2>&1 &
fi

pid=$!
echo "[$(date '+%F %T')] Launched (PID $pid). Use 'tail -f $outfile' to monitor." | tee -a "$outfile"
echo "PID $pid"