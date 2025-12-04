#!/usr/bin/env bash
# run_namd_mig.sh
#
# Run NAMD3-GPU inside an existing interactive SLURM allocation,
# with robust MIG selection (0-based only), **automatic CPU pinning by MIG index**,
# and background execution with streaming logs.
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

# --- Defaults you can override via env ---
: "${TOTAL_LOGICAL_CPUS:=72}"  # logical CPUs on the node (Linux numbering 0..71)
: "${MIG_SLOTS:=7}"            # number of MIG slices you target (GH200 1g.12gb typically 7)
: "${AUTO_CPU_PIN:=1}"         # 1=auto pin cores by MIG index; 0=leave PPN/PEMAP/COMMAP as provided
: "${COMM_SHARES_CORE:=1}"     # 1=comm thread shares a core inside the block (PPN=10), 0=reserve 1 core (PPN=9)

# These are baseline defaults; they will be **overwritten** if AUTO_CPU_PIN=1
: "${PEMAP:=1-71}"             # worker CPU map (Charm++)
: "${COMMAP:=0}"               # comm thread CPU (Charm++)
: "${PPN:=71}"                 # number of worker threads
: "${DEVICES:=0}"              # with a single GPU/MIG visible, +devices 0 selects it
: "${USE_SRUN:=auto}"          # auto|yes|no
: "${SRUN_MPI:=pmi2}"          # pmi2|pmix|none  (default pmi2 to match site template)

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

# --- MIG selection (resolve index and/or uuid) ---
if [[ -n "$mig_uuid" && -n "$mig_index" ]]; then
  echo "ERROR: Use either --mig or --mig-uuid, not both." | tee -a "$outfile"
  exit 2
fi

sel_index=""
if [[ -n "$mig_uuid" ]]; then
  export CUDA_VISIBLE_DEVICES="$mig_uuid"
  # Try to derive its 0-based index (for CPU pinning)
  mapfile -t _uuids < <(list_mig_uuids || true)
  for i in "${!_uuids[@]}"; do
    if [[ "${_uuids[$i]}" == "$mig_uuid" ]]; then
      sel_index="$i"
      break
    fi
  done
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
  sel_index="$mig_index"
fi

# --- Auto CPU pinning by MIG index (0..6) ---
if [[ "${AUTO_CPU_PIN}" == "1" ]]; then
  if [[ -z "${sel_index}" ]]; then
    echo "WARNING: No MIG index resolved; AUTO_CPU_PIN requested but cannot compute CPU block. Using provided PPN/PEMAP/COMMAP." | tee -a "$outfile"
  else
    # Compute cores per MIG (e.g., 72/7 = 10); leave any remainder cores unused.
    cores_per_mig=$(( TOTAL_LOGICAL_CPUS / MIG_SLOTS ))
    start_core=$(( sel_index * cores_per_mig ))
    end_core=$(( start_core + cores_per_mig - 1 ))
    if (( end_core >= TOTAL_LOGICAL_CPUS )); then
      end_core=$(( TOTAL_LOGICAL_CPUS - 1 ))
    fi

    if (( start_core < 0 || start_core >= TOTAL_LOGICAL_CPUS )); then
      echo "ERROR: Computed start core $start_core is out of range for TOTAL_LOGICAL_CPUS=$TOTAL_LOGICAL_CPUS" | tee -a "$outfile"
      exit 2
    fi

    if [[ "${COMM_SHARES_CORE}" == "1" ]]; then
      # Use the full block for workers, comm shares first core
      PEMAP="${start_core}-${end_core}"
      COMMAP="${start_core}"
      PPN="${cores_per_mig}"
    else
      # Reserve first core for comm; workers use the remaining cores in the block
      if (( cores_per_mig <= 1 )); then
        echo "ERROR: cores_per_mig=$cores_per_mig too small to reserve a comm core." | tee -a "$outfile"
        exit 2
      fi
      PEMAP="$((start_core+1))-${end_core}"
      COMMAP="${start_core}"
      PPN="$((cores_per_mig - 1))"
    fi

    {
      echo "CPU pinning by MIG index:"
      echo "  TOTAL_LOGICAL_CPUS=$TOTAL_LOGICAL_CPUS  MIG_SLOTS=$MIG_SLOTS  cores_per_mig=$cores_per_mig"
      echo "  MIG index=$sel_index  -> core block ${start_core}-${end_core}"
      echo "  COMM_SHARES_CORE=$COMM_SHARES_CORE  => COMMAP=$COMMAP  PEMAP=$PEMAP  PPN=$PPN"
    } | tee -a "$outfile"
  fi
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