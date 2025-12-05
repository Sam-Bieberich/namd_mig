#!/bin/bash
# setup_mig_cgroups.sh
# Creates cgroup v2 partitions matched to MIG instances for NAMD workloads
# Each MIG gets 10 CPU cores and 1 NUMA node

set -e

# Configuration
TOTAL_CPUS=72
NUM_MIG_INSTANCES=7
CORES_PER_MIG=10
CGROUP_BASE="/sys/fs/cgroup/mig"

# Explicit CPU and NUMA mappings for 7 MIG instances
# MIG 0: cores 0-9,   NUMA node 0
# MIG 1: cores 10-19, NUMA node 1
# ... and so on
MIG_CPU_RANGES=("0-9" "10-19" "20-29" "30-39" "40-49" "50-59" "60-71")
MIG_MEM_NODES=("0" "1" "2" "3" "4" "5" "6")

echo "=========================================="
echo "MIG + CPU + Memory Partitioning Setup"
echo "=========================================="
echo "Total CPUs: $TOTAL_CPUS"
echo "MIG Instances: $NUM_MIG_INSTANCES"
echo "Cores per MIG: $CORES_PER_MIG"
echo ""

# Check for root privileges
if [ "$EUID" -ne 0 ]; then 
    echo "ERROR: This script must be run with sudo"
    echo "Usage: sudo $0"
    exit 1
fi

# Verify cgroup v2 is mounted
if [ ! -f "/sys/fs/cgroup/cgroup.controllers" ]; then
    echo "ERROR: cgroup v2 not detected at /sys/fs/cgroup"
    echo "Please ensure your system is using cgroup v2"
    exit 1
fi

# Get MIG device UUIDs from nvidia-smi
echo "Detecting MIG instances..."
mapfile -t MIG_UUIDS < <(nvidia-smi -L 2>/dev/null | grep "MIG" | grep -oP 'UUID: \K[^)]+')

if [ ${#MIG_UUIDS[@]} -eq 0 ]; then
    echo "ERROR: No MIG instances found"
    echo "Please run the MIG setup script first (e.g., mig_easy_setup.sh)"
    exit 1
fi

if [ ${#MIG_UUIDS[@]} -ne $NUM_MIG_INSTANCES ]; then
    echo "WARNING: Expected $NUM_MIG_INSTANCES MIG instances, found ${#MIG_UUIDS[@]}"
    echo "Adjusting configuration..."
    NUM_MIG_INSTANCES=${#MIG_UUIDS[@]}
fi

echo "Found $NUM_MIG_INSTANCES MIG instances:"
for i in "${!MIG_UUIDS[@]}"; do
    echo "  MIG $i: ${MIG_UUIDS[$i]}"
done
echo ""

# Create base MIG cgroup directory
echo "Creating base cgroup directory: $CGROUP_BASE"
mkdir -p "$CGROUP_BASE"

# Enable cpuset controller at root
if [ -f "/sys/fs/cgroup/cgroup.subtree_control" ]; then
    echo "+cpuset" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null || true
fi

# Initialize MIG parent cgroup
if [ -d "$CGROUP_BASE" ]; then
    # Enable cpuset controller in parent
    if [ -f "$CGROUP_BASE/cgroup.subtree_control" ]; then
        echo "+cpuset" > "$CGROUP_BASE/cgroup.subtree_control" 2>/dev/null || true
    fi
    
    # Set parent memory nodes (get from root effective)
    ROOT_MEMS=$(cat /sys/fs/cgroup/cpuset.mems.effective 2>/dev/null || echo "0-7")
    echo "$ROOT_MEMS" > "$CGROUP_BASE/cpuset.mems" 2>/dev/null || true
    
    # Set parent CPUs (all available)
    echo "0-$((TOTAL_CPUS-1))" > "$CGROUP_BASE/cpuset.cpus" 2>/dev/null || true
fi

echo ""
echo "Creating cgroups for each MIG instance..."
echo "=========================================="

# Create cgroup for each MIG instance
for i in $(seq 0 $((NUM_MIG_INSTANCES - 1))); do
    CGROUP_PATH="$CGROUP_BASE/mig$i"
    
    # Get CPU range and NUMA node for this MIG
    CPU_RANGE="${MIG_CPU_RANGES[$i]}"
    MEM_NODE="${MIG_MEM_NODES[$i]}"
    MIG_UUID="${MIG_UUIDS[$i]}"
    
    echo ""
    echo "Setting up MIG instance $i:"
    echo "  UUID: $MIG_UUID"
    echo "  CPU cores: $CPU_RANGE"
    echo "  NUMA mem node: $MEM_NODE"
    
    # Create cgroup directory
    mkdir -p "$CGROUP_PATH"
    
    # Enable cpuset controller in parent (redundant but safe)
    if [ -f "$CGROUP_BASE/cgroup.subtree_control" ]; then
        echo "+cpuset" > "$CGROUP_BASE/cgroup.subtree_control" 2>/dev/null || true
    fi
    
    # CRITICAL: Set memory nodes BEFORE CPUs (cgroup v2 requirement)
    echo "$MEM_NODE" > "$CGROUP_PATH/cpuset.mems" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "  WARNING: Failed to set memory node, trying parent mems"
        PARENT_MEMS=$(cat "$CGROUP_BASE/cpuset.mems" 2>/dev/null || echo "0")
        echo "$PARENT_MEMS" > "$CGROUP_PATH/cpuset.mems" 2>/dev/null || true
    fi
    
    # Set CPU affinity
    echo "$CPU_RANGE" > "$CGROUP_PATH/cpuset.cpus" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "  ERROR: Failed to set CPU range $CPU_RANGE"
        continue
    fi
    
    # Verify settings
    if [ -f "$CGROUP_PATH/cpuset.cpus.effective" ]; then
        EFFECTIVE_CPUS=$(cat "$CGROUP_PATH/cpuset.cpus.effective")
        EFFECTIVE_MEMS=$(cat "$CGROUP_PATH/cpuset.mems.effective")
        echo "  ✓ Created cgroup: $CGROUP_PATH"
        echo "    Effective CPUs: $EFFECTIVE_CPUS"
        echo "    Effective MEMs: $EFFECTIVE_MEMS"
    else
        echo "  ⚠ Created but cannot verify: $CGROUP_PATH"
    fi
done

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Verification commands:"
echo "  # List all MIG cgroups"
echo "  for i in {0..6}; do echo \"MIG \$i: \$(cat $CGROUP_BASE/mig\$i/cpuset.cpus.effective)\"; done"
echo ""
echo "  # Check MIG devices"
echo "  nvidia-smi -L | grep MIG"
echo ""
echo "Next steps:"
echo "  1. Verify setup with: sudo ./test_mig_cgroups.sh 0"
echo "  2. Run NAMD with: ./run_namd_mig.sh --mig 0 input.namd output.log"
echo ""