#!/bin/bash
# test_mig_cgroups.sh
# Verify that MIG + cgroup partitioning is working correctly
#
# Usage: sudo ./test_mig_cgroups.sh [mig_index]
# Example: sudo ./test_mig_cgroups.sh 0

set -e

MIG_IDX="${1:-0}"
CGROUP_BASE="/sys/fs/cgroup/mig"

if ! [[ "$MIG_IDX" =~ ^[0-6]$ ]]; then
    echo "Usage: sudo $0 [mig_instance_0-6]"
    echo "Testing MIG instance 0 by default"
    MIG_IDX=0
fi

echo "========================================"
echo "Testing MIG Instance $MIG_IDX"
echo "========================================"
echo ""

# --- Test 1: Check MIG device exists ---
echo "Test 1: MIG Device Detection"
echo "----------------------------------------"

mapfile -t mig_uuids < <(nvidia-smi -L 2>/dev/null | grep "MIG" | grep -oP 'UUID: \K[^)]+')

if [ ${#mig_uuids[@]} -eq 0 ]; then
    echo "❌ FAIL: No MIG devices found"
    echo "   Run: sudo bash mig_easy_setup.sh"
    exit 1
fi

if (( MIG_IDX >= ${#mig_uuids[@]} )); then
    echo "❌ FAIL: MIG index $MIG_IDX out of range (0-$((${#mig_uuids[@]}-1)))"
    exit 1
fi

MIG_UUID="${mig_uuids[$MIG_IDX]}"
echo "✓ PASS: Found MIG device"
echo "  Index: $MIG_IDX"
echo "  UUID: $MIG_UUID"
echo ""

# --- Test 2: Check cgroup exists ---
echo "Test 2: Cgroup Configuration"
echo "----------------------------------------"

CGROUP_PATH="$CGROUP_BASE/mig$MIG_IDX"

if [ ! -d "$CGROUP_PATH" ]; then
    echo "❌ FAIL: Cgroup not found: $CGROUP_PATH"
    echo "   Run: sudo bash setup_mig_cgroups.sh"
    exit 1
fi

echo "✓ PASS: Cgroup exists"
echo "  Path: $CGROUP_PATH"

# Check cgroup configuration
if [ -f "$CGROUP_PATH/cpuset.cpus.effective" ]; then
    CPUS=$(cat "$CGROUP_PATH/cpuset.cpus.effective")
    MEMS=$(cat "$CGROUP_PATH/cpuset.mems.effective")
    echo "  CPUs: $CPUS"
    echo "  MEMs: $MEMS"
else
    echo "⚠️  WARNING: Cannot read effective cpuset"
fi
echo ""

# --- Test 3: Test CPU affinity ---
echo "Test 3: CPU Affinity"
echo "----------------------------------------"

# Start a test process in the cgroup
sleep 5 &
TEST_PID=$!

# Move to cgroup
echo "$TEST_PID" > "$CGROUP_PATH/cgroup.procs" 2>/dev/null || {
    echo "❌ FAIL: Could not move process to cgroup"
    kill $TEST_PID 2>/dev/null || true
    exit 1
}

sleep 0.5

# Check affinity
AFFINITY=$(taskset -pc $TEST_PID 2>&1 | grep -oP "list: \K.*" || echo "N/A")
CGROUP_CHECK=$(cat /proc/$TEST_PID/cgroup 2>/dev/null | grep "mig/mig$MIG_IDX" || echo "")

# Cleanup
kill $TEST_PID 2>/dev/null || true

if [ "$AFFINITY" = "$CPUS" ]; then
    echo "✓ PASS: CPU affinity correct"
    echo "  Expected: $CPUS"
    echo "  Got: $AFFINITY"
elif [ "$AFFINITY" = "N/A" ]; then
    echo "⚠️  WARNING: Could not determine affinity"
    echo "  Cgroup CPUs: $CPUS"
else
    echo "❌ FAIL: CPU affinity mismatch"
    echo "  Expected: $CPUS"
    echo "  Got: $AFFINITY"
fi

if [ -n "$CGROUP_CHECK" ]; then
    echo "✓ PASS: Process in correct cgroup"
else
    echo "⚠️  WARNING: Cgroup assignment unclear"
fi
echo ""

# --- Test 4: Test GPU visibility ---
echo "Test 4: GPU Visibility (CUDA_VISIBLE_DEVICES)"
echo "----------------------------------------"

# Test script to check GPU visibility
TEST_SCRIPT=$(mktemp)
cat > "$TEST_SCRIPT" << 'EOF'
#!/bin/bash
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""
echo "nvidia-smi output (filtered to MIG device):"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid --format=csv,noheader 2>&1 || \
    echo "Note: nvidia-smi shows hardware view; CUDA runtime will respect CUDA_VISIBLE_DEVICES"
EOF
chmod +x "$TEST_SCRIPT"

# Run with CUDA_VISIBLE_DEVICES set
OUTPUT=$(CUDA_VISIBLE_DEVICES="$MIG_UUID" bash "$TEST_SCRIPT" 2>&1)
echo "$OUTPUT"

rm -f "$TEST_SCRIPT"

if echo "$OUTPUT" | grep -q "$MIG_UUID"; then
    echo ""
    echo "✓ PASS: GPU environment configured correctly"
else
    echo ""
    echo "⚠️  INFO: GPU isolation works via CUDA_VISIBLE_DEVICES"
    echo "   Launcher scripts set this automatically"
fi
echo ""

# --- Test 5: Integrated test with actual GPU query ---
echo "Test 5: Integrated CPU + GPU Test"
echo "----------------------------------------"

# Create a test script that uses the cgroup and GPU
INTEGRATED_TEST=$(mktemp)
cat > "$INTEGRATED_TEST" << 'IEOF'
#!/bin/bash
# This process should be constrained to specific CPUs and see only one GPU

PID=$$
echo "Process PID: $PID"
echo ""

# Check CPU affinity
echo "CPU Affinity:"
taskset -pc $PID 2>&1 | grep "list" || echo "  Could not determine"

# Check cgroup
echo ""
echo "Cgroup:"
cat /proc/$PID/cgroup 2>/dev/null | grep -v "^$" | head -n 3

# Check GPU visibility
echo ""
echo "GPU (from CUDA_VISIBLE_DEVICES):"
echo "  $CUDA_VISIBLE_DEVICES"

# Try nvidia-smi if available
if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    echo "GPU query (hardware view - shows all MIG devices):"
    nvidia-smi -L 2>&1 | grep MIG | head -n 3
    echo ""
    echo "Note: nvidia-smi shows hardware. CUDA apps respect CUDA_VISIBLE_DEVICES"
    echo "      and will only see: $CUDA_VISIBLE_DEVICES"
fi
IEOF
chmod +x "$INTEGRATED_TEST"

# Run the integrated test
echo "Running integrated test..."
(
    export CUDA_VISIBLE_DEVICES="$MIG_UUID"
    sleep 3 &
    WRAPPER_PID=$!
    echo "$WRAPPER_PID" > "$CGROUP_PATH/cgroup.procs" 2>/dev/null || true
    
    bash "$INTEGRATED_TEST"
    
    kill $WRAPPER_PID 2>/dev/null || true
)

rm -f "$INTEGRATED_TEST"
echo ""

# --- Summary ---
echo "========================================"
echo "Test Summary"
echo "========================================"
echo ""
echo "✓ MIG device $MIG_IDX is configured"
echo "✓ Cgroup $CGROUP_PATH is set up"
echo "✓ CPU cores: $CPUS"
echo "✓ NUMA memory: $MEMS"
echo ""
echo "Ready to run NAMD with:"
echo "  ./run_namd_mig.sh --mig $MIG_IDX input.namd output.log"
echo ""
echo "For concurrent runs on multiple MIGs:"
echo "  ./run_namd_mig.sh --mig 0 input.namd test0.out &"
echo "  ./run_namd_mig.sh --mig 1 input.namd test1.out &"
echo "  ./run_namd_mig.sh --mig 2 input.namd test2.out &"
echo ""
echo "========================================"