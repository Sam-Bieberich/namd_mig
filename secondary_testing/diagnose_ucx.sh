#!/bin/bash
# diagnose_ucx.sh
# Diagnose UCX and MIG configuration issues
# Usage: ./diagnose_ucx.sh [mig_index]

MIG_IDX="${1:-0}"

echo "========================================"
echo "UCX and MIG Diagnostic Tool"
echo "========================================"
echo ""

# Load modules
module load cuda/12.6 2>/dev/null || true
module load nvmath/12.6.0 2>/dev/null || true
module load openmpi/5.0.5 2>/dev/null || true
module load ucx/1.18.0 2>/dev/null || true
module load namd-gpu/3.0.2 2>/dev/null || true

echo "1. Module Status"
echo "----------------------------------------"
module list 2>&1 | grep -E "(cuda|ucx|openmpi|namd)"
echo ""

echo "2. MIG Device Status"
echo "----------------------------------------"
nvidia-smi -L | grep MIG
echo ""

echo "3. UCX Configuration"
echo "----------------------------------------"
echo "UCX_TLS: ${UCX_TLS:-not set}"
echo "UCX_MEMTYPE_CACHE: ${UCX_MEMTYPE_CACHE:-not set}"
echo "UCX_IB_GPU_DIRECT_RDMA: ${UCX_IB_GPU_DIRECT_RDMA:-not set}"
echo ""

if command -v ucx_info >/dev/null 2>&1; then
    echo "UCX Version:"
    ucx_info -v 2>&1 | head -n 5
    echo ""
    
    echo "Available UCX Transports:"
    ucx_info -d | grep -A 5 "Transport:" | head -n 20
else
    echo "ucx_info not found"
fi
echo ""

echo "4. CUDA Visibility Test"
echo "----------------------------------------"
mapfile -t mig_uuids < <(nvidia-smi -L 2>/dev/null | grep "MIG" | grep -oP 'UUID: \K[^)]+')

if [ ${#mig_uuids[@]} -eq 0 ]; then
    echo "ERROR: No MIG devices found"
    exit 1
fi

if (( MIG_IDX >= ${#mig_uuids[@]} )); then
    echo "ERROR: MIG index $MIG_IDX out of range"
    exit 1
fi

MIG_UUID="${mig_uuids[$MIG_IDX]}"
echo "Testing MIG $MIG_IDX: $MIG_UUID"
echo ""

# Test CUDA device visibility with a simple CUDA program
cat > /tmp/test_cuda_$$.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("ERROR: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    printf("CUDA device count: %d\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    }
    
    return 0;
}
EOF

echo "Compiling CUDA test program..."
if nvcc -o /tmp/test_cuda_$$ /tmp/test_cuda_$$.cu 2>/dev/null; then
    echo "Running with CUDA_VISIBLE_DEVICES=$MIG_UUID"
    CUDA_VISIBLE_DEVICES=$MIG_UUID /tmp/test_cuda_$$
    rm -f /tmp/test_cuda_$$ /tmp/test_cuda_$$.cu
else
    echo "ERROR: Could not compile CUDA test (nvcc not available)"
    rm -f /tmp/test_cuda_$$.cu
fi
echo ""

echo "5. MPI + UCX Test"
echo "----------------------------------------"
echo "Testing simple MPI program with UCX..."

cat > /tmp/test_mpi_$$.c << 'EOF'
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    printf("Rank %d of %d: MPI initialized successfully\n", rank, size);
    
    MPI_Finalize();
    return 0;
}
EOF

echo "Compiling MPI test program..."
if mpicc -o /tmp/test_mpi_$$ /tmp/test_mpi_$$.c 2>/dev/null; then
    echo "Running with srun..."
    export UCX_TLS=self,sm
    export UCX_WARN_UNUSED_ENV_VARS=n
    
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        srun --ntasks=1 --mpi=pmix /tmp/test_mpi_$$ 2>&1
    else
        echo "Not in SLURM allocation, running directly..."
        /tmp/test_mpi_$$
    fi
    
    rm -f /tmp/test_mpi_$$ /tmp/test_mpi_$$.c
else
    echo "ERROR: Could not compile MPI test (mpicc not available)"
    rm -f /tmp/test_mpi_$$.c
fi
echo ""

echo "6. Recommended UCX Settings for MIG"
echo "----------------------------------------"
cat << 'SETTINGS'
# Add these to your run script or environment:

export UCX_TLS=self,sm,cuda_copy,cuda_ipc
export UCX_MEMTYPE_CACHE=n
export UCX_WARN_UNUSED_ENV_VARS=n
export UCX_IB_GPU_DIRECT_RDMA=no
export UCX_RNDV_SCHEME=put_zcopy

# For debugging UCX issues:
# export UCX_LOG_LEVEL=info

# If still having issues, try disabling CUDA transports:
# export UCX_TLS=self,sm
SETTINGS
echo ""

echo "7. Alternative: Try Without UCX"
echo "----------------------------------------"
echo "If UCX continues to fail, you can run NAMD without MPI:"
echo ""
echo "Direct execution (no srun, single-node only):"
echo "  CUDA_VISIBLE_DEVICES=$MIG_UUID \\"
echo "  taskset -c 0-9 \\"
echo "  namd3 +ppn 9 +pemap 1-9 +commap 0 +devices 0 input.namd > output.log"
echo ""

echo "========================================"
echo "Diagnostic Complete"
echo "========================================"