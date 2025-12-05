# MIG + Cgroup Setup Guide for NAMD on GH200

This guide shows how to partition a GH200 node into 7 isolated GPU+CPU+Memory partitions for running multiple NAMD jobs concurrently.

## System Configuration

- **GPU**: 1x GH200 → 7 MIG partitions (1g.12gb each)
- **CPU**: 72 cores → 7 partitions (10 cores each, plus 2 unused)
- **Memory**: 8 NUMA nodes → 7 assignments (one per MIG)

### Partition Mapping

| MIG | CPU Cores | NUMA Node | Use Case |
|-----|-----------|-----------|----------|
| 0   | 0-9       | 0         | NAMD job 1 |
| 1   | 10-19     | 1         | NAMD job 2 |
| 2   | 20-29     | 2         | NAMD job 3 |
| 3   | 30-39     | 3         | NAMD job 4 |
| 4   | 40-49     | 4         | NAMD job 5 |
| 5   | 50-59     | 5         | NAMD job 6 |
| 6   | 60-71     | 6         | NAMD job 7 |

---

## Setup Steps

### 1. Create MIG Partitions

First, create 7 MIG instances on the GPU:

```bash
# Make script executable
chmod +x mig_easy_setup.sh

# Run with sudo (requires admin privileges)
sudo bash mig_easy_setup.sh
```

**Verify MIG creation:**
```bash
nvidia-smi -L | grep MIG
```

You should see 7 MIG devices listed.

### 2. Create CPU/Memory Cgroups

Next, create cgroups that match each MIG partition with CPU cores and NUMA memory:

```bash
# Make script executable
chmod +x setup_mig_cgroups.sh

# Run with sudo
sudo bash setup_mig_cgroups.sh
```

**Verify cgroup creation:**
```bash
for i in {0..6}; do 
    echo "MIG $i: $(cat /sys/fs/cgroup/mig/mig$i/cpuset.cpus.effective)"
done
```

### 3. Test the Setup

Verify everything works before running NAMD:

```bash
# Make test script executable
chmod +x test_mig_cgroups.sh

# Test MIG instance 0
sudo ./test_mig_cgroups.sh 0

# Test another instance
sudo ./test_mig_cgroups.sh 2
```

---

## Running NAMD

### Single Job

Run a single NAMD job on MIG partition 0:

```bash
# Get an interactive allocation
salloc -N 1 -n 1 -p gh -t 01:00:00

# Once in the allocation, run NAMD
./run_namd_mig.sh --mig 0 stmv.namd test0.out

# Monitor output
tail -f test0.out
```

### Multiple Concurrent Jobs

Run multiple NAMD jobs simultaneously, each on its own MIG:

```bash
# Get an interactive allocation
salloc -N 1 -n 1 -p gh -t 02:00:00

# Launch jobs in background (one per MIG)
./run_namd_mig.sh --mig 0 stmv.namd test0.out &
./run_namd_mig.sh --mig 1 stmv.namd test1.out &
./run_namd_mig.sh --mig 2 stmv.namd test2.out &
./run_namd_mig.sh --mig 3 stmv.namd test3.out &
./run_namd_mig.sh --mig 4 stmv.namd test4.out &
./run_namd_mig.sh --mig 5 stmv.namd test5.out &
./run_namd_mig.sh --mig 6 stmv.namd test6.out &

# Monitor all jobs
watch -n 1 'jobs -l; echo ""; nvidia-smi'
```

### Monitor Individual Jobs

```bash
# Watch specific output file
tail -f test0.out

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor CPU usage (shows which cores each job uses)
htop
```

---

## File Descriptions

### Setup Scripts

- **`mig_easy_setup.sh`**: Creates 7 MIG GPU partitions (run once per node reset)
- **`setup_mig_cgroups.sh`**: Creates cgroups for CPU/memory isolation (run after MIG setup)
- **`test_mig_cgroups.sh`**: Verifies that partitioning is working correctly

### Runtime Scripts

- **`run_namd_mig.sh`**: Launches NAMD with proper GPU+CPU+memory isolation
  - Automatically sets CUDA_VISIBLE_DEVICES to the correct MIG UUID
  - Moves process into the appropriate cgroup
  - Configures Charm++ thread mapping for optimal CPU usage

---

## Troubleshooting

### Problem: "No MIG devices found"

**Solution:** Run the MIG setup script first:
```bash
sudo bash mig_easy_setup.sh
nvidia-smi -L | grep MIG
```

### Problem: "Cgroup not found"

**Solution:** Run the cgroup setup script:
```bash
sudo bash setup_mig_cgroups.sh
ls -la /sys/fs/cgroup/mig/
```

### Problem: "Not inside a SLURM allocation"

**Solution:** Get an interactive allocation first:
```bash
salloc -N 1 -n 1 -p gh -t 01:00:00
```

### Problem: Jobs see all 7 GPUs instead of just one

**Explanation:** The `nvidia-smi` command shows the hardware view (all MIG instances). However, CUDA runtime respects `CUDA_VISIBLE_DEVICES`, so each job only uses its assigned MIG.

**Verify:** Check the job's output log - it should show only 1 CUDA device.

### Problem: CPU cores not properly isolated

**Check cgroup assignment:**
```bash
# While job is running, check its PID
jobs -l  # Get PID

# Check cgroup
cat /proc/<PID>/cgroup

# Check CPU affinity
taskset -pc <PID>
```

**Re-run setup if needed:**
```bash
sudo bash setup_mig_cgroups.sh
```

### Problem: Permission denied when moving process to cgroup

**Solution:** The cgroup operations require appropriate permissions. Options:

1. Run the launcher with sudo (not recommended for HPC)
2. Configure cgroup permissions for your user
3. Use the `cgexec` command (if available)

---

## Advanced Usage

### Custom CPU Allocation

Edit `setup_mig_cgroups.sh` to change CPU mappings:

```bash
# Example: Give MIG 0 more cores
MIG_CPU_RANGES=("0-15" "16-25" "26-35" "36-45" "46-55" "56-65" "66-71")
```

### Different MIG Configurations

For 3-way MIG setup instead of 7-way:

```bash
# In setup_mig_cgroups.sh, modify:
NUM_MIG_INSTANCES=3
CORES_PER_MIG=24
MIG_CPU_RANGES=("0-23" "24-47" "48-71")
MIG_MEM_NODES=("0-1" "2-4" "5-7")

# In run_namd_mig.sh, modify:
: "${MIG_SLOTS:=3}"
: "${CORES_PER_MIG:=24}"
```

### Monitoring Resource Usage

```bash
# Real-time GPU monitoring
nvidia-smi dmon -i 0

# Per-MIG statistics
nvidia-smi mig -lgi

# CPU usage per cgroup
systemd-cgtop

# Detailed process info
htop -p $(pgrep -d',' namd3)
```

---

## Quick Reference

### Startup Sequence
```bash
# 1. Setup (once)
sudo bash mig_easy_setup.sh
sudo bash setup_mig_cgroups.sh
sudo ./test_mig_cgroups.sh 0

# 2. Get allocation
salloc -N 1 -n 1 -p gh -t 02:00:00

# 3. Run jobs
./run_namd_mig.sh --mig 0 input.namd output.log
```

### Verification Commands
```bash
# Check MIG devices
nvidia-smi -L | grep MIG

# Check cgroups
ls -la /sys/fs/cgroup/mig/

# Check cgroup CPUs
for i in {0..6}; do echo "MIG $i: $(cat /sys/fs/cgroup/mig/mig$i/cpuset.cpus.effective)"; done

# Monitor running jobs
jobs -l
watch -n 1 nvidia-smi
```

### Cleanup
```bash
# Kill all NAMD processes
pkill -f namd3

# Delete MIG instances (requires node reset usually)
sudo nvidia-smi mig -dci
sudo nvidia-smi mig -dgi

# Remove cgroups (they'll be recreated on next setup)
sudo rm -rf /sys/fs/cgroup/mig
```

---

## Performance Tips

1. **NUMA Awareness**: The setup binds each MIG to a specific NUMA node to minimize memory access latency

2. **Charm++ Threading**: Each job uses 9 worker threads + 1 comm thread (10 cores total)

3. **UCX Settings**: Configured for optimal single-node performance with GPU-aware MPI

4. **Concurrent Runs**: Use `SRUN_OVERLAP=1` to allow multiple srun steps in the same allocation

5. **I/O**: Use separate output files for each job to avoid write conflicts

---

## Notes

- MIG setup requires sudo/admin privileges
- Cgroup setup requires sudo/admin privileges  
- NAMD jobs run as your regular user within the cgroup
- The 2 "unused" cores (60-71 is 12 cores for MIG 6) provide flexibility
- This setup is optimized for GH200 with 72 CPU cores and 8 NUMA nodes