# NAMD + NVIDIA MIG Testing Repository

This repository is for testing NAMD (Nanoscale Molecular Dynamics) using NVIDIA MIG (Multi-Instance GPU) partitions.

## Modules needed

```bash
module load cuda/12.6
module load nvmath/12.6.0
module load openmpi/5.0.5
module load namd-gpu/3.0.1
```

## How to see results in real time

```bash
tail -f test.out
```

## Note 10/22

CPUs are not partitioned appropriately for the MIG test.

## Workflow

The following command runs NAMD with MIG level 6 using the provided configuration file and outputs the results to `test6.out`.

```bash
bash run_namd_mig.sh --mig 6 hanning_namd/stmv.namd test6.out
```

Current code puts the job in the right place, but the CPU partitioning is not yet optimal. 

Also when you run several at the same time it does not work
