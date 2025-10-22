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

```
tail -f test.out
```

## Protocol

```
./local_run.sh hanning_namd/stmv.namd test.out
``` 

## Note 10/22

CPUs are not partitioned appropriately for the MIG test. 