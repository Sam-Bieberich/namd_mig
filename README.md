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

--------------------

## Workflow

First, need to ensure that MIG is running. Can check with nvidia-smi or just run

```bash
sudo bash 7way_mig.sh
```

### Single MIG test (srun)

The following command runs NAMD with MIG level 6 using the provided configuration file and outputs the results to `test6.out`.

```bash
bash run_namd_mig.sh --mig 6 /scratch/11098/sambieberich/hanning_namd/stmv.namd test6.out
```

Current code puts the job in the right place, but the CPU partitioning is not yet optimal, when you run several at the same time it does not work due to slurm.

### Multiple MIG tests (sequential)

```bash
# Each will use its own MIG slice and its own 10-core block
./non_srun.sh --mig 2 ../hanning_namd/stmv.namd test2.out
./non_srun.sh --mig 0 ../hanning_namd/stmv.namd test0.out
```

### Multiple MIG tests (parallel)

```bash
SRUN_MPI=pmi2 ./non_srun.sh --mig 0 ../hanning_namd/stmv.namd test0.out
SRUN_MPI=pmi2 ./non_srun.sh --mig 1 ../hanning_namd/stmv.namd test1.out
SRUN_MPI=pmi2 ./non_srun.sh --mig 2 ../hanning_namd/stmv.namd test2.out
SRUN_MPI=pmi2 ./non_srun.sh --mig 3 ../hanning_namd/stmv.namd test3.out
SRUN_MPI=pmi2 ./non_srun.sh --mig 4 ../hanning_namd/stmv.namd test4.out
SRUN_MPI=pmi2 ./non_srun.sh --mig 5 ../hanning_namd/stmv.namd test5.out
SRUN_MPI=pmi2 ./non_srun.sh --mig 6 ../hanning_namd/stmv.namd test6.out
```

If running with the 3-way MIG or another configuration, can adjust some parameters

```bash
MIG_SLOTS=3 TOTAL_LOGICAL_GPUS=72 SRUN_MPI=pmi2 ./non_srun.sh --mig 1 ../hanning_namd/stmv.namd test1.out
```
