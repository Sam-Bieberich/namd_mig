# Workflow

The following command runs NAMD with MIG level 6 using the provided configuration file and outputs the results to `test6.out`.

```bash
bash run_namd_mig.sh --mig 6 hanning_namd/stmv.namd test6.out
```

Current code puts the job in the right place, but the CPU partitioning is not yet optimal. 

Also when you run several at the same time it does not work
