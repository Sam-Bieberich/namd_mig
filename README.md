# NAMD + NVIDIA MIG Testing Repository

This repository is for testing NAMD (Nanoscale Molecular Dynamics) using NVIDIA MIG (Multi-Instance GPU) partitions.

Purpose
- Provide reproducible configurations, scripts, and notes for running NAMD benchmarks and simulations on systems that use NVIDIA GPUs configured with MIG partitions.

What you'll find here
- Scripts and configuration files for launching NAMD workloads (placeholders or actual scripts may be present).
- Documentation and notes about MIG partitioning, GPU allocation, and any observed performance characteristics.

Usage notes
- Ensure your system has NVIDIA GPUs supporting MIG (A100, H100, etc.) and that MIG is enabled via the NVIDIA driver and NVIDIA-smi.
- Allocate appropriate MIG instances for the workload before running NAMD.
- Adjust NAMD configuration and node allocation to match the number and size of MIG instances.

Example commands
- The exact scripts and commands depend on the contents of this repository and your cluster setup. Typical steps include:
  1. Configure MIG instances with `nvidia-smi mig -cgi ...` or through your cluster management tools.
  2. Verify MIG instances with `nvidia-smi --query-gpu=name,gpu_uuid,mig.mode.current --format=csv`.
  3. Launch NAMD with the desired MPI runtime and point to the MIG devices allocated to your job.

Contributing
- Please open issues or pull requests with additional scripts, configs, or notes about running NAMD with MIG.

License & Contact
- Add a license or contact information as appropriate for your project.
