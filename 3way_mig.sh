#!/bin/bash
echo "Starting MIG setup (3 partitions)"

# checking that MIG is enabled
echo "MIG partitioning enabled check"

sudo nvidia-smi -i 0 -mig 1

echo "--------------------------"


echo "Deleting any old instances"

sudo nvidia-smi mig -dci -i 0   # Delete all compute instances
sudo nvidia-smi mig -dgi -i 0   # Delete all GPU instances

echo "MIG Profiles"

nvidia-smi mig -lgip

echo "--------------------------"


echo "Creating 3 partitions (profile id 14)"
sudo nvidia-smi mig -cgi 14,14,14 -C

echo "Confirming partitions"

nvidia-smi -L

echo "--------------------------"

# How to use this script:

# Make the script executable (if not already)
# chmod +x three_way_mig_14.sh

# Run the script
# ./three_way_mig_14.sh
