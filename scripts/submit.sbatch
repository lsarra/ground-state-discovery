#!/bin/bash -l
# Script to submit any job with a given config file.

#SBATCH -J "quantum_decomposition"
#SBATCH -C rome
#SBATCH --output=logs/job-%j_output.log
## SBATCH --error=logs/job-%j_error.log
#SBATCH -t 168:0:0
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1000GB

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

# enable logging
export  CUDA_LAUNCH_BLOCKING=1.

conda activate sun
$@