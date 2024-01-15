#!/bin/bash

#SBATCH --job-name=MultiGPU
#SBATCH -D .
#SBATCH --output=submit-MultiGPU.o%j
#SBATCH --error=submit-MultiGPU.e%j
#SBATCH -A cuda
#SBATCH -p cuda
### Se piden 4 GPUs
#SBATCH --gres=gpu:4

export PATH=/Soft/cuda/12.2.2/bin:$PATH

./mergesort.exe 10

