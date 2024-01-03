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

# Para comprobar que funciona no es necesario usar matrices muy grandes
# Con N = 1024 es suficiente

./mergesort.exe 10
#./kernel4GPUs.exe 2048 N
#./kernel4GPUs.exe 4096 N
#./kernel4GPUs.exe 8192 N

nsys nvprof --print-gpu-trace ./mergesort.exe 20

#nsys nvprof --print-gpu-trace ./kernel4GPUs.exe 1024 N
