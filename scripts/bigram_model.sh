#!/bin/bash
#
#SBATCH --job-name=bigram
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --partition=gpu_a100
#SBATCH --gpus 2
#SBATCH --time=00:10:00
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out
#SBATCH --mem=10G

source slurm/load_venv.sh

# option "gpu" uses all available gpus; otherwise specify number 
torchrun --standalone --nproc_per_node=gpu scripts/bigram_model.py
