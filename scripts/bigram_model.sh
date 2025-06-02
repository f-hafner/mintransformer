#!/bin/bash
#
#SBATCH --job-name=bigram
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --time=00:05:00
#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out
#SBATCH --partition=rome
#SBATCH --mem=10G

source slurm/load_venv.sh

python scripts/bigram_model.py
