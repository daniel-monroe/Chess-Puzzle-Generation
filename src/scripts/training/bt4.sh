#!/bin/bash -l
# Slurm wrapper: BT4 finetune vs. the original MaskedDiffusion model on one 8-GPU node (4 GPUs each).
# Run setup_b300.sh once on the node (or a shared filesystem) first. Knobs: see launch_bt4.sh.
#SBATCH --job-name=BT4Diffusion
#SBATCH --time=3-00:00:00
#SBATCH --output=bt4_%j.out
#SBATCH --mem=512G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=16
#SBATCH --constraint="b300|h200"

WAIT=1 bash launch_bt4.sh
