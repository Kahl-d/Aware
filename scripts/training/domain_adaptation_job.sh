#!/bin/bash
#SBATCH --job-name=mlm_adapt_new
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1                    # Requesting 2 GPUs
#SBATCH --cpus-per-task=8               # Increased CPUs for better data loading
#SBATCH --time=07:55:00                 # Added a time limit just under 8 hours
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kkhan@sfsu.edu

# Load environment
source ../../miniconda/bin/activate llm_env

# Fix CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# (Optional) Print GPU info
nvidia-smi

# --- THIS IS THE CRITICAL CHANGE ---
# Run the distributed training using accelerate launch
# This command uses both GPUs you requested.
python3 test.py

# Deactivate
conda deactivate