#!/bin/bash

# This script handles the setup and execution of the training process.

# Exit immediately if a command exits with a non-zero status.
set -e

# Find the base directory of the conda installation and source the conda shell functions
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the correct conda environment
conda activate text-diffusion-poc

# Run the training script as a module, passing all command-line arguments to it
# This allows you to specify the GPU, e.g., ./run_train.sh --gpu 1
echo "Starting SEDD training..."
python -m src.trainer "$@"