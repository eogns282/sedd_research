#!/bin/bash

# This script handles the setup and execution of the perplexity evaluation.

# Exit immediately if a command exits with a non-zero status.
set -e

# Find the base directory of the conda installation and source the conda shell functions
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the correct conda environment
conda activate text-diffusion-poc

# Add the src directory to the Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the evaluation script
# You can optionally pass an experiment ID as the first argument
# e.g., ./run_perplexity_eval.sh 20250705_180000
echo "Starting perplexity evaluation..."
python evaluate.py "$@"