#!/bin/bash

# This script handles the setup and execution of the generation process.

# Exit immediately if a command exits with a non-zero status.
set -e

# Find the base directory of the conda installation and source the conda shell functions
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the correct conda environment
conda activate text-diffusion-poc

# Add the src directory to the Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the generation script
echo "Starting text generation..."
python src/generate.py