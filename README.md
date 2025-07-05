# SEDD Research Baseline

This repository contains a minimal, research-ready baseline for the paper "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (SEDD). The goal of this codebase is to provide a clean, modular, and extensible starting point for research on discrete diffusion models for language.

## Project Structure

The project is organized into the following directories and files:

```
/
|
├── configs/
│   └── base_config.py      # All hyperparameters and settings for experiments.
│
├── data.py                 # Handles dataset loading (WikiText-2) and preprocessing.
│
├── model.py                # The Transformer model architecture.
│
├── diffusion/
│   ├── __init__.py
│   ├── graph.py            # Implements the discrete state graph and transition matrices.
│   ├── noise_schedule.py   # Defines the noise schedules (Geometric, LogLinear).
│   └── diffusion_process.py# Manages the forward (noising) and reverse (sampling) processes.
│
├── losses.py               # Implements the Score-Entropy loss function.
│
├── trainer.py              # The main training and evaluation script.
│
├── generate.py             # A script to generate samples from a saved model.
│
└── requirements.txt        # The required Python libraries.
```

## Setup

1.  **Create and activate a conda environment:**
    ```bash
    conda create -n sedd-research python=3.9 -y
    conda activate sedd-research
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### Training

To start a training run, simply execute the `trainer.py` script:

```bash
python trainer.py
```

The script will:
1.  Load the configuration from `configs/base_config.py`.
2.  Initialize the dataset, model, and diffusion components.
3.  Run the training loop, periodically logging the training loss.
4.  After each epoch, it will evaluate the model on the validation set and print the validation loss.
5.  Save model checkpoints to the `checkpoints/` directory at the interval specified in the config.

### Generation

To generate text from a trained model, run the `generate.py` script:

```bash
python generate.py
```

This script will:
1.  Load the latest checkpoint from the `checkpoints/` directory.
2.  Initialize the model and diffusion process.
3.  Generate a new text sample using the reverse diffusion process.
4.  Print the generated text to the console.

## Extending the Codebase for Research

This baseline is designed to be easily extensible. Here are some ideas for how you can build upon it for your own research:

*   **Implement the full Score-Entropy loss:** The current `losses.py` uses a simplified version of the Score-Entropy loss. You can implement the full loss function from the paper to get a more faithful reproduction of the SEDD model.
*   **Experiment with different model architectures:** Modify `model.py` to try different Transformer architectures, such as adding more layers, using different attention mechanisms, or exploring entirely new model families.
*   **Design new noise schedules:** Add new noise schedules to `diffusion/noise_schedule.py` and see how they affect the model's performance.
*   **Explore different graph structures:** The current `diffusion/graph.py` uses a simple uniform transition. You can implement other graph structures, such as a masking-based graph or a graph based on word embeddings, to better capture the relationships between tokens.
*   **Train on larger datasets:** Modify `data.py` to use different datasets from the Hugging Face `datasets` library.

Good luck with your research!
