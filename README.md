# SEDD Research

This repository contains a minimal, research-ready baseline for the paper "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (SEDD). The goal of this codebase is to provide a clean, modular, and extensible starting point for research on discrete diffusion models for language.

## Project Structure

The project is organized into the following directories and files:

```
/
|
├── src/
│   ├── configs/
│   │   └── base_config.py      # All hyperparameters and settings for experiments.
│   │
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── graph.py            # Implements the discrete state graph and transition matrices.
│   │   ├── noise_schedule.py   # Defines the noise schedules (Geometric, LogLinear).
│   │   └── diffusion_process.py# Manages the forward (noising) and reverse (sampling) processes.
│   │
│   ├── data.py                 # Handles dataset loading (WikiText-2) and preprocessing.
│   ├── generate.py             # A script to generate samples from a saved model.
│   ├── losses.py               # Implements the Score-Entropy loss function.
│   ├── model.py                # The Transformer model architecture.
│   └── trainer.py              # The main training and evaluation script.
│
├── requirements.txt        # The required Python libraries.
├── run_train.sh            # A shell script to simplify the training process.
└── setup.py                # The setup script for installing the project as a package.
```

## Setup

1.  **Create and activate a conda environment:**
    ```bash
    conda create -n text-diffusion-poc python=3.9 -y
    conda activate text-diffusion-poc
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the project in editable mode:**
    ```bash
    pip install -e .
    ```

## How to Run

### Training

To start a training run, use the `run_train.sh` script. You can specify the GPU to use by passing the `--gpu` argument. For example, to use GPU 1:

```bash
./run_train.sh --gpu 1
```

The script will:
1.  Activate the `text-diffusion-poc` conda environment.
2.  Run the training script located at `src/trainer.py`.
3.  Load the configuration from `src/configs/base_config.py`.
4.  Initialize the dataset, model, and diffusion components.
5.  Run the training loop, logging the training loss to `wandb`.
6.  After each epoch, it will evaluate the model on the validation set and log the validation loss.
7.  Save model checkpoints to the `checkpoints/` directory.

### Generation

To generate text from a trained model, run the `generate.py` script:

```bash
python src/generate.py
```

This script will:
1.  Load the latest checkpoint from the `checkpoints/` directory.
2.  Initialize the model and diffusion process.
3.  Generate a new text sample using the reverse diffusion process.
4.  Print the generated text to the console.

## Extending the Codebase for Research

This baseline is designed to be easily extensible. Here are some ideas for how you can build upon it for your own research:

*   **Implement the full Score-Entropy loss:** The current `losses.py` uses a simplified version of the Score-Entropy loss. You can implement the full loss function from the paper to get a more faithful reproduction of the SEDD model.
*   **Experiment with different model architectures:** Modify `src/model.py` to try different Transformer architectures, such as adding more layers, using different attention mechanisms, or exploring entirely new model families.
*   **Design new noise schedules:** Add new noise schedules to `src/diffusion/noise_schedule.py` and see how they affect the model's performance.
*   **Explore different graph structures:** The current `src/diffusion/graph.py` uses a simple uniform transition. You can implement other graph structures, such as a masking-based graph or a graph based on word embeddings, to better capture the relationships between tokens.
*   **Train on larger datasets:** Modify `src/data.py` to use different datasets from the Hugging Face `datasets` library.