# Score-Entropy Discrete Diffusion (SEDD) Research Framework

This repository provides a research-grade framework for experimenting with Score-Entropy Discrete Diffusion models, based on the paper ["Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution"](https://arxiv.org/abs/2310.16834).

The framework is designed to be clear, extensible, and easy to use for comparing different diffusion strategies and for exploring new ideas in discrete diffusion. After a successful research arc, this repository now contains a novel, state-of-the-art uniform-based diffusion model that outperforms the standard absorbing state baseline.

## Project Structure

```
/
├─── checkpoints/      # Saved model checkpoints
├─── analysis_results/ # Detailed analysis outputs (metrics, samples)
├─── src/
│   ├─── data.py         # Data loading and preprocessing
│   ├─── diffusion/      # Core diffusion logic (process, graphs, schedules)
│   ├─── model.py        # Transformer model architecture
│   ├─── utils/
│   │   └─── config_loader.py # Handles loading YAML configs
│   ├─── losses.py       # Training loss function
│   └─── trainer.py      # Main training script
├─── freq_hybrid_config.yaml # Configuration for the new SOTA model
├─── run_freq_hybrid_experiment.sh # Script to reproduce the SOTA result
└─── requirements.txt  # Python dependencies
```

## Setup

1.  **Create and activate a conda environment:**
    ```bash
    conda create -n text-diffusion-poc python=3.9
    conda activate text-diffusion-poc
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run Experiments

This framework is configured using `.yaml` files. The final, state-of-the-art model is the **Frequency-Aware Hybrid Model**.

### Reproducing the Final Results

To train and evaluate the final, best-performing model from scratch, use the `run_freq_hybrid_experiment.sh` script. This will train the model, run a full analysis, and save the results.
```bash
./run_freq_hybrid_experiment.sh
```
This script uses the `freq_hybrid_config.yaml` file. The final results will be located in `analysis_results/final_freq_hybrid/`.

## Core Concepts

### Diffusion Process

The core of the model is the `DiffusionProcess` defined in `src/diffusion/diffusion_process.py`. It uses a `Graph` object to define the transition probabilities between states.

### Graphs

The `Graph` objects in `src/diffusion/graph.py` define the noising strategy. The key innovation of this project is the **`FrequencyHybridGraph`**, which combines two successful techniques:
-   **Hybrid Noise:** A small fraction (10%) of noise is the `[MASK]` token, providing a stable anchor for the model.
-   **Frequency-Aware Noise:** The remaining 90% of noise is not purely uniform, but is sampled from the real-world frequency distribution of the training data, providing a more plausible and less destructive form of corruption.

### Model

The denoising model is a standard Transformer architecture, defined in `src/model.py`. It is trained to predict the original, uncorrupted token given a noised input.
