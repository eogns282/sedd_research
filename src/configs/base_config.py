# --- Base Configuration for SEDD Research Baseline ---
# This file contains all the hyperparameters and settings for the experiments.
# Modifying this file is the primary way to configure different training runs.

# --- Dataset Configuration ---
# The name of the dataset to use from the Hugging Face `datasets` library.
DATASET_NAME: str = "wikitext"
# The specific configuration of the dataset (e.g., 'wikitext-2-raw-v1').
DATASET_CONFIG: str = "wikitext-2-raw-v1"
# The size of the vocabulary. For BERT, this is typically 30522.
VOCAB_SIZE: int = 30522
# The maximum sequence length for the model. Sentences will be padded or truncated to this length.
MAX_SEQ_LEN: int = 128

# --- Model Architecture ---
# The dimensionality of the model's embeddings and hidden states.
D_MODEL: int = 256
# The number of attention heads in the Transformer's multi-head attention layers.
N_HEAD: int = 8
# The number of Transformer encoder layers.
NUM_ENCODER_LAYERS: int = 6
# The dimensionality of the feed-forward network inside the Transformer encoder.
DIM_FEEDFORWARD: int = 1024
# The dropout rate used in the model.
DROPOUT: float = 0.1

# --- Diffusion Process ---
# The type of graph to use for the diffusion process. Options: "uniform", "absorbing".
GRAPH_TYPE: str = "uniform"
# The token ID to use for the absorbing state. Typically, this is a [MASK] token.
# For BERT, the [MASK] token ID is 103.
MASK_TOKEN_ID: int = 103
# The total number of diffusion timesteps.
NUM_TIMESTEPS: int = 1000
# The type of noise schedule to use. Options: "geometric", "loglinear".
NOISE_SCHEDULE: str = "geometric"
# The minimum noise level (sigma) for the geometric schedule.
SIGMA_MIN: float = 1e-3
# The maximum noise level (sigma) for the geometric schedule.
SIGMA_MAX: float = 1.0

# --- Training ---
# The number of samples per batch.
BATCH_SIZE: int = 32
# The learning rate for the Adam optimizer.
LEARNING_RATE: float = 1e-4
# The total number of training epochs.
NUM_EPOCHS: int = 10
# The maximum gradient norm for gradient clipping.
GRAD_CLIP: float = 1.0
# The number of warmup steps for the learning rate scheduler.
WARMUP_STEPS: int = 1000

# --- Logging and Saving ---
# The interval (in steps) at which to log training progress.
LOG_INTERVAL: int = 100
# The interval (in epochs) at which to save a model checkpoint.
SAVE_INTERVAL: int = 1
# The directory where model checkpoints will be saved.
CHECKPOINT_DIR: str = "checkpoints"