"""
This module defines the loss function for training the diffusion model.

The core idea is to train the model to reverse the noising process. This is
achieved by:
1. Taking a clean data sample (x_0).
2. Adding a random amount of noise to get a corrupted sample (x_t).
3. Feeding the noisy sample (x_t) and the timestep (t) to the model.
4. Asking the model to predict the original clean sample (x_0).
5. Calculating a loss based on how well the model's prediction matches the
   original data.
"""

import torch
from typing import Any, Callable

from src.diffusion.diffusion_process import DiffusionProcess
from src.model import TransformerModel

def get_loss_fn(config: Any) -> Callable[[TransformerModel, torch.Tensor, DiffusionProcess], torch.Tensor]:
    """
    A factory function that creates and returns the loss function for training.

    Using a factory pattern allows for potential future flexibility, such as
    returning different loss functions based on the configuration.

    Args:
        config (Any): The global configuration object.

    Returns:
        Callable: The actual loss function to be used in the training loop.
    """
    
    def loss_fn(model: TransformerModel, batch: torch.Tensor, diffusion_process: DiffusionProcess) -> torch.Tensor:
        """
        Calculates the training loss for a single batch of data.

        This function encapsulates the core logic of a single training step for
        a diffusion model.

        Args:
            model (TransformerModel): The model being trained.
            batch (torch.Tensor): A batch of clean, original token sequences (x_0).
                                  Shape: [batch_size, seq_len].
            diffusion_process (DiffusionProcess): The diffusion process manager which
                                                  handles the noising.

        Returns:
            torch.Tensor: A single scalar tensor representing the mean loss for the batch,
                          ready for backpropagation.
        """
        # --- Step 1: Sample a random timestep for each sequence in the batch ---
        # We sample from a continuous time t in (0, 1]. A small epsilon (1e-5) is
        # used to avoid sampling t=0, where there is no noise.
        t = torch.rand(batch.shape[0], device=batch.device) * (1 - 1e-5) + 1e-5

        # --- Step 2: Create the noisy input (x_t) ---
        # Use the diffusion process to add the appropriate amount of noise for the
        # sampled timesteps `t`.
        noisy_batch, _ = diffusion_process.add_noise(batch, t)

        # --- Step 3: Get the model's prediction ---
        # The model takes the noisy batch and the continuous time `t` as input.
        # It's trained to predict the logits corresponding to the *original* clean batch.
        
        # For Classifier-Free Guidance, we randomly drop the context (the noisy input)
        # with a certain probability and train the model on an unconditional objective.
        context_mask = torch.rand(batch.shape[0], device=batch.device) > config.training.unconditional_prob
        predicted_logits = model(noisy_batch, t, context_mask=context_mask)

        # --- Step 4: Calculate the loss ---
        # We delegate the final loss calculation to the graph object. In our baseline,
        # this is a simple cross-entropy loss, which is a standard and effective
        # objective for training a model to predict the original tokens.
        # The `sigma` and `noisy_batch` are passed for API consistency, even if
        # unused by the simplified loss.
        sigma = diffusion_process.noise_schedule.total(t)
        loss_per_token = diffusion_process.graph.score_entropy(
            score=predicted_logits,
            sigma=sigma,
            x_t=noisy_batch,
            x_0=batch
        )

        # --- Step 5: Return the final mean loss ---
        # We take the mean of the loss across all tokens and all sequences in the batch.
        return loss_per_token.mean()

    return loss_fn