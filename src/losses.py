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
import wandb
from typing import Any, Callable, Tuple

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
    
    def loss_fn(model: TransformerModel, batch: torch.Tensor, diffusion_process: DiffusionProcess, analysis_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
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
            analysis_mode (bool): If True, skips logging to wandb.

        Returns:
            A tuple containing:
            - torch.Tensor: A single scalar tensor representing the mean loss for the batch.
            - torch.Tensor: The corruption mask used for this batch.
        """
        # --- Step 1: Sample a random timestep for each sequence in the batch ---
        t = torch.rand(batch.shape[0], device=batch.device) * (1 - 1e-5) + 1e-5

        # --- Step 2: Create the noisy input (x_t) ---
        noisy_batch, _, corruption_mask = diffusion_process.add_noise(batch, t)

        # --- Step 3: Get the model's prediction ---
        context_mask = torch.rand(batch.shape[0], device=batch.device) > config.training.unconditional_prob
        
        model_output = model(
            noisy_batch, 
            t, 
            context_mask=context_mask, 
            corruption_mask=corruption_mask
        )

        # --- Step 4: Calculate the loss ---
        use_gated_attention = getattr(config.model, 'use_gated_attention', False)
        if use_gated_attention:
            predicted_logits, gate_scores = model_output
        else:
            predicted_logits = model_output

        sigma = diffusion_process.noise_schedule.total(t)
        loss_per_token = diffusion_process.graph.score_entropy(
            score=predicted_logits,
            sigma=sigma,
            x_t=noisy_batch,
            x_0=batch
        )

        # --- Step 5: Calculate and Log Component Losses ---
        if corruption_mask is not None and not analysis_mode:
            loss_corrupted = (loss_per_token * corruption_mask).sum() / corruption_mask.sum()
            loss_uncorrupted = (loss_per_token * ~corruption_mask).sum() / (~corruption_mask).sum()
            
            wandb.log({
                "loss_corrupted": loss_corrupted.item(),
                "loss_uncorrupted": loss_uncorrupted.item()
            })

        # --- Step 6: Return the final mean loss ---
        final_loss = loss_per_token.mean()

        loss_weight = getattr(config.training, 'loss_corrupted_weight', 1.0)
        if loss_weight > 1.0 and corruption_mask is not None:
            weights = torch.ones_like(loss_per_token)
            weights[corruption_mask] = loss_weight
            final_loss = (loss_per_token * weights).mean()

        if use_gated_attention and corruption_mask is not None:
            gate_logits = gate_scores
            
            use_soft_gate_loss = getattr(config.training, 'use_soft_gate_loss', False)
            if use_soft_gate_loss:
                gate_targets = torch.ones_like(corruption_mask, dtype=torch.float)
                uniform_noise_mask = corruption_mask & (noisy_batch != config.vocab.mask_token_id)
                
                gate_targets[noisy_batch == config.vocab.mask_token_id] = 0.0
                gate_targets[uniform_noise_mask] = config.training.gate_loss_uniform_target
            else:
                gate_targets = corruption_mask.float()

            gate_loss_fn = torch.nn.BCEWithLogitsLoss()
            gate_loss = gate_loss_fn(gate_logits, gate_targets)
            
            if not analysis_mode:
                wandb.log({"gate_loss": gate_loss.item()})
            
            gate_weight = getattr(config.training, 'gate_loss_weight', 0.1)
            final_loss = final_loss + gate_weight * gate_loss

        return final_loss, corruption_mask

    return loss_fn