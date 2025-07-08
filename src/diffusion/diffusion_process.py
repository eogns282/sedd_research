"""
This module defines the DiffusionProcess class, which orchestrates the
forward (noising) and reverse (denoising) processes.

It acts as a high-level controller that uses a NoiseSchedule and a Graph
to perform the core operations of the diffusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple, Union

from .noise_schedule import NoiseSchedule
from .graph import UniformGraph, AbsorbingGraph

class DiffusionProcess:
    """
    Manages the forward and reverse diffusion processes.

    This class brings together the noise schedule and the state transition graph
    to implement the two key operations of a diffusion model:
    1.  `add_noise`: The forward process (q(x_t | x_0)), which gradually corrupts
        clean data into noise. This is used during training.
    2.  `remove_noise`: The reverse process (p(x_{t-1} | x_t)), which uses a
        trained model to iteratively denoise data, starting from pure noise
        to generate a clean sample. This is used for sampling/generation.
    """
    def __init__(self, noise_schedule: NoiseSchedule, graph: Union[UniformGraph, AbsorbingGraph], diffusion_config: Any, vocab_config: Any):
        """
        Initializes the DiffusionProcess.

        Args:
            noise_schedule (NoiseSchedule): An instance of a noise schedule class.
            graph (Union[UniformGraph, AbsorbingGraph]): An instance of a graph class.
            diffusion_config (Any): The diffusion-specific configuration object.
            vocab_config (Any): The vocabulary-specific configuration object.
        """
        self.noise_schedule = noise_schedule
        self.graph = graph
        self.num_timesteps = diffusion_config.num_timesteps
        self.mask_token_id = vocab_config.mask_token_id

    def add_noise(self, original_tokens: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward process to corrupt clean tokens into noisy tokens.

        This function takes a batch of clean sequences (x_0) and a tensor of
        timesteps (t), and returns a corrupted version (x_t) where the amount
        of corruption is determined by the noise schedule at time t.

        Args:
            original_tokens (torch.Tensor): The clean input tokens, x_0.
                                            Shape: [batch_size, seq_len].
            t (torch.Tensor): A tensor of continuous time values in [0, 1] for each
                              item in the batch. Shape: [batch_size,].

        Returns:
            A tuple containing:
            - noisy_tokens (torch.Tensor): The corrupted tokens, x_t.
                                           Shape: [batch_size, seq_len].
            - original_tokens (torch.Tensor): The original tokens, passed through for
                                              convenience in loss calculation.
                                              Shape: [batch_size, seq_len].
            - corruption_mask (torch.Tensor or None): A boolean mask indicating
                                                      which tokens were corrupted.
                                                      Shape: [batch_size, seq_len].
                                                      Returns None if the graph is not
                                                      the UniformGraph.
        """
        # 1. Get the total noise G(t) from the schedule for the given timesteps.
        total_noise = self.noise_schedule.total(t)

        # 2. Calculate the corruption probability `p = 1 - exp(-G(t))`.
        # This is the core formula linking the noise schedule to the probability
        # of a token being altered.
        corruption_prob = 1.0 - torch.exp(-total_noise)

        # 3. Use the graph's `sample_transition` method to apply the noise.
        # This delegates the actual noising logic (uniform vs. absorbing) to the
        # appropriate graph object.
        noisy_tokens, corruption_mask = self.graph.sample_transition(original_tokens, corruption_prob)
        
        return noisy_tokens, original_tokens, corruption_mask

    @torch.no_grad()
    def remove_noise(self, model: nn.Module, noisy_tokens: torch.Tensor, t: torch.Tensor, guidance_scale: float = 1.0, **kwargs) -> torch.Tensor:
        """
        Performs one step of the reverse (denoising) process with optional
        Classifier-Free Guidance (CFG).

        Args:
            model (nn.Module): The trained Transformer model.
            noisy_tokens (torch.Tensor): The noisy tokens at the current timestep, x_t.
                                         Shape: [batch_size, seq_len].
            t (torch.Tensor): The current continuous time value. Shape: [batch_size,].
            guidance_scale (float): The scale for CFG. A value of 1.0 is standard diffusion,
                                    while values > 1.0 increase the influence of the
                                    conditional prediction.
            **kwargs: Additional keyword arguments to pass to the model (e.g., draft_tokens).

        Returns:
            torch.Tensor: The partially denoised tokens for the next step, x_{t-1}.
                          Shape: [batch_size, seq_len].
        """
        model.eval()
        
        # For CFG, we make two predictions: one conditional and one unconditional
        conditional_output = model(noisy_tokens, t, context_mask=None, **kwargs)
        unconditional_output = model(noisy_tokens, t, context_mask=torch.ones_like(t).bool(), **kwargs)
        
        # Handle tuple output from gated attention model
        if isinstance(conditional_output, tuple):
            conditional_logits = conditional_output[0]
            unconditional_logits = unconditional_output[0]
        else:
            conditional_logits = conditional_output
            unconditional_logits = unconditional_output
        
        # Combine the logits using the guidance scale
        guided_logits = unconditional_logits + guidance_scale * (conditional_logits - unconditional_logits)
        
        # Convert logits to probabilities and sample
        predicted_probs = F.softmax(guided_logits, dim=-1)
        predicted_tokens = torch.multinomial(
            predicted_probs.view(-1, predicted_probs.size(-1)), 1
        ).view(noisy_tokens.size())

        # Decide which tokens to update
        if isinstance(self.graph, AbsorbingGraph):
            mask = (noisy_tokens == self.mask_token_id)
            denoised_tokens = torch.where(mask, predicted_tokens, noisy_tokens)
        else:
            denoised_tokens = predicted_tokens
        
        return denoised_tokens
