import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple
from .noise_schedule import NoiseSchedule
from .graph import UniformGraph

class DiffusionProcess:
    """
    Manages the forward (noising) and reverse (sampling) diffusion processes.
    """
    def __init__(self, noise_schedule: NoiseSchedule, graph: UniformGraph, config: Any):
        """
        Initializes the diffusion process.

        Args:
            noise_schedule (NoiseSchedule): The noise schedule to use.
            graph (UniformGraph): The discrete state graph.
            config (Any): The configuration object.
        """
        self.noise_schedule = noise_schedule
        self.graph = graph
        self.num_timesteps = config.NUM_TIMESTEPS
        self.mask_token_id = 103  # [MASK] token id for bert-base-uncased

    def add_noise(self, original_tokens: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Corrupts the tokens using a mask-based approach, which is a simplified
        version of the SEDD forward process.

        Args:
            original_tokens (torch.Tensor): The original tokens x_0. Shape: [batch_size, seq_len].
            t (torch.Tensor): The continuous time t (from 0 to 1). Shape: [batch_size,].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the noisy tokens and the original tokens.
        """
        total_noise = self.noise_schedule.total(t)
        corruption_prob = 1.0 - torch.exp(-total_noise)
        corruption_prob = corruption_prob.view(-1, 1)

        corruption_mask = torch.rand_like(original_tokens.float()) < corruption_prob
        
        # Don't corrupt special tokens ([CLS], [SEP], [PAD])
        corruption_mask[:, 0] = False
        sep_indices = (original_tokens == 102).max(dim=1).indices
        for i in range(original_tokens.size(0)):
            corruption_mask[i, sep_indices[i]:] = False

        noisy_tokens = original_tokens.clone()
        noisy_tokens[corruption_mask] = self.mask_token_id
        
        return noisy_tokens, original_tokens

    @torch.no_grad()
    def remove_noise(self, model: nn.Module, noisy_tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Denoises the tokens for one timestep using the model's prediction and sampling.

        Args:
            model (nn.Module): The Transformer model.
            noisy_tokens (torch.Tensor): The noisy tokens at the current timestep. Shape: [batch_size, seq_len].
            t (torch.Tensor): The current timestep. Shape: [batch_size,].

        Returns:
            torch.Tensor: The denoised tokens for the next timestep. Shape: [batch_size, seq_len].
        """
        model.eval()
        
        predicted_logits = model(noisy_tokens, t)
        predicted_probs = F.softmax(predicted_logits, dim=-1)
        
        # Sample from the predicted distribution
        predicted_tokens = torch.multinomial(predicted_probs.view(-1, predicted_probs.size(-1)), 1)
        predicted_tokens = predicted_tokens.view(noisy_tokens.size())

        # Only replace the [MASK] tokens
        mask = (noisy_tokens == self.mask_token_id)
        denoised_tokens = noisy_tokens.clone()
        denoised_tokens[mask] = predicted_tokens[mask]
        
        return denoised_tokens