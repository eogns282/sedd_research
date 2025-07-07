import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple, Union
from .noise_schedule import NoiseSchedule
from .graph import UniformGraph, AbsorbingGraph

class DiffusionProcess:
    """
    Manages the forward (noising) and reverse (sampling) diffusion processes.
    """
    def __init__(self, noise_schedule: NoiseSchedule, graph: Union[UniformGraph, AbsorbingGraph], config: Any):
        """
        Initializes the diffusion process.

        Args:
            noise_schedule (NoiseSchedule): The noise schedule to use.
            graph (Union[UniformGraph, AbsorbingGraph]): The discrete state graph.
            config (Any): The configuration object.
        """
        self.noise_schedule = noise_schedule
        self.graph = graph
        self.num_timesteps = config.NUM_TIMESTEPS
        self.mask_token_id = config.MASK_TOKEN_ID

    def add_noise(self, original_tokens: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Corrupts the tokens using the graph's transition sampler.

        Args:
            original_tokens (torch.Tensor): The original tokens x_0. Shape: [batch_size, seq_len].
            t (torch.Tensor): The continuous time t (from 0 to 1). Shape: [batch_size,].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the noisy tokens and the original tokens.
        """
        total_noise = self.noise_schedule.total(t)
        noisy_tokens = self.graph.sample_transition(original_tokens, total_noise)
        
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

        # In the absorbing case, we only replace the [MASK] tokens.
        # For the uniform case, this logic might need to be adjusted if we want to be more selective.
        # However, for a general implementation, we can replace all tokens based on the model's prediction.
        if isinstance(self.graph, AbsorbingGraph):
            mask = (noisy_tokens == self.mask_token_id)
            denoised_tokens = noisy_tokens.clone()
            denoised_tokens[mask] = predicted_tokens[mask]
        else:
            # For the uniform graph, every token is potentially changed, so we replace all of them.
            denoised_tokens = predicted_tokens
        
        return denoised_tokens