import torch
from typing import Any, Callable
from src.diffusion.diffusion_process import DiffusionProcess
from src.model import TransformerModel

def get_loss_fn(config: Any) -> Callable[[TransformerModel, torch.Tensor, DiffusionProcess], torch.Tensor]:
    """
    Returns the loss function for training the SEDD model.
    This function is a factory that creates and returns the actual loss function.
    """
    def loss_fn(model: TransformerModel, batch: torch.Tensor, diffusion_process: DiffusionProcess) -> torch.Tensor:
        """
        Calculates the loss for a given batch.

        Args:
            model (TransformerModel): The Transformer model.
            batch (torch.Tensor): A batch of original data (token IDs). Shape: [batch_size, seq_len].
            diffusion_process (DiffusionProcess): The diffusion process object.

        Returns:
            torch.Tensor: The mean loss for the batch.
        """
        # 1. Sample a random continuous time t from (0, 1]
        t = torch.rand(batch.shape[0], device=batch.device) * (1 - 1e-5) + 1e-5
        
        # 2. Add noise to the batch to get x_t
        noisy_batch, _ = diffusion_process.add_noise(batch, t)
        
        # 3. Convert continuous time t to discrete timesteps for the model's embedding layer
        timesteps = (t * config.NUM_TIMESTEPS).long()
        
        # 4. Get the model's prediction (logits) for the original tokens
        logits = model(noisy_batch, timesteps)
        
        # 5. Get the noise level (sigma) for the loss calculation
        sigma = diffusion_process.noise_schedule.total(t)
        
        # 6. Calculate the score entropy loss using the graph from the diffusion process
        loss = diffusion_process.graph.score_entropy(logits, sigma, noisy_batch, batch)
        
        return loss.mean()

    return loss_fn
