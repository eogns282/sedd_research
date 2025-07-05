import torch
import torch.nn.functional as F
from noise_lib import GeometricNoise

class DiscreteDiffusion:
    def __init__(self, num_timesteps=1000, mask_token_id=4, device='cpu'):
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id
        self.device = device
        self.noise_schedule = GeometricNoise(sigma_min=1e-3, sigma_max=1.0, device=device)

    def add_noise(self, original_tokens, t):
        """
        Forward process: Corrupts the tokens by replacing them with [MASK].
        """
        total_noise, _ = self.noise_schedule(t / self.num_timesteps)
        corruption_prob = 1.0 - torch.exp(-total_noise)
        corruption_prob = corruption_prob.view(-1, 1)

        corruption_mask = torch.rand_like(original_tokens.float()) < corruption_prob
        
        corruption_mask[:, 0] = False # Don't corrupt [CLS]
        # Find the index of the first [SEP] token for each sequence
        sep_indices = (original_tokens == 102).max(dim=1).indices
        for i in range(original_tokens.size(0)):
            corruption_mask[i, sep_indices[i]:] = False # Don't corrupt [SEP] and after

        noisy_tokens = original_tokens.clone()
        noisy_tokens[corruption_mask] = self.mask_token_id
        
        return noisy_tokens, original_tokens

    @torch.no_grad()
    def remove_noise(self, model, noisy_tokens, t):
        """
        Reverse process: Denoise the tokens for one timestep using sampling.
        """
        model.eval()
        
        predicted_logits = model(noisy_tokens, t)
        
        # Use the full probability distribution to sample the next tokens
        predicted_probs = F.softmax(predicted_logits, dim=-1)
        
        # For this enhanced PoC, we will sample from the distribution
        # instead of just taking the argmax.
        predicted_tokens = torch.multinomial(predicted_probs.view(-1, predicted_probs.size(-1)), num_samples=1)
        predicted_tokens = predicted_tokens.view(noisy_tokens.size())

        # Only replace the [MASK] tokens
        mask = (noisy_tokens == self.mask_token_id)
        denoised_tokens = noisy_tokens.clone()
        denoised_tokens[mask] = predicted_tokens[mask]
        
        return denoised_tokens
