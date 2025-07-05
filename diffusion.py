import torch

class Diffusion:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = (1. - self.betas).to(device)
        self.alpha_cumprods = torch.cumprod(self.alphas, dim=0).to(device)

    def add_noise(self, original_data, t):
        """
        Forward process: Adds noise to the data.
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprods[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprods[t]).view(-1, 1, 1)
        noise = torch.randn_like(original_data)
        
        noisy_data = sqrt_alpha_cumprod * original_data + sqrt_one_minus_alpha_cumprod * noise
        return noisy_data, noise

    @torch.no_grad()
    def remove_noise(self, model, noisy_data, t):
        """
        Reverse process: Denoise the data for one timestep.
        """
        
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_cumprod_t = self.alpha_cumprods[t].view(-1, 1, 1)
        
        predicted_noise = model(noisy_data, t)
        
        # Formula from DDPM paper
        term1 = (1 / torch.sqrt(alpha_t))
        term2 = ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t))
        
        denoised_data = term1 * (noisy_data - term2 * predicted_noise)
        
        if t[0] > 0:
            noise = torch.randn_like(denoised_data)
            alpha_cumprod_prev = self.alpha_cumprods[t-1].view(-1, 1, 1)
            variance = (1. - alpha_cumprod_prev) / (1. - alpha_cumprod_t) * self.betas[t].view(-1, 1, 1)
            denoised_data += torch.sqrt(variance) * noise

        return denoised_data
