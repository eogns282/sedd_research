import abc
import torch
import torch.nn as nn

class Noise(abc.ABC, nn.Module):
    """
    Abstract base class for noise schedules.
    """
    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t):
        """
        Rate of change of noise, i.e., g(t).
        """
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """
        Total noise, i.e., G(t) = integral from 0 to t of g(s)ds.
        """
        pass

class GeometricNoise(Noise, nn.Module):
    """
    Geometric noise schedule from the official SEDD repository.
    """
    def __init__(self, sigma_min=1e-3, sigma_max=1.0, device='cpu'):
        super().__init__()
        self.sigmas = torch.tensor([sigma_min, sigma_max]).to(device)
        self.device = device

    def rate_noise(self, t):
        """g(t) = (sigma_max/sigma_min)^t * log(sigma_max/sigma_min)"""
        log_ratio = torch.log(self.sigmas[1] / self.sigmas[0])
        return self.sigmas[0] * torch.pow(self.sigmas[1] / self.sigmas[0], t) * log_ratio

    def total_noise(self, t):
        """G(t) = sigma_min * (sigma_max/sigma_min)^t"""
        return self.sigmas[0] * torch.pow(self.sigmas[1] / self.sigmas[0], t)
