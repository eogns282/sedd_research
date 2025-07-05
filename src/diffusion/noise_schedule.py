import abc
import torch
import torch.nn as nn
from typing import Any

class NoiseSchedule(abc.ABC, nn.Module):
    """
    Abstract base class for noise schedules.
    A noise schedule defines the amount of noise at each timestep.
    """
    @abc.abstractmethod
    def rate(self, t: torch.Tensor) -> torch.Tensor:
        """The rate of change of noise, g(t)."""
        pass

    @abc.abstractmethod
    def total(self, t: torch.Tensor) -> torch.Tensor:
        """The total noise at time t, G(t) = integral of g(s)ds from 0 to t."""
        pass

class GeometricNoise(NoiseSchedule):
    """Geometric noise schedule."""
    def __init__(self, sigma_min: float, sigma_max: float):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def rate(self, t: torch.Tensor) -> torch.Tensor:
        """g(t) = d/dt [sigma_min * (sigma_max/sigma_min)^t]"""
        return torch.full_like(t, torch.log(torch.tensor(self.sigma_max)) - torch.log(torch.tensor(self.sigma_min))) * \
               (self.sigma_min * (self.sigma_max / self.sigma_min) ** t)

    def total(self, t: torch.Tensor) -> torch.Tensor:
        """G(t) = sigma_min * (sigma_max/sigma_min)^t"""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

class LogLinearNoise(NoiseSchedule):
    """Log-linear noise schedule."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def rate(self, t: torch.Tensor) -> torch.Tensor:
        """g(t) = d/dt [-log(1 - (1-eps)*t)]"""
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total(self, t: torch.Tensor) -> torch.Tensor:
        """G(t) = -log(1 - (1-eps)*t)"""
        return -torch.log1p(-(1 - self.eps) * t)

def get_noise_schedule(config: Any) -> NoiseSchedule:
    """
    Factory function to get a noise schedule based on the configuration.
    """
    if config.NOISE_SCHEDULE == "geometric":
        return GeometricNoise(config.SIGMA_MIN, config.SIGMA_MAX)
    elif config.NOISE_SCHEDULE == "loglinear":
        return LogLinearNoise()
    else:
        raise ValueError(f"Unknown noise schedule: {config.NOISE_SCHEDULE}")