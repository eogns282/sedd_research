"""
This module defines the noise schedules for the diffusion process.

The noise schedule dictates the amount of noise added at each timestep `t`
of the forward process. It is a critical component that determines the dynamics
of both noising and denoising.

The key function is `total(t)`, which represents the total accumulated noise G(t)
at a continuous time `t` from 0 to 1. This G(t) is then used to calculate the
corruption probability `p = 1 - exp(-G(t))`.
"""

import abc
import torch
import torch.nn as nn
from typing import Any

class NoiseSchedule(abc.ABC, nn.Module):
    """
    Abstract base class for noise schedules.
    
    A noise schedule defines the function G(t), representing the total noise
    level at time t.
    """
    @abc.abstractmethod
    def total(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates the total noise G(t) at time t.

        Args:
            t (torch.Tensor): A tensor of continuous time values, where t is in [0, 1].
                              Shape: [batch_size,].

        Returns:
            torch.Tensor: The total noise G(t) for each time in the input tensor.
                          Shape: [batch_size,].
        """
        pass

class GeometricNoise(NoiseSchedule):
    """
    A geometric noise schedule.

    This schedule is defined such that the total noise G(t) interpolates
    logarithmically between two endpoints, G(0) and G(1). This results in an
    exponential change in the signal-to-noise ratio, which is a common
    and effective choice for diffusion models.
    """
    def __init__(self, g_min: float, g_max: float):
        """
        Initializes the GeometricNoise schedule.

        Args:
            g_min (float): The total noise at t=0, G(0). Should be close to 0.
            g_max (float): The total noise at t=1, G(1). A larger value means
                           more noise at the end of the forward process.
        """
        super().__init__()
        self.g_min = g_min
        self.g_max = g_max

    def total(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates G(t) using logarithmic interpolation.
        
        The formula is G(t) = G(0) + (G(1) - G(0)) * t.
        In this implementation, we use g_min for G(0) and g_max for G(1).
        """
        return self.g_min + (self.g_max - self.g_min) * t

class LogLinearNoise(NoiseSchedule):
    """
    A log-linear noise schedule, as proposed in some earlier diffusion papers.
    
    This schedule is defined by G(t) = -log(1 - (1-eps)*t).
    """
    def __init__(self, eps: float = 1e-4):
        """
        Initializes the LogLinearNoise schedule.

        Args:
            eps (float): A small constant to prevent numerical instability at t=1.
        """
        super().__init__()
        self.eps = eps

    def total(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates G(t) = -log(1 - (1-eps)*t).
        
        `torch.log1p(x)` is used for numerical stability as it calculates log(1+x).
        """
        return -torch.log1p(-(1 - self.eps) * t)

def get_noise_schedule(schedule_config: Any) -> NoiseSchedule:
    """
    Factory function to create a noise schedule instance from the configuration.

    This allows for easy switching between different noise schedules by changing
    the configuration file.

    Args:
        schedule_config (Any): A configuration object that contains the necessary
                               hyperparameters for the chosen noise schedule. 
                               (e.g., config.diffusion.noise_schedule)

    Returns:
        NoiseSchedule: An instance of the specified noise schedule.
        
    Raises:
        ValueError: If the specified noise schedule is unknown.
    """
    schedule_type = schedule_config.name.lower()
    if schedule_type == "geometric":
        return GeometricNoise(schedule_config.g_min, schedule_config.g_max)
    elif schedule_type == "loglinear":
        return LogLinearNoise()
    else:
        raise ValueError(f"Unknown noise schedule: {schedule_config.name}")
