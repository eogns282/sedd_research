import torch
import torch.nn.functional as F
from typing import Tuple

class UniformGraph:
    """
    Implements a uniform discrete state graph for the diffusion process.
    In this graph, a token can transition to any other token in the vocabulary
    with a certain probability, or remain unchanged.
    """
    def __init__(self, vocab_size: int):
        """
        Initializes the graph.

        Args:
            vocab_size (int): The size of the vocabulary.
        """
        self.vocab_size = vocab_size

    def transition_matrix(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculates the transition matrix Q_t for a given noise level sigma.
        The formula is: Q_t = (1 - sigma) * I + sigma * (1/D) * 11^T
        where D is the vocabulary size, I is the identity matrix, and 11^T is a matrix of ones.

        Args:
            sigma (torch.Tensor): The noise level at time t. Shape: [batch_size,].

        Returns:
            torch.Tensor: The transition matrix Q_t. Shape: [batch_size, vocab_size, vocab_size].
        """
        I = torch.eye(self.vocab_size, device=sigma.device)
        U = torch.ones(self.vocab_size, self.vocab_size, device=sigma.device) / self.vocab_size
        return (1 - sigma.view(-1, 1, 1)) * I + sigma.view(-1, 1, 1) * U

    def sample_transition(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Samples from the forward process q(x_t | x_0) using the transition matrix.

        Args:
            x (torch.Tensor): The original tokens x_0. Shape: [batch_size, seq_len].
            sigma (torch.Tensor): The noise level at time t. Shape: [batch_size,].

        Returns:
            torch.Tensor: The noisy tokens x_t. Shape: [batch_size, seq_len].
        """
        q_t = self.transition_matrix(sigma)
        # A simplified sampling method. A more efficient approach would leverage the matrix properties.
        probs = q_t[torch.arange(x.size(0)).unsqueeze(1), x]
        return torch.multinomial(probs.view(-1, self.vocab_size), 1).view(x.size())

    def score_entropy(self, score: torch.Tensor, sigma: torch.Tensor, x_t: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Score-Entropy loss.
        For this baseline, we use a simplified version that is closely related to cross-entropy,
        but it serves as a placeholder for the full Score-Entropy loss from the paper.

        Args:
            score (torch.Tensor): The model's output logits. Shape: [batch_size, seq_len, vocab_size].
            sigma (torch.Tensor): The noise level at time t. Shape: [batch_size,].
            x_t (torch.Tensor): The noisy tokens at time t. Shape: [batch_size, seq_len].
            x_0 (torch.Tensor): The original tokens. Shape: [batch_size, seq_len].

        Returns:
            torch.Tensor: The calculated loss for each sample in the batch. Shape: [batch_size, seq_len].
        """
        # The score is log p(x_0 | x_t). We want to maximize the log-likelihood of the true data,
        # which is equivalent to minimizing the negative log-likelihood (cross-entropy).
        loss = F.cross_entropy(score.view(-1, self.vocab_size), x_0.view(-1), reduction='none')
        return loss.view(x_0.shape)