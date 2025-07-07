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
        Samples from the forward process q(x_t | x_0) efficiently without creating the full transition matrix.

        Args:
            x (torch.Tensor): The original tokens x_0. Shape: [batch_size, seq_len].
            sigma (torch.Tensor): The noise level at time t. Shape: [batch_size,].

        Returns:
            torch.Tensor: The noisy tokens x_t. Shape: [batch_size, seq_len].
        """
        bs, seq_len = x.shape
        sigma = sigma.view(bs, 1)

        # Probability of corrupting a token
        p_corrupt = sigma
        # Probability of a token staying the same
        p_identity = 1 - p_corrupt

        # Decide which tokens to corrupt
        corrupt_mask = torch.rand(bs, seq_len, device=x.device) < p_corrupt

        # Sample new tokens for the corrupted positions
        uniform_dist = torch.ones(bs, seq_len, self.vocab_size, device=x.device) / self.vocab_size
        corrupted_tokens = torch.multinomial(uniform_dist.view(-1, self.vocab_size), 1).view(bs, seq_len)

        # Combine original and corrupted tokens
        return torch.where(corrupt_mask, corrupted_tokens, x)

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


class AbsorbingGraph:
    """
    Implements an absorbing discrete state graph for the diffusion process.
    In this graph, a token can transition to a special [MASK] token or remain unchanged.
    """
    def __init__(self, vocab_size: int, mask_token_id: int):
        """
        Initializes the graph.

        Args:
            vocab_size (int): The size of the vocabulary.
            mask_token_id (int): The ID of the [MASK] token.
        """
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def transition_matrix(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculates the transition matrix Q_t for a given noise level sigma.
        The formula is: Q_t = (1 - sigma) * I + sigma * e_m 1^T
        where e_m is a one-hot vector for the [MASK] token.

        Args:
            sigma (torch.Tensor): The noise level at time t. Shape: [batch_size,].

        Returns:
            torch.Tensor: The transition matrix Q_t. Shape: [batch_size, vocab_size, vocab_size].
        """
        I = torch.eye(self.vocab_size, device=sigma.device)
        mask_row = torch.zeros(1, self.vocab_size, device=sigma.device)
        mask_row[:, self.mask_token_id] = 1
        U = mask_row.repeat(self.vocab_size, 1)
        return (1 - sigma.view(-1, 1, 1)) * I + sigma.view(-1, 1, 1) * U

    def sample_transition(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Samples from the forward process q(x_t | x_0) efficiently for the absorbing state.

        Args:
            x (torch.Tensor): The original tokens x_0. Shape: [batch_size, seq_len].
            sigma (torch.Tensor): The noise level at time t. Shape: [batch_size,].

        Returns:
            torch.Tensor: The noisy tokens x_t. Shape: [batch_size, seq_len].
        """
        bs, seq_len = x.shape
        sigma = sigma.view(bs, 1)

        # Probability of a token being absorbed (masked)
        p_absorb = sigma

        # Decide which tokens to absorb
        absorb_mask = torch.rand(bs, seq_len, device=x.device) < p_absorb

        # Apply the mask
        return torch.where(absorb_mask, torch.full_like(x, self.mask_token_id), x)

    def score_entropy(self, score: torch.Tensor, sigma: torch.Tensor, x_t: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Score-Entropy loss for the absorbing state.

        Args:
            score (torch.Tensor): The model's output logits. Shape: [batch_size, seq_len, vocab_size].
            sigma (torch.Tensor): The noise level at time t. Shape: [batch_size,].
            x_t (torch.Tensor): The noisy tokens at time t. Shape: [batch_size, seq_len].
            x_0 (torch.Tensor): The original tokens. Shape: [batch_size, seq_len].

        Returns:
            torch.Tensor: The calculated loss for each sample in the batch. Shape: [batch_size, seq_len].
        """
        loss = F.cross_entropy(score.view(-1, self.vocab_size), x_0.view(-1), reduction='none')
        return loss.view(x_0.shape)