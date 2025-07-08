"""
This module defines the state transition graphs used in the diffusion process.

A "graph" in this context represents the set of all possible states (the vocabulary)
and the rules for transitioning between them during the forward (noising) process.
Each graph type corresponds to a different noising strategy.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

# --- Base Class for Graphs (for potential future extension) ---
# While not strictly necessary for two graphs, a base class would be good practice
# if more graph types were to be added. For now, we'll keep them separate
# but with identical method signatures for polymorphic use in DiffusionProcess.

class UniformGraph:
    """
    Implements a uniform discrete state graph for the diffusion process.

    In this noising scheme, a token at time t can transition to any other token
    in the vocabulary with equal probability. This is conceptually similar to
    adding uniform noise to the one-hot representation of the tokens.
    """
    def __init__(self, vocab_size: int):
        """
        Initializes the UniformGraph.

        Args:
            vocab_size (int): The total number of tokens in the vocabulary.
        """
        self.vocab_size = vocab_size

    def sample_transition(self, x_0: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a noisy state x_t given an original state x_0 and a corruption probability p.

        This method efficiently samples x_t without building the full, memory-intensive
        transition matrix. It works by deciding which tokens to corrupt based on p,
        and then replacing those tokens with new ones drawn from a uniform distribution
        over the entire vocabulary.

        Args:
            x_0 (torch.Tensor): The original, clean tokens.
                                Shape: [batch_size, seq_len].
            p (torch.Tensor): The probability of corruption for each item in the batch.
                              This value is derived from the noise schedule's total noise G(t).
                              Shape: [batch_size,].

        Returns:
            A tuple containing:
            - torch.Tensor: The noisy tokens x_t at the corresponding timestep.
                            Shape: [batch_size, seq_len].
            - torch.Tensor: The boolean mask indicating which tokens were corrupted.
                            Shape: [batch_size, seq_len].
        """
        # Ensure p can be broadcast correctly to the shape of x_0
        batch_size, seq_len = x_0.shape
        p = p.view(batch_size, 1) # Shape: [batch_size, 1]

        # Create a random mask to decide which tokens to corrupt.
        # A token is corrupted if a random number is less than its corruption probability.
        # Shape: [batch_size, seq_len]
        corrupt_mask = torch.rand(batch_size, seq_len, device=x_0.device) < p

        # Generate the replacement tokens by sampling from a uniform distribution.
        # This creates a completely random new sentence.
        # Shape: [batch_size, seq_len]
        corrupted_tokens = torch.randint(
            0, self.vocab_size, (batch_size, seq_len), device=x_0.device
        )

        # Use the mask to combine the original and corrupted tokens.
        # If corrupt_mask is True, use the new corrupted_token; otherwise, keep the original x_0.
        noisy_tokens = torch.where(corrupt_mask, corrupted_tokens, x_0)
        return noisy_tokens, corrupt_mask

    def score_entropy(self, score: torch.Tensor, sigma: torch.Tensor, x_t: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the model's prediction.

        In this simplified baseline, this is equivalent to the standard cross-entropy loss
        between the model's predicted scores (logits) and the original, clean tokens (x_0).
        The goal is to maximize the log-likelihood of the true data given the noisy version,
        which is what cross-entropy achieves.

        Args:
            score (torch.Tensor): The model's output logits.
                                  Shape: [batch_size, seq_len, vocab_size].
            sigma (torch.Tensor): The noise level at time t. Unused in this simplified loss
                                  but kept for a consistent interface. Shape: [batch_size,].
            x_t (torch.Tensor): The noisy tokens at time t. Unused in this simplified loss.
                                Shape: [batch_size, seq_len].
            x_0 (torch.Tensor): The original, clean tokens that serve as the target.
                                Shape: [batch_size, seq_len].

        Returns:
            torch.Tensor: The calculated loss for each sample in the batch.
                          Shape: [batch_size, seq_len].
        """
        # Reshape the score and target tensors for cross_entropy
        # score: [batch_size * seq_len, vocab_size]
        # x_0:   [batch_size * seq_len]
        loss = F.cross_entropy(
            score.view(-1, self.vocab_size),
            x_0.view(-1),
            reduction='none' # Return loss per element
        )
        # Reshape the loss back to the original sequence shape
        return loss.view(x_0.shape)


class AbsorbingGraph:
    """
    Implements an absorbing discrete state graph for the diffusion process.

    In this noising scheme, a token at time t can either stay the same or
    transition to a single, special "absorbing" state (the [MASK] token).
    Once a token is absorbed, it cannot transition to any other state.
    """
    def __init__(self, vocab_size: int, mask_token_id: int):
        """
        Initializes the AbsorbingGraph.

        Args:
            vocab_size (int): The total number of tokens in the vocabulary.
            mask_token_id (int): The integer ID for the [MASK] token, which serves
                                 as the absorbing state.
        """
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def sample_transition(self, x_0: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Samples a noisy state x_t given an original state x_0 and an absorption probability p.

        This method efficiently samples x_t by deciding which tokens to absorb (mask)
        based on p, and then replacing those tokens with the `mask_token_id`.

        Args:
            x_0 (torch.Tensor): The original, clean tokens.
                                Shape: [batch_size, seq_len].
            p (torch.Tensor): The probability of absorption for each item in the batch.
                              This value is derived from the noise schedule's total noise G(t).
                              Shape: [batch_size,].

        Returns:
            A tuple containing:
            - torch.Tensor: The noisy tokens x_t at the corresponding timestep.
                            Shape: [batch_size, seq_len].
            - None: Returns None for the mask to maintain a consistent API with
                    other graph types.
        """
        # Ensure p can be broadcast correctly to the shape of x_0
        batch_size, seq_len = x_0.shape
        p = p.view(batch_size, 1) # Shape: [batch_size, 1]

        # Create a random mask to decide which tokens to absorb.
        # A token is absorbed if a random number is less than its absorption probability.
        # Shape: [batch_size, seq_len]
        absorb_mask = torch.rand(batch_size, seq_len, device=x_0.device) < p

        # Create a tensor full of the MASK token ID
        mask_tokens = torch.full_like(x_0, self.mask_token_id)

        # Use the mask to combine the original and masked tokens.
        # If absorb_mask is True, use the MASK token; otherwise, keep the original x_0.
        noisy_tokens = torch.where(absorb_mask, mask_tokens, x_0)
        return noisy_tokens, absorb_mask

    def score_entropy(self, score: torch.Tensor, sigma: torch.Tensor, x_t: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the model's prediction.

        This is identical to the UniformGraph's loss function in this baseline implementation.
        It computes the cross-entropy between the model's predictions and the original tokens.

        Args:
            score (torch.Tensor): The model's output logits.
                                  Shape: [batch_size, seq_len, vocab_size].
            sigma (torch.Tensor): The noise level at time t. Unused.
                                  Shape: [batch_size,].
            x_t (torch.Tensor): The noisy tokens at time t. Unused.
                                Shape: [batch_size, seq_len].
            x_0 (torch.Tensor): The original, clean tokens (target).
                                Shape: [batch_size, seq_len].

        Returns:
            torch.Tensor: The calculated loss for each sample in the batch.
                          Shape: [batch_size, seq_len].
        """
        loss = F.cross_entropy(
            score.view(-1, self.vocab_size),
            x_0.view(-1),
            reduction='none'
        )
        return loss.view(x_0.shape)


class HybridGraph:
    """
    Implements a hybrid discrete state graph for the diffusion process.

    This graph combines the Uniform and Absorbing noising schemes. A portion of
    the corrupted tokens are replaced by the [MASK] token, while the rest are
    replaced by random tokens from the vocabulary.
    """
    def __init__(self, vocab_size: int, mask_token_id: int, mask_ratio: float):
        """
        Initializes the HybridGraph.

        Args:
            vocab_size (int): The total number of tokens in the vocabulary.
            mask_token_id (int): The integer ID for the [MASK] token.
            mask_ratio (float): The proportion of corrupted tokens to be masked.
        """
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.mask_ratio = mask_ratio

    def sample_transition(self, x_0: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Samples a noisy state x_t using a hybrid of absorbing and uniform noise.

        Args:
            x_0 (torch.Tensor): The original, clean tokens.
            p (torch.Tensor): The probability of corruption for each item in the batch.

        Returns:
            A tuple containing:
            - torch.Tensor: The noisy tokens x_t.
            - None: To maintain a consistent API.
        """
        batch_size, seq_len = x_0.shape
        p = p.view(batch_size, 1)

        # 1. Create the primary corruption mask
        corrupt_mask = torch.rand(batch_size, seq_len, device=x_0.device) < p

        # 2. Decide which of the corrupted tokens will be masked vs. uniform
        mask_decision = torch.rand(batch_size, seq_len, device=x_0.device) < self.mask_ratio
        
        # Create masks for each type of noise
        absorb_mask = corrupt_mask & mask_decision
        uniform_mask = corrupt_mask & ~mask_decision

        # 3. Generate the noise
        mask_tokens = torch.full_like(x_0, self.mask_token_id)
        uniform_tokens = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=x_0.device)

        # 4. Apply the noise
        noisy_tokens = torch.where(absorb_mask, mask_tokens, x_0)
        noisy_tokens = torch.where(uniform_mask, uniform_tokens, noisy_tokens)
        
        return noisy_tokens, corrupt_mask

    def score_entropy(self, score: torch.Tensor, sigma: torch.Tensor, x_t: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the model's prediction.
        """
        loss = F.cross_entropy(
            score.view(-1, self.vocab_size),
            x_0.view(-1),
            reduction='none'
        )
        return loss.view(x_0.shape)