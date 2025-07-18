"""
This module defines the neural network architecture for the diffusion model.

The core component is the TransformerModel, which is a standard Transformer
encoder architecture. A key feature is the addition of time embeddings, which
allows the model to condition its predictions on the noise level of the input.
"""

import torch
import torch.nn as nn
import math
from typing import Any, Optional

class SinusoidalTimeEmbedding(nn.Module):
    """
    A module for creating sinusoidal time embeddings.

    This embedding allows the model to be conditioned on the continuous time `t`,
    which represents the noise level. It transforms a scalar time value into a
    high-dimensional vector that can be added to the token embeddings. This is
    a standard technique in diffusion models (e.g., DDPM, Score-SDE).
    """
    def __init__(self, d_model: int):
        """
        Initializes the SinusoidalTimeEmbedding layer.

        Args:
            d_model (int): The dimensionality of the embedding vector. This should
                           match the model's main dimensionality (D_MODEL).
        """
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Creates the time embedding for a batch of time values.

        Args:
            t (torch.Tensor): A tensor of continuous time values.
                              Shape: [batch_size,].

        Returns:
            torch.Tensor: The calculated time embedding vectors.
                          Shape: [batch_size, d_model].
        """
        device = t.device
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer models.

    This injects information about the relative or absolute position of tokens
    in the sequence, which is necessary because the self-attention mechanism
    is otherwise permutation-invariant.
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input token embeddings.
                              Shape: [batch_size, seq_len, d_model].
        Returns:
            torch.Tensor: The embeddings with positional information added.
                          Shape: [batch_size, seq_len, d_model].
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerModel(nn.Module):
    """
    The core Transformer-based model for the diffusion process.

    This model is designed to predict the original clean tokens (x_0) given a
    noisy input (x_t) and the corresponding timestep (t).
    """
    def __init__(self, config: Any):
        """
        Initializes the TransformerModel.

        Args:
            config (Any): The global configuration object.
        """
        super().__init__()
        self.config = config
        self.d_model = config.model.d_model

        # --- Layers ---
        self.token_embedding = nn.Embedding(config.vocab.size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, config.dataset.max_seq_len)
        self.time_embedding = SinusoidalTimeEmbedding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.model.n_head,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.model.num_encoder_layers
        )
        self.output_layer = nn.Linear(self.d_model, config.vocab.size)

        # --- Gated Attention Mechanism (Idea 2) ---
        self.use_gated_attention = getattr(config.model, 'use_gated_attention', False)
        if self.use_gated_attention:
            self.gate_predictor = nn.Linear(self.d_model, 1)

        # --- Self-Conditioned Refinement Mechanism ---
        self.use_self_conditioning = getattr(config.model, 'use_self_conditioning', False)
        if self.use_self_conditioning:
            # This layer will project the concatenated input back to d_model
            self.self_cond_projection = nn.Linear(self.d_model * 2, self.d_model)


    def forward(self, 
                src: torch.Tensor, 
                t: torch.Tensor, 
                context_mask: torch.Tensor = None, 
                corruption_mask: torch.Tensor = None,
                draft_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            src (torch.Tensor): The input sequence of noisy token IDs.
            t (torch.Tensor): The continuous time values for each sequence.
            context_mask (torch.Tensor, optional): A boolean mask for CFG.
            corruption_mask (torch.Tensor, optional): A boolean mask indicating
                                                      which tokens are noise.
            draft_tokens (torch.Tensor, optional): The draft sequence from the first
                                                   pass of generation for self-conditioning.

        Returns:
            torch.Tensor: The predicted logits over the vocabulary for each token.
                          If using gated attention, also returns the gate scores.
        """
        if context_mask is not None:
            # Create a mask of shape [batch_size, 1] for broadcasting
            cfg_mask = context_mask.unsqueeze(1).to(src.device)
            # Create unconditional tokens (all MASK)
            unconditional_tokens = torch.full_like(src, self.config.vocab.mask_token_id)
            # Where the mask is True, use the unconditional tokens
            src = torch.where(cfg_mask, unconditional_tokens, src)

        token_emb = self.token_embedding(src) * math.sqrt(self.d_model)
        pos_encoded_emb = self.pos_encoder(token_emb)
        time_emb = self.time_embedding(t).unsqueeze(1)
        
        x = pos_encoded_emb + time_emb

        # --- Self-Conditioning Logic ---
        if self.use_self_conditioning and draft_tokens is not None:
            with torch.no_grad(): # Don't backprop through the draft
                draft_emb = self.token_embedding(draft_tokens) * math.sqrt(self.d_model)
            
            # Concatenate and project
            x = torch.cat([x, draft_emb], dim=-1)
            x = self.self_cond_projection(x)

        transformer_output = self.transformer_encoder(x)

        # --- Gated Attention Logic ---
        if self.use_gated_attention:
            gate_logits = self.gate_predictor(transformer_output)
            gated_output = transformer_output * torch.sigmoid(gate_logits)
            logits = self.output_layer(gated_output)
            return logits, gate_logits.squeeze(-1)
        else:
            logits = self.output_layer(transformer_output)
            return logits
