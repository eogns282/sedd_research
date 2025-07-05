import torch
import torch.nn as nn
import math
from typing import Any

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so they can be summed.
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerModel(nn.Module):
    """
    The main Transformer model for the SEDD baseline.
    This model takes a sequence of noisy tokens and a timestep and predicts the original tokens.
    """
    def __init__(self, config: Any):
        super().__init__()
        self.d_model = config.D_MODEL
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.D_MODEL)
        self.pos_encoder = PositionalEncoding(config.D_MODEL, config.MAX_SEQ_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            config.D_MODEL,
            config.N_HEAD,
            config.DIM_FEEDFORWARD,
            config.DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.NUM_ENCODER_LAYERS)
        self.time_embedding = nn.Embedding(config.NUM_TIMESTEPS, config.D_MODEL)
        self.output_layer = nn.Linear(config.D_MODEL, config.VOCAB_SIZE)

    def forward(self, src: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): The input sequence of noisy tokens. Shape: [batch_size, seq_len].
            timesteps (torch.Tensor): The current timestep for each sample in the batch. Shape: [batch_size,].

        Returns:
            torch.Tensor: The predicted logits for the original tokens. Shape: [batch_size, seq_len, vocab_size].
        """
        # 1. Get token embeddings and scale
        # Shape: [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        
        # 2. Add positional encoding
        # Shape: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        pos_encoded = self.pos_encoder(embedded)
        
        # 3. Get timestep embeddings and add them
        # Shape: [batch_size,] -> [batch_size, d_model] -> [batch_size, 1, d_model]
        time_emb = self.time_embedding(timesteps).unsqueeze(1)
        x = pos_encoded + time_emb
        
        # 4. Pass through the transformer encoder
        # Shape: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        transformer_output = self.transformer_encoder(x)
        
        # 5. Predict the logits for the original tokens
        # Shape: [batch_size, seq_len, d_model] -> [batch_size, seq_len, vocab_size]
        logits = self.output_layer(transformer_output)
        
        return logits