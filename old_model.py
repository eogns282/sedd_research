import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerDenoisingModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, num_timesteps=1000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.time_embedding = nn.Embedding(num_timesteps, d_model)
        self.output_layer = nn.Linear(d_model, vocab_size) # Predicts the logits for each token in the vocabulary

    def forward(self, noisy_tokens, timesteps):
        # noisy_tokens: (batch_size, seq_len)
        # timesteps: (batch_size,)
        
        # 1. Get token embeddings
        x = self.embedding(noisy_tokens) * math.sqrt(self.d_model)
        
        # 2. Add positional encoding
        x = self.pos_encoder(x)
        
        # 3. Get timestep embeddings and add them
        time_emb = self.time_embedding(timesteps).unsqueeze(1)
        x = x + time_emb
        
        # 4. Pass through the transformer
        transformer_output = self.transformer_encoder(x)
        
        # 5. Predict the logits for the original tokens
        predicted_logits = self.output_layer(transformer_output)
        
        return predicted_logits