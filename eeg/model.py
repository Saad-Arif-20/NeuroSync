import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, embedding_dim)
        x = x + self.pe[:x.size(0)]
        return x

class EEGEncoder(nn.Module):
    def __init__(self, num_channels=64, projection_dim=512):
        super().__init__()
        
        # 1D CNN to extract local temporal features from raw EEG channels
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=128, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Transformer for global dependencies over time
        self.pos_encoder = PositionalEncoding(d_model=256)
        encoder_layers = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)
        
        # Final projection to match the shared embedding space (512-dim)
        self.projection = nn.Linear(256, projection_dim)
        
    def forward(self, x):
        # x shape: (batch_size, num_channels, sequence_length)
        
        # 1. Temporal feature extraction
        features = self.conv_blocks(x) 
        
        # 2. Prepare for Transformer (seq_len, batch_size, embedding_dim)
        features = features.permute(2, 0, 1) 
        features = self.pos_encoder(features)
        
        # 3. Apply Transformer
        transformer_out = self.transformer_encoder(features)
        
        # 4. Global average pooling over the sequence length
        pooled_out = transformer_out.mean(dim=0) # shape: (batch_size, 256)
        
        # 5. Project to shared embedding space
        projected = self.projection(pooled_out) # shape: (batch_size, 512)
        
        return projected
