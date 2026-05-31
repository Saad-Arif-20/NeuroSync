import torch
import torch.nn as nn
from transformers import AutoModel
import torchvision.models as models

class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", projection_dim=512):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        # DistilBERT outputs 768 dim embeddings
        self.projection = nn.Linear(768, projection_dim)
        
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token representation (or mean pooling)
        x = output.last_hidden_state[:, 0, :]
        x = self.projection(x)
        return x

class ImageEncoder(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        # Using a pretrained ResNet18 for lightweight feature extraction
        resnet = models.resnet18(pretrained=True)
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        # ResNet18 outputs 512 dim features
        self.projection = nn.Linear(512, projection_dim)
        
    def forward(self, images):
        x = self.feature_extractor(images)
        x = x.view(x.size(0), -1) # Flatten
        x = self.projection(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eeg.model import EEGEncoder

class SharedEmbeddingSpace(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        self.text_encoder = TextEncoder(projection_dim=projection_dim)
        self.image_encoder = ImageEncoder(projection_dim=projection_dim)
        
        # Add EEG encoder for Phase 3
        self.eeg_encoder = EEGEncoder(projection_dim=projection_dim)
        
    def forward(self, images=None, input_ids=None, attention_mask=None, eeg_signals=None):
        features = {}
        if images is not None:
            features['image_embeddings'] = self.image_encoder(images)
        if input_ids is not None and attention_mask is not None:
            features['text_embeddings'] = self.text_encoder(input_ids, attention_mask)
        if eeg_signals is not None:
            features['eeg_embeddings'] = self.eeg_encoder(eeg_signals)
            
        return features
