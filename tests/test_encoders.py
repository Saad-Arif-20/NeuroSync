import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multimodal_encoder import SharedEmbeddingSpace
from eeg.model import EEGEncoder


def test_image_encoder_output_shape():
    model = SharedEmbeddingSpace(projection_dim=512)
    dummy_image = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model.image_encoder(dummy_image)
    assert out.shape == (2, 512), f"Expected (2, 512), got {out.shape}"


def test_text_encoder_output_shape():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = SharedEmbeddingSpace(projection_dim=512)
    tokens = tokenizer("a photo of a dog", return_tensors="pt",
                       padding="max_length", truncation=True, max_length=64)
    with torch.no_grad():
        out = model.text_encoder(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"]
        )
    assert out.shape == (1, 512), f"Expected (1, 512), got {out.shape}"


def test_eeg_encoder_output_shape():
    model = EEGEncoder(num_channels=64, projection_dim=512)
    dummy_eeg = torch.randn(2, 64, 1000)
    with torch.no_grad():
        out = model(dummy_eeg)
    assert out.shape == (2, 512), f"Expected (2, 512), got {out.shape}"
