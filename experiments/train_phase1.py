import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torchvision import transforms
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.multimodal_encoder import SharedEmbeddingSpace
from models.loss import contrastive_loss

# Mock Dataset for Phase 1 (Replace with Flickr8k/COCO later)
class MockMultimodalDataset(Dataset):
    def __init__(self, tokenizer, num_samples=100):
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        
        # ResNet18 expects 224x224 images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a mock 3-channel image (e.g. RGB)
        img = torch.rand(3, 224, 224) 
        img = self.transform(transforms.ToPILImage()(img))
        
        # Mock text caption
        text = f"This is a mock description for image number {idx} showing a random object."
        
        encoded_text = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=32, 
            return_tensors="pt"
        )
        
        return {
            "image": img,
            "input_ids": encoded_text["input_ids"].squeeze(0),
            "attention_mask": encoded_text["attention_mask"].squeeze(0)
        }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = SharedEmbeddingSpace(projection_dim=512).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Dataloader
    dataset = MockMultimodalDataset(tokenizer, num_samples=320)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    epochs = 3
    
    print("Starting Phase 1 Training (Contrastive Learning)...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            features = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss = contrastive_loss(
                image_embeddings=features["image_embeddings"], 
                text_embeddings=features["text_embeddings"]
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ---")
        
    print("Training Complete! Shared embedding space has been learned.")
    
if __name__ == "__main__":
    train()
