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

from datasets import load_dataset
from PIL import Image

class Flickr8kDataset(Dataset):
    def __init__(self, tokenizer, max_samples=None):
        self.tokenizer = tokenizer
        print("Downloading Flickr8k dataset from HuggingFace (this might take a moment)...")
        # 'nlphuji/flickr8k' contains images and captions
        self.hf_dataset = load_dataset("nlphuji/flickr8k", split="train")
        
        if max_samples:
            self.hf_dataset = self.hf_dataset.select(range(max_samples))
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # HuggingFace datasets automatically load images as PIL Images
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        img_tensor = self.transform(image)
        
        # Flickr8k has multiple captions, we'll pick the first one for simplicity
        caption = item["caption"][0] if isinstance(item["caption"], list) else item["caption"]
        
        encoded_text = self.tokenizer(
            caption, 
            padding='max_length', 
            truncation=True, 
            max_length=32, 
            return_tensors="pt"
        )
        
        return {
            "image": img_tensor,
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
    
    # Dataloader (Limiting to 1000 samples for the first test run)
    dataset = Flickr8kDataset(tokenizer, max_samples=1000)
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
