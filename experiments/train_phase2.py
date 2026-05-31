import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llm_assistant import MultimodalLLMAssistant

# Mock Dataset: Simulating output from Phase 1
class MockEmbeddingToTextDataset(Dataset):
    def __init__(self, tokenizer, num_samples=100):
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Mock a 512-dim embedding (Simulating what our Image/EEG encoder outputs)
        mock_embedding = torch.randn(512) * 0.1
        
        # 2. Mock a text prompt and expected output
        prompt = "Describe this signal:"
        target_response = f"This signal indicates a high probability of recognizing object number {idx}."
        
        # Combine prompt and target for training
        full_text = prompt + " " + target_response
        
        encoded_text = self.tokenizer(
            full_text,
            padding='max_length',
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )
        
        # For language modeling, labels are usually the same as input_ids
        labels = encoded_text["input_ids"].clone()
        # Ignore pad tokens in loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "embedding": mock_embedding,
            "input_ids": encoded_text["input_ids"].squeeze(0),
            "attention_mask": encoded_text["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading LLM (This might take a moment to download weights)...")
    # Using a small LLM for demonstration
    model = MultimodalLLMAssistant(llm_model_name="microsoft/phi-2", embedding_dim=512)
    model.to(device)
    
    # We only optimize the LoRA weights and our projection layer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    
    dataset = MockEmbeddingToTextDataset(model.tokenizer, num_samples=160)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    epochs = 2
    
    print("Starting Phase 2 Training (LoRA Fine-Tuning)...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            embeddings = batch["embedding"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_embeddings=embeddings, 
                input_ids=input_ids, 
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent NaN in fp16
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | LLM Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ---")
        
    print("Training Complete! Testing Generation...")
    
    # Test generation
    model.eval()
    with torch.no_grad():
        test_embedding = torch.randn(1, 512).to(device)
        response = model.generate(test_embedding, prompt_text="Describe this signal: ", max_new_tokens=20)
        print(f"\nGenerative Test Output:\n{response}")
        
    # SAVE THE MODEL WEIGHTS
    save_path = "neurosync_phase2_lora.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Phase 2 Model successfully saved to {save_path}")

if __name__ == "__main__":
    train()
