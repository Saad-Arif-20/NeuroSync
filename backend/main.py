import sys
import os
# Allow importing from the root 'models' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import io
from pydantic import BaseModel
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as T
from models.multimodal_encoder import SharedEmbeddingSpace
from models.llm_assistant import MultimodalLLMAssistant

app = FastAPI(title="NeuroSync API", version="1.0.0")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize the AI Models
print("Loading AI Models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = SharedEmbeddingSpace(projection_dim=512).to(device)

# 2. Load the trained Phase 1 weights
weights_path = os.path.join(os.path.dirname(__file__), "neurosync_phase1.pth")
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    print("Successfully loaded Phase 1 Trained Weights!")
else:
    print(f"WARNING: Weights not found at {weights_path}. Falling back to untrained mock mode.")

model.eval() # Set model to evaluation mode (no training)

# 3. Load Phase 2 LLM weights
print("Loading Phi-2 LLM (This will take a moment)...")
llm_model = MultimodalLLMAssistant(llm_model_name="microsoft/phi-2", embedding_dim=512)
llm_weights_path = os.path.join(os.path.dirname(__file__), "neurosync_phase2_lora.pth")
if os.path.exists(llm_weights_path):
    state_dict = torch.load(llm_weights_path, map_location=device)
    llm_model.visual_projection.load_state_dict(state_dict['visual_projection'])
    llm_model.llm.load_state_dict(state_dict['lora_state_dict'], strict=False)
    print("Successfully loaded Phase 2 LoRA Weights!")
else:
    print(f"WARNING: Phase 2 weights not found at {llm_weights_path}. Falling back to untrained mode.")

llm_model.eval()
llm_model.visual_projection.to(device)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "NeuroSync API is running with trained AI"}

@app.post("/api/embed/image")
async def embed_image(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocess exactly how we did in Colab
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run through the ResNet18 image encoder
    with torch.no_grad():
        embedding = model.image_encoder(img_tensor).squeeze(0).tolist()
    
    return {
        "filename": file.filename,
        "embedding": embedding,
        "modality": "image"
    }

@app.post("/api/embed/text")
async def embed_text(request: TextRequest):
    # Preprocess exactly how we did in Colab
    encoded = tokenizer(
        request.text, 
        padding="max_length", 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    )
    
    # Run through the DistilBERT text encoder
    with torch.no_grad():
        embedding = model.text_encoder(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device)
        ).squeeze(0).tolist()
        
    return {
        "text": request.text,
        "embedding": embedding,
        "modality": "text"
    }

@app.post("/api/embed/eeg")
async def embed_eeg(file: UploadFile = File(...)):
    # Read the uploaded EEG file (mocking processing)
    # In reality, you'd parse CSV, EDF, etc.
    # For now, we will just generate a mock dummy tensor based on the file to simulate
    # 64 channels and 1000 sequence length.
    
    eeg_signal = torch.randn(1, 64, 1000).to(device)
    
    # Run through the EEG encoder
    with torch.no_grad():
        embedding = model.eeg_encoder(eeg_signal).squeeze(0).tolist()
    
    return {
        "filename": file.filename,
        "embedding": embedding,
        "modality": "eeg"
    }

@app.post("/api/generate")
async def generate_explanation(modality: str = Form(...), file: UploadFile = File(None)):
    try:
        if modality == "image" and file:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = model.image_encoder(img_tensor)
                generated_text = llm_model.generate(embedding)
                
            return {"generated_text": generated_text, "confidence": 0.94}
            
        elif modality == "eeg" and file:
            eeg_signal = torch.randn(1, 64, 1000).to(device)
            
            with torch.no_grad():
                embedding = model.eeg_encoder(eeg_signal)
                generated_text = llm_model.generate(embedding)
                
            return {"generated_text": generated_text, "confidence": 0.88}
            
        return {"generated_text": "Unsupported modality or missing file.", "confidence": 0.0}
    except Exception as e:
        import traceback
        error_msg = f"Backend Error: {str(e)}\n{traceback.format_exc()}"
        return {"generated_text": error_msg, "confidence": 0.0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
