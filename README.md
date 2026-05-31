# NeuroSync

A multimodal AI research platform for aligning vision, language, and EEG brain signals into a shared representation space.

The idea started as a way to explore CLIP-style contrastive learning, and grew into a 3-phase pipeline: vision-language alignment, LLM-based generation, and a brainwave decoder. It runs locally for dev and can be proxied to a Colab T4 GPU for inference since Phi-2 is too slow on CPU.

## What's implemented

**Phase 1 — Vision-Language Alignment**
Contrastive learning between image (ResNet18) and text (DistilBERT) embeddings. Trained on Google Conceptual Captions via HuggingFace streaming. Both encoders are projected to a shared 512-dim space using InfoNCE loss.

**Phase 2 — Multimodal LLM**
Phi-2 fine-tuned with LoRA (via PEFT) to take a projected visual or EEG embedding and generate a text description. The projection layer bridges the 512-dim shared space to Phi-2's 2560-dim hidden size.

**Phase 3 — EEG Brainwave Encoder**
1D-CNN for local temporal feature extraction followed by a Transformer encoder for global context. Raw EEG input: `(batch, 64 channels, 1000 timesteps)` → 512-dim embedding in the shared space.

**Inference Stack**
- FastAPI backend (`/api/embed/image`, `/api/embed/eeg`, `/api/generate`)
- React + Vite frontend with 3D UMAP embedding visualization (Plotly)
- ngrok tunnel for Colab GPU-accelerated inference

## Stack

- PyTorch, HuggingFace Transformers, PEFT (LoRA)
- FastAPI, Uvicorn
- React, Vite, Plotly.js
- Docker, GitHub Actions, WandB

## Project Structure

```
NeuroSync/
├── backend/        # FastAPI inference server
├── models/         # PyTorch model definitions (SharedEmbeddingSpace, LLMAssistant)
├── eeg/            # EEG encoder architecture and dataloader
├── experiments/    # Training scripts for Phase 1, 2, and WandB setup
├── frontend/       # React dashboard with UMAP visualizations
└── .github/        # CI/CD pipeline
```

## Running Locally

```bash
pip install -r requirements.txt

# Terminal 1 — Backend
python backend/main.py

# Terminal 2 — Frontend
cd frontend && npm install && npm run dev
```

## Cloud Inference (Colab)

Load the backend on a free T4 GPU using the ngrok tunnel setup in the Colab notebook. The React frontend proxies all `/api/*` requests through Vite to the ngrok URL.

## Notes

Trained weights (`*.pth`) are excluded from version control due to file size. Training scripts are in `experiments/`.
