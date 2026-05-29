# NeuroSync

A Multimodal Representation Learning and Brain-AI Research Platform.

NeuroSync is an advanced AI research framework that unifies **Text**, **Image**, and **Brain Signals (EEG)** into a shared representation space. This enables cross-modal reasoning, allowing the system to decode brain activity into visual and textual concepts, and vice-versa.

## Architecture Highlights
- **Vision/Language (Phase 1):** CLIP-like contrastive learning alignment using PyTorch, DistilBERT, and ResNet/ViT.
- **Generative AI (Phase 2):** Fine-tuned open-weights LLMs (Llama/Mistral) via LoRA for multi-modal context explanation.
- **Brain Decoding (Phase 3 & 4):** Deep learning architectures (CNN/Transformers) for mapping raw EEG temporal sequences to the shared embedding space.
- **MLOps & Full-Stack (Phase 5-7):** FastAPI inference backend, React dashboard with UMAP visualizations, Docker, and Weights & Biases experiment tracking.

## Repository Structure
```
NeuroSync/
├── frontend/       # React Web Application
├── backend/        # FastAPI Inference Server
├── models/         # PyTorch Architecture Definitions
├── eeg/            # EEG Preprocessing & Dataloaders
├── vision/         # Image Processing pipelines
├── language/       # Text Processing & Tokenization
├── datasets/       # Data storage (ignored in version control)
├── experiments/    # Experiment runner scripts & configurations
├── notebooks/      # Exploratory Data Analysis (Jupyter)
├── research/       # Research reports and findings
├── docker/         # Container configurations
├── tests/          # PyTest suite
└── docs/           # Technical documentation
```

## Setup (Local Development)

Install dependencies:
```bash
pip install -r requirements.txt
```
