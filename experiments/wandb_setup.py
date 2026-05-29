import wandb
import os

def initialize_experiment_tracking(project_name="NeuroSync", experiment_name="Phase3-CNN-Transformer"):
    """
    Initializes Weights & Biases (WandB) for tracking PyTorch training metrics.
    Recruiters love seeing this because it proves you know how to run 
    industrial-scale machine learning experiments.
    """
    
    # In a real environment, you'd run `wandb login` in your terminal first
    # or set the WANDB_API_KEY environment variable.
    
    wandb.init(
        project=project_name,
        name=experiment_name,
        config={
            "learning_rate": 1e-4,
            "architecture": "CNN + Transformer",
            "dataset": "Mock-EEG-1000seq",
            "epochs": 10,
            "batch_size": 32,
            "embedding_dim": 512
        }
    )
    
    return wandb

# Example usage inside your training loop:
# wandb.log({"loss": loss.item(), "accuracy": accuracy})
