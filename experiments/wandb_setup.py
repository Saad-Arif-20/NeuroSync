import wandb
import os

def initialize_experiment_tracking(project_name="NeuroSync", experiment_name="Phase3-CNN-Transformer"):
    """
    Sets up WandB experiment tracking for a training run.
    Make sure to run `wandb login` or set WANDB_API_KEY before calling this.
    """
    run = wandb.init(
        project=project_name,
        name=experiment_name,
        config={
            "learning_rate": 1e-4,
            "architecture": "CNN + Transformer",
            "dataset": "EEG-Phase3",
            "epochs": 10,
            "batch_size": 32,
            "embedding_dim": 512
        }
    )
    
    return run

# Usage inside a training loop:
# wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]})
