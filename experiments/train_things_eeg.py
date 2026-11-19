"""Train an EEG encoder to align with CLIP image features on THINGS-EEG2 and report 200-way retrieval.

Example (Colab/GPU):
  python experiments/train_things_eeg.py --subject_dir data/sub-01 \
      --train_feat data/train_feat.npy --test_feat data/test_feat.npy
  add --shuffle_labels for the control run (expected: chance, 0.5% top-1 / 2.5% top-5)
  add --encoder transformer to use the original NeuroSync CNN+Transformer EEGEncoder instead.
Only the FINAL epoch is reported; the test set is never used to choose epochs or hyperparameters.
"""
import argparse, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eeg.things_dataset import ThingsEEGDataset, load_eeg_split
from eeg.baseline_encoder import ConvEEGBaseline
from eeg.model import EEGEncoder
from eeg.metrics import retrieval_accuracy
from models.loss import clip_loss


def build_encoder(name, channels, samples):
    if name == "baseline":
        return ConvEEGBaseline(channels, samples)
    return TransformerWrapper(channels)


class TransformerWrapper(torch.nn.Module):
    """Original NeuroSync EEGEncoder + L2 norm + learnable temperature, so it plugs into the same loop."""
    def __init__(self, channels):
        super().__init__()
        self.enc = EEGEncoder(num_channels=channels, projection_dim=512)
        self.logit_scale = torch.nn.Parameter(torch.tensor(2.659))
    def forward(self, x):
        return torch.nn.functional.normalize(self.enc(x), dim=-1)


@torch.no_grad()
def embed_all(model, eeg, device):
    model.eval()
    return model(eeg.to(device)).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject_dir", required=True)
    p.add_argument("--train_feat", required=True)
    p.add_argument("--test_feat", required=True)
    p.add_argument("--encoder", choices=["baseline", "transformer"], default="baseline")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle_labels", action="store_true")
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_tr, x_te = load_eeg_split(a.subject_dir, "training"), load_eeg_split(a.subject_dir, "test")
    y_tr, y_te = np.load(a.train_feat), np.load(a.test_feat)
    train = ThingsEEGDataset(x_tr, y_tr, shuffle_labels=a.shuffle_labels, seed=a.seed)
    loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, drop_last=False)
    x_te = torch.tensor(x_te); y_te = torch.tensor(y_te)

    model = build_encoder(a.encoder, x_tr.shape[1], x_tr.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    for epoch in range(1, a.epochs + 1):
        model.train()
        for batch in loader:
            z = model(batch["eeg"].to(device))
            loss = clip_loss(z, batch["image_feature"].to(device), model.logit_scale)
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 == 0:
            print(f"epoch {epoch} loss {loss.item():.3f} test {retrieval_accuracy(embed_all(model, x_te, device), y_te)}")
    final = retrieval_accuracy(embed_all(model, x_te, device), y_te)
    print(f"FINAL encoder={a.encoder} shuffle_labels={a.shuffle_labels} seed={a.seed} "
          f"top1={final[1]:.1f}% top5={final[5]:.1f}% (chance 0.5% / 2.5%)")


if __name__ == "__main__":
    main()
