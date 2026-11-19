"""Loader for the THINGS-EEG2 preprocessed data (Gifford et al. 2022, OSF project 3jk45).

Each item is (EEG, image feature). EEG repetitions of the same image are averaged to
raise the signal-to-noise ratio (4 reps for training images, 80 for test images).
Image features are pre-computed CLIP embeddings stored in the same order as the EEG
(the order of image_metadata.npy), see experiments/colab_baseline_full.py cell 5.
"""
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def load_eeg_split(subject_dir, split):
    """Return EEG as float32 array (n_images, channels, samples). split: 'training' or 'test'."""
    pattern = f"{subject_dir}/**/preprocessed_eeg_{split}.npy"
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern}")
    raw = np.load(matches[0], allow_pickle=True).item()
    data = raw["preprocessed_eeg_data"]          # (images, repetitions, channels, samples)
    return data.mean(axis=1).astype(np.float32)  # average over repetitions


class ThingsEEGDataset(Dataset):
    def __init__(self, eeg, image_features, shuffle_labels=False, seed=0):
        eeg = torch.as_tensor(eeg, dtype=torch.float32)
        feats = torch.as_tensor(image_features, dtype=torch.float32)
        if len(eeg) != len(feats):
            raise ValueError(f"EEG has {len(eeg)} items but image features has {len(feats)}")
        if shuffle_labels:  # control: breaks the EEG<->image pairing, so accuracy must fall to chance
            g = torch.Generator().manual_seed(seed)
            feats = feats[torch.randperm(len(feats), generator=g)]
        self.eeg, self.feats = eeg, feats

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return {"eeg": self.eeg[idx], "image_feature": self.feats[idx]}
