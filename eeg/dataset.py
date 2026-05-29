import torch
from torch.utils.data import Dataset
import numpy as np

class EEGDataset(Dataset):
    def __init__(self, data_paths=None, num_channels=64, seq_length=1000):
        """
        In a real scenario, data_paths would be a list of paths to .edf or .fif files.
        We'd use libraries like MNE-Python (mne.io.read_raw_edf) to load them.
        """
        self.data_paths = data_paths
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.num_samples = 200 # Mock dataset size
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Mocking raw EEG data: (num_channels, sequence_length)
        # In reality, this would be: 
        # raw_data = mne.io.read_raw_edf(self.data_paths[idx])
        # eeg_signal = raw_data.get_data()
        
        eeg_signal = np.random.randn(self.num_channels, self.seq_length).astype(np.float32)
        
        # Normalize the signal (standard practice in EEG processing)
        mean = np.mean(eeg_signal, axis=1, keepdims=True)
        std = np.std(eeg_signal, axis=1, keepdims=True)
        eeg_signal = (eeg_signal - mean) / (std + 1e-6)
        
        # We simulate the corresponding text description/label
        # (e.g., "The subject is looking at a picture of a dog")
        target_label = "subject_viewing_object"
        
        return {
            "eeg": torch.tensor(eeg_signal),
            "label": target_label
        }
