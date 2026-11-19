import os, sys, tempfile
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eeg.things_dataset import ThingsEEGDataset, load_eeg_split
from eeg.baseline_encoder import ConvEEGBaseline
from eeg.model import EEGEncoder
from eeg.metrics import retrieval_accuracy
from models.loss import clip_loss

CH, T = 17, 100  # THINGS-EEG2 preprocessed: 17 occipito-parietal channels, 100 samples (-0.2..0.79 s)


def test_baseline_encoder_shape_and_norm():
    out = ConvEEGBaseline(CH, T)(torch.randn(4, CH, T))
    assert out.shape == (4, 512)
    assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-4)


def test_original_transformer_encoder_accepts_17x100():
    assert EEGEncoder(num_channels=CH)(torch.randn(2, CH, T)).shape == (2, 512)


def test_retrieval_perfect_and_chance():
    z = torch.nn.functional.normalize(torch.randn(200, 512), dim=-1)
    assert retrieval_accuracy(z, z) == {1: 100.0, 5: 100.0}
    chance = retrieval_accuracy(torch.randn(200, 512), torch.randn(200, 512))
    assert chance[1] < 5 and chance[5] < 10   # random embeddings stay near 0.5% / 2.5%


def test_clip_loss_lower_when_aligned():
    z = torch.nn.functional.normalize(torch.randn(32, 512), dim=-1)
    scale = torch.tensor(2.659)
    aligned = clip_loss(z, z, scale)
    unaligned = clip_loss(z, torch.nn.functional.normalize(torch.randn(32, 512), dim=-1), scale)
    assert aligned < unaligned


def test_load_split_averages_repetitions():
    with tempfile.TemporaryDirectory() as d:
        data = np.random.randn(6, 4, CH, T).astype(np.float32)
        np.save(os.path.join(d, "preprocessed_eeg_training.npy"),
                {"preprocessed_eeg_data": data}, allow_pickle=True)
        x = load_eeg_split(d, "training")
        assert x.shape == (6, CH, T)
        assert np.allclose(x, data.mean(1), atol=1e-6)


def test_dataset_shuffle_control_changes_pairing_and_validates_length():
    eeg, feats = torch.randn(50, CH, T), torch.randn(50, 512)
    ds = ThingsEEGDataset(eeg, feats)
    ctrl = ThingsEEGDataset(eeg, feats, shuffle_labels=True, seed=1)
    assert not torch.equal(ds.feats, ctrl.feats)
    assert torch.equal(ds.feats.sort(0).values, ctrl.feats.sort(0).values)  # same features, different order
    try:
        ThingsEEGDataset(eeg, feats[:10]); assert False
    except ValueError:
        pass
