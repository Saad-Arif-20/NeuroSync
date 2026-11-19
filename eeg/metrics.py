import torch


@torch.no_grad()
def retrieval_accuracy(eeg_emb, image_emb, ks=(1, 5)):
    """N-way retrieval: for each EEG embedding i, rank all N image embeddings; the correct one is image i.
    Returns {k: percentage}. Chance level for top-k is 100*k/N (0.5% / 2.5% for N=200)."""
    eeg_emb = torch.nn.functional.normalize(eeg_emb.float(), dim=-1)
    image_emb = torch.nn.functional.normalize(image_emb.float(), dim=-1)
    ranking = (eeg_emb @ image_emb.T).argsort(dim=1, descending=True)
    truth = torch.arange(len(eeg_emb), device=ranking.device).unsqueeze(1)
    return {k: (ranking[:, :k] == truth).any(1).float().mean().item() * 100 for k in ks}
