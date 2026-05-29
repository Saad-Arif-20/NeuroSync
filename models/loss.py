import torch
import torch.nn.functional as F

def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    # Normalize the embeddings
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    
    # Calculate cosine similarity
    # Shape: (batch_size, batch_size)
    logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    
    # The targets are simply the diagonal (i.e. index 0 to batch_size - 1)
    batch_size = image_embeddings.shape[0]
    labels = torch.arange(batch_size).to(image_embeddings.device)
    
    # Calculate Cross Entropy Loss in both directions (Image -> Text and Text -> Image)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    # Average the loss
    loss = (loss_i + loss_t) / 2
    return loss
