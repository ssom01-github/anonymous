import torch
import torch.nn.functional as F
import os

# Paths
embedding_path = './embeddings/train_embeddings.pt'
save_path = './kl_divergence/'
os.makedirs(save_path, exist_ok=True)

# Load embeddings
embeddings = torch.load(embedding_path)

# KL Divergence Calculation
def calculate_kl_divergence(p, q):
    p = F.log_softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    return F.kl_div(p, q, reduction='batchmean')

kl_scores = []

for (embedding, label) in embeddings:
    first_layer = embedding[0]
    last_layer = embedding[-1]

    kl_per_layer = []

    for i in range(embedding.size(0)):
        current_layer = embedding[i]
        kl_first = calculate_kl_divergence(current_layer, first_layer)
        kl_last = calculate_kl_divergence(current_layer, last_layer)

        kl_per_layer.append((kl_first.item() + kl_last.item()) / 2)  # Average KL

    kl_scores.append(kl_per_layer)

kl_scores = torch.tensor(kl_scores)

# Save KL scores
torch.save(kl_scores, os.path.join(save_path, 'train_kl_scores.pt'))

print("KL divergence saved successfully!")
