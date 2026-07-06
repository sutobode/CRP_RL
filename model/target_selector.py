import torch
import torch.nn as nn


class TargetSelector(nn.Module):
    """
    Learned target selection policy for HTR.
    
    Input:
        node_embeddings: (batch, n_stacks, embed_dim) — per-stack encoder outputs
        graph_embedding: (batch, embed_dim) — global graph embedding (mean pooling)
        mask: (batch, n_stacks) — boolean, True = valid target
        
    Output:
        logits: (batch, n_stacks) — unnormalized target scores
    """
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_embeddings, graph_embedding, mask=None):
        g = graph_embedding.unsqueeze(1).expand(-1, node_embeddings.size(1), -1)
        x = torch.cat([node_embeddings, g], dim=-1)
        logits = self.net(x).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        return logits
