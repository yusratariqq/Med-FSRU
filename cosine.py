

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class CosineRAG(nn.Module):
    """
    Cosine-similarity retrieval module.

    """

    def __init__(
        self,
        embedding_path: str,
        device: torch.device,
        d_model: int,
        top_k: int = 3,
        temperature: float = 0.5,
        texts_path: str = None,
        min_similarity: float = 0.35,
    ):
        super().__init__()
        self.device = device
        self.top_k         = top_k
        self.temperature   = temperature   
        self.min_similarity = min_similarity  
        self.entry_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.confidence_gate = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # Load KB embeddings
        if embedding_path.endswith(".npy"):
            emb = np.load(embedding_path)
            kb  = torch.tensor(emb, dtype=torch.float32, device=device)
        else:
            kb = torch.load(
                embedding_path,
                map_location=device,
                weights_only=False
            )

        if kb.shape[1] != d_model:
            raise ValueError(
                f"KB embedding dim ({kb.shape[1]}) != model d_model ({d_model}). "
                f"Rebuild the KB with medical_rag_builder.py using the current "
                f"d_model before training."
            )
        kb = kb.to(device)

        kb = F.normalize(kb, dim=-1)
        self.register_buffer("kb_embeddings", kb)
        print(f"✅ KB loaded: {kb.shape[0]} entries, dim={kb.shape[1]}")
        self.kb_texts = None
        if texts_path is not None:
            import pickle, os
            if os.path.exists(texts_path):
                with open(texts_path, "rb") as f:
                    self.kb_texts = pickle.load(f)
                print(f"✅ KB texts loaded for inspection: {len(self.kb_texts)} entries")
            else:
                print(f"⚠️ texts_path given but not found: {texts_path} — inspection disabled")

    def lookup_texts(self, topk_indices: torch.Tensor):
        """Map retrieved indices back to their KB text, for logging/debugging."""
        if self.kb_texts is None:
            return None
        idx = topk_indices.detach().cpu().tolist()
        return [[self.kb_texts[i] for i in row] for row in idx]

    def cosine_score(
        self,
        query: torch.Tensor,
        keys: torch.Tensor
    ) -> torch.Tensor:
        
        query = F.normalize(query, dim=-1)
        keys  = F.normalize(keys,  dim=-1)
        return torch.matmul(query, keys.T).clamp(-1.0, 1.0)

    def forward(self, direct_query: torch.Tensor):
        
        query = F.layer_norm(direct_query, [direct_query.size(-1)])
        scores = self.cosine_score(query, self.kb_embeddings)   
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)
        retrieved = self.kb_embeddings[topk_indices]          
        relevance_mask = (topk_scores >= self.min_similarity).float().unsqueeze(-1)
        retrieved = F.layer_norm(retrieved, [retrieved.size(-1)])

        rel_weights = F.softmax(
            topk_scores / self.temperature, dim=-1
        ).unsqueeze(-1)

        query_expanded = query.unsqueeze(1).expand(-1, retrieved.size(1), -1)  # [B, top_k, d_model]
        gate_input      = torch.cat([query_expanded, retrieved], dim=-1)       # [B, top_k, 2*d_model]
        gate            = self.entry_gate(gate_input)                          # [B, top_k, 1]

        # AFTER
        top1_score = topk_scores[:, :1]                      
        confidence = self.confidence_gate(top1_score)        
        confidence = confidence.unsqueeze(1)                  

        retrieved = retrieved * rel_weights * gate * relevance_mask * confidence
        return retrieved, topk_indices
