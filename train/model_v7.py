"""
Stage 2 — Attention-augmented proposer (model_v7)

Architecture:
    Input projection: Linear(25 -> 128)
    Transformer block: MultiHeadAttention(128, heads=4) + FFN(128 -> 512 -> 128)
    Dual heads: Linear(128 -> 8) for op, Linear(128 -> 8) for tgt

Key change from v6:
    v6: each step predicted independently from single feature vector
    v7: each step attends to ALL previous steps in the 12-block sequence
        — the model sees the full execution context before predicting

This gives the model:
    - Awareness of where it is in the canonical sequence
    - Awareness of what ops were chosen in previous blocks
    - Causal masking — block N can only see blocks 0..N-1

Input:  [B, 12, 25]  — batch of 12-block sequences
Output: op_logits  [B, 12, 8]
        tgt_logits [B, 12, 8]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

FEATURE_DIM   = 25
SEQ_LEN       = 12
D_MODEL       = 128
N_HEADS       = 4
D_FF          = 512
N_OP_CLASSES  = 8
N_TGT_CLASSES = 8
DROPOUT       = 0.1

OP_NAMES  = ["SELECT_UNIVERSE","WITNESS_NEAREST","ATTEND","FFN_STEP",
             "PROJECT_LAYER","RETURN_SET","ACCEPT","REJECT"]
TGT_NAMES = ["PHONEME","SYLLABLE","MORPHEME","WORD",
             "PHRASE","SEMANTIC","DISCOURSE","none"]


class TransformerBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with causal mask
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, is_causal=mask is not None)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class ProposerV7(nn.Module):
    """
    Attention-augmented tower proposer.
    Processes full 12-block sequence with causal self-attention.
    """
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(FEATURE_DIM, D_MODEL)
        self.pos_emb    = nn.Embedding(SEQ_LEN, D_MODEL)
        self.transformer = nn.ModuleList([
            TransformerBlock() for _ in range(2)
        ])
        self.op_head    = nn.Linear(D_MODEL, N_OP_CLASSES)
        self.tgt_head   = nn.Linear(D_MODEL, N_TGT_CLASSES)
        self.dropout    = nn.Dropout(DROPOUT)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 12, 25]
        B, T, _ = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        h    = self.dropout(self.input_proj(x) + self.pos_emb(pos))

        # Causal mask: block i cannot attend to block j > i
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        for block in self.transformer:
            h = block(h, mask)

        return self.op_head(h), self.tgt_head(h)   # [B, 12, 8], [B, 12, 8]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = ProposerV7()
    print(f"ProposerV7 parameters: {model.n_params():,}")
    x = torch.randn(4, 12, 25)
    op, tgt = model(x)
    print(f"op shape:  {tuple(op.shape)}")
    print(f"tgt shape: {tuple(tgt.shape)}")
    print("OK")
