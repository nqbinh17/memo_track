import math
import copy

import torch, torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class ContextAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.encoder.embed_dim
        self.heads = cfg.encoder.attention_heads
        self.head_dim = self.embed_dim // self.heads
        self.eps = 1e-6
        self.linears = clones(nn.Linear(self.embed_dim, self.embed_dim), 4)
        self.dropout = nn.Dropout(p=cfg.dropout)

    def forward(self, x, mask):
        batch_size = x.size(0)
        query, key, value = \
            [l(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
             .view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)