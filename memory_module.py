# model/memory_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryModule(nn.Module):
    def __init__(self, mem_dim=512, fea_dim=512, num_heads=4):
        super(MemoryModule, self).__init__()
        self.num_heads = num_heads
        self.memory = nn.Parameter(torch.randn(mem_dim, fea_dim))
        self.q_proj = nn.Linear(fea_dim, fea_dim)
        self.k_proj = nn.Linear(fea_dim, fea_dim)
        self.v_proj = nn.Linear(fea_dim, fea_dim)

    def forward(self, x):
        b, c, h, w = x.size()
        x_flat = x.permute(0, 2, 3, 1).reshape(b, -1, c)
        
        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (c ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        dist = torch.cdist(out, self.memory.unsqueeze(0), p=2)
        min_idx = dist.argmin(dim=-1)
        mem_out = self.memory[min_idx]

        enhanced = out + mem_out
        enhanced = enhanced.reshape(b, h, w, c).permute(0, 3, 1, 2)

        return enhanced, attn_weights
