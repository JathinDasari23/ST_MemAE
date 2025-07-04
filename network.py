# model/network.py

import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .memory_module import MemoryModule

class STMemAE(nn.Module):
    def __init__(self, config):
        super(STMemAE, self).__init__()
        self.encoder = Encoder(in_channels=config["channels"])
        self.memory = MemoryModule(
            mem_dim=config["num_memory_items"],
            fea_dim=512,
            num_heads=config["num_heads"]
        )
        self.decoder = Decoder(out_channels=config["channels"])

    def forward(self, x):
        features = self.encoder(x)
        memory_out, attn = self.memory(features[-1])
        new_features = features[:-1] + [memory_out]
        out = self.decoder(new_features)
        return out, attn
