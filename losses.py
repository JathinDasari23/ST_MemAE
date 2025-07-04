# utils/losses.py

import torch
import torch.nn.functional as F

def prediction_loss(pred, target):
    return F.mse_loss(pred, target)

def temporal_feature_shrinkage_loss(mem_out, query):
    return torch.norm(mem_out - query, p=2)

def temporal_feature_separation_loss(mem_out, mem_sec):
    return torch.norm(mem_out - mem_sec, p=2)

def bml_loss(L1, L2, sigma1, sigma2):
    return (1 / (2 * sigma1**2)) * L1 + (1 / (2 * sigma2**2)) * L2 + torch.log(sigma1 * sigma2)
