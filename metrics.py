# utils/metrics.py

import torch

def psnr(mse):
    return 10 * torch.log10(1 / (mse + 1e-8))

def compute_auc(y_true, y_scores):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_scores)
