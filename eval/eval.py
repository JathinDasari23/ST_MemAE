# eval.py

import torch
from torch.utils.data import DataLoader
from model.network import STMemAE
from utils.datasets import VideoFramesDataset
from utils.metrics import psnr
from config import CONFIG

dataset = VideoFramesDataset(root_dir="data/test")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = STMemAE(CONFIG).to(CONFIG["device"])
model.eval()

psnr_scores = []

with torch.no_grad():
    for batch in loader:
        frames, target = batch
        frames = frames.to(CONFIG["device"])
        target = target.to(CONFIG["device"])

        inp = frames[:, -1]
        pred, _ = model(inp)

        mse = torch.mean((pred - target)**2)
        psnr_val = psnr(mse)
        psnr_scores.append(psnr_val.item())

print("Average PSNR:", sum(psnr_scores)/len(psnr_scores))
