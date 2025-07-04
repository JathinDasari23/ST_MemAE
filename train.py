# train.py

import torch
from torch.utils.data import DataLoader
from model.network import STMemAE
from utils.datasets import VideoFramesDataset
from utils.losses import prediction_loss
from config import CONFIG

dataset = VideoFramesDataset(root_dir="data/train")
loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

model = STMemAE(CONFIG).to(CONFIG["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

for epoch in range(CONFIG["num_epochs"]):
    for batch in loader:
        frames, target = batch
        frames = frames.to(CONFIG["device"])
        target = target.to(CONFIG["device"])

        inp = frames[:, -1]
        pred, _ = model(inp)

        loss = prediction_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} Loss: {loss.item():.6f}")
