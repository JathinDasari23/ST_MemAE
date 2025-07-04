# utils/datasets.py

import os
import glob
import cv2
import torch
from torch.utils.data import Dataset

class VideoFramesDataset(Dataset):
    def __init__(self, root_dir, resize=(256, 256), n_frames=4):
        self.files = sorted(
            glob.glob(os.path.join(root_dir, "*.jpg")) +
            glob.glob(os.path.join(root_dir, "*.png"))
        )
        self.n_frames = n_frames
        self.resize = resize

    def __len__(self):
        return len(self.files) - self.n_frames

    def __getitem__(self, idx):
        frames = []
        for i in range(self.n_frames):
            img = cv2.imread(self.files[idx + i])
            img = cv2.resize(img, self.resize)
            img = img / 255.0
            frames.append(img.transpose(2, 0, 1))
        target = cv2.imread(self.files[idx + self.n_frames])
        target = cv2.resize(target, self.resize)
        target = target / 255.0
        target = target.transpose(2, 0, 1)
        return torch.FloatTensor(frames), torch.FloatTensor(target)
