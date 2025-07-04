# model/decoder.py

import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super(Decoder, self).__init__()
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, features):
        e1, e2, e3, e4 = features
        x = self.up1(e4)
        x = torch.cat([x, e3], dim=1)
        x = self.dec1(x)
        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        x = self.up3(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec3(x)
        out = self.out_conv(x)
        return out
