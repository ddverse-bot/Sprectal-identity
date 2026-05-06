import torch.nn as nn
import torch.nn.functional as F
from .spectral_layer import SpectralTPSLayer

class SpectralUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, size=512):
        super().__init__()
        self.enc1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # The Spectral bottleneck (restores high-freq energy)
        self.spectral_fix = SpectralTPSLayer(size // 4) 
        
        self.dec1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.ConvTranspose2d(32, out_ch, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = self.pool(F.relu(self.enc2(self.pool(x))))
        x = self.spectral_fix(x)
        x = F.relu(self.dec1(x))
        return torch.sigmoid(self.dec2(x))
