import torch
import torch.nn as nn
import torch.fft

class SpectralTPSLayer(nn.Module):
    def __init__(self, size=512, initial_beta=4.0):
        super().__init__()
        # Learned Identity Parameters
        self.beta = nn.Parameter(torch.tensor([initial_beta]))
        self.alpha = nn.Parameter(torch.tensor([-2.0])) 
        
        # Precompute the harmonic grid
        coords = torch.fft.fftfreq(size)
        u, v = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('omega', torch.sqrt(u**2 + v**2 + 1e-8))

    def forward(self, x):
        x_fft = torch.fft.fftn(x, dim=(-2, -1))
        # λ(ω) = exp(α) * ω^(4 - β) 
        # When β=4, this becomes a scale-invariant constant
        reg_filter = torch.exp(self.alpha) * torch.pow(self.omega, 4.0 - self.beta)
        h_w = 1.0 / (1.0 + reg_filter)
        return torch.fft.ifftn(x_fft * h_w, dim=(-2, -1)).real