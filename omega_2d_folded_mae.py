import torch
import torch.nn as nn
from torch.utils.data import Dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class SpatioTemporal2DMAE(nn.Module):
    def __init__(self, in_channels=3, days=15, ticks_per_day=64, embed_dim=128, mask_ratio=0.70):
        super().__init__()
        self.mask_ratio = mask_ratio
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim*2, kernel_size=3, stride=(2, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim*2, embed_dim*4, kernel_size=3, stride=(2, 2), padding=1),
            nn.GELU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim*4, embed_dim*2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim*2, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)
        )
        
        self.mask_token = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x):
        B, C, D, T = x.shape
        noise = torch.rand(B, 1, D, T, device=x.device)
        mask = (noise < self.mask_ratio).float()
        
        x_masked = x * (1 - mask) + self.mask_token * mask
        latent = self.encoder(x_masked)
        pred = self.decoder(latent)
        
        mse_per_window = ((pred - x) ** 2 * mask).sum(dim=[2, 3]) / (mask.sum(dim=[2, 3]) + 1e-9)
        mean_x = (x * mask).sum(dim=[2, 3], keepdim=True) / (mask.sum(dim=[2, 3], keepdim=True) + 1e-9)
        var_x_per_window = ((x - mean_x) ** 2 * mask).sum(dim=[2, 3]) / (mask.sum(dim=[2, 3]) + 1e-9)
        
        fvu_per_window = mse_per_window / (var_x_per_window + 1e-6)
        return mse_per_window.mean(), fvu_per_window.mean()

class TimeFoldedDataset(Dataset):
    def __init__(self, data_tensor, boundaries, days=15, ticks_per_day=64, physical_ticks_per_day=4800):
        self.data = data_tensor
        self.days = days
        self.ticks_per_day = ticks_per_day
        self.physical_ticks_per_day = physical_ticks_per_day
        
        self.intraday_stride = physical_ticks_per_day // ticks_per_day
        self.intraday_span = (ticks_per_day - 1) * self.intraday_stride + 1
        self.total_span = (days - 1) * physical_ticks_per_day + self.intraday_span
        self.valid_indices = []
        
        for start, end in boundaries:
            if end - start > self.total_span:
                self.valid_indices.extend(range(start, end - self.total_span))

        if len(self.valid_indices) == 0:
            raise ValueError(f"Span {self.total_span} is too large. Check ticker lengths.")
            
    def __len__(self): 
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        day_slices = []
        for d in range(self.days):
            d_start = start_idx + d * self.physical_ticks_per_day
            day_slice = self.data[d_start : d_start + self.intraday_span : self.intraday_stride]
            day_slices.append(day_slice)
            
        matrix_2d = torch.stack(day_slices, dim=0)
        matrix_2d = matrix_2d.permute(2, 0, 1).clone()
        return matrix_2d
