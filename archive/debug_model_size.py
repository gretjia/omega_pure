import torch
import torch.nn as nn
class EpiplexityMAE(nn.Module):
    def __init__(self, feature_dim=3, seq_len=64, embed_dim=64, num_heads=4, depth=4, mask_ratio=0.70):
        super().__init__()
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # PyTorch 2.x 会在此自动路由至 FlashAttention 节约显存
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_proj = nn.Linear(embed_dim, feature_dim)

    def forward(self, x):
        B, T, F = x.shape
        x_emb = self.input_proj(x) + self.pos_embed
        noise = torch.rand(B, T, device=x.device)
        mask = (noise < self.mask_ratio).bool()
        x_masked = x_emb.clone()
        x_masked[mask] = self.mask_token
        latent = self.encoder(x_masked)
        pred = self.output_proj(latent)
        return pred

model = EpiplexityMAE(seq_len=128)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")

# Simulate Forward Pass Memory
batch = torch.randn(512, 128, 3)
try:
    pred = model(batch)
    print(f"Forward Pass Success with Batch 512. Pred Shape: {pred.shape}")
except Exception as e:
    print(f"Forward Pass Failed: {e}")
