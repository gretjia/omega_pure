import torch
class EpiplexityShardDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, seq_len, stride):
        self.data = data_tensor
        self.seq_len = seq_len
        self.stride = stride
        self.span = (seq_len - 1) * stride + 1
    def __len__(self):
        return len(self.data) - self.span
    def __getitem__(self, idx):
        return self.data[idx : idx + self.span : self.stride]

data = torch.randn(1000000, 3)
ds = EpiplexityShardDataset(data, 128, 499)
print(f"Dataset length: {len(ds)}")
loader = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True, num_workers=0)
import time
t0 = time.time()
for i, batch in enumerate(loader):
    if i % 100 == 0:
        print(f"Batch {i} loaded in {time.time()-t0:.4f}s")
        t0 = time.time()
    if i > 500: break
