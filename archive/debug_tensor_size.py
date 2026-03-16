import torch
# 1 shard is approx 72 million rows.
# Trial 12 was requesting 26 shards = 26 * 72M = ~1.87 Billion rows.
# Let's calculate raw tensor size in memory.
rows = 1.87 * 10**9
cols = 3
bytes_per_float32 = 4

total_bytes = rows * cols * bytes_per_float32
print(f"Raw Tensor Size: {total_bytes / (1024**3):.2f} GB")
