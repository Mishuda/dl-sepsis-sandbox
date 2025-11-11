# Produces a tiny synthetic dataset [N,T,F] to unblock the pipeline.
import numpy as np, os
os.makedirs("data/processed", exist_ok=True)
N,T,F = 256, 12, 8
x = np.random.randn(N,T,F).astype("float32")
mask = np.ones_like(x, dtype="float32")
y = (x.mean(axis=(1,2)) > 0).astype("int64")
np.savez("data/processed/train.npz", x=x, y=y, mask=mask)
np.savez("data/processed/val.npz",   x=x[:64], y=y[:64], mask=mask[:64])
print("Wrote data/processed/train.npz and val.npz")
