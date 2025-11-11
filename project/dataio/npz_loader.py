# project/dataio/npz_loader.py
import numpy as np
from torch.utils.data import Dataset

class NPZDataset(Dataset):
    """
    Supports both:
      - fixed-length x: float32 array [N,T,F]
      - variable-length x: object array of length N with entries [Ti,F] (Ti can vary)

    Optional keys:
      - len: int32/int64 array [N] of true lengths (used when present)
      - mask: same shape as x (3D or object array) with 0/1 for observed values
      - y: labels [N] (ignored by the VAE but kept for compatibility)
    """
    def __init__(self, path: str):
        d = np.load(path, allow_pickle=True)

        if "x" not in d:
            raise ValueError(f"{path} missing required 'x' key")

        self.x = d["x"]                         # [N,T,F] or object array of [Ti,F]
        self.len = d["len"] if "len" in d else None
        self.mask = d["mask"] if "mask" in d else None
        self.y = d["y"] if "y" in d else None  # optional; not used by VAE

        self.n = len(self.x)

        # Basic sanity checks
        if isinstance(self.x, np.ndarray) and self.x.dtype != object:
            if self.x.ndim != 3:
                raise ValueError(f"'x' must be [N,T,F] if not object array, got shape {self.x.shape}")
        if self.len is not None and len(self.len) != self.n:
            raise ValueError("'len' must have length N")

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        xi = self.x[i]
        # Ensure float32 arrays per item
        if isinstance(xi, np.ndarray):
            x = xi.astype(np.float32, copy=False)
        else:
            # object arrays sometimes store lists COERCE
            x = np.asarray(xi, dtype=np.float32)

        item = {"x": x}

        # true length
        if self.len is not None:
            item["len"] = int(self.len[i])
        else:
            item["len"] = int(x.shape[0])

        # optional mask
        if self.mask is not None:
            mi = self.mask[i]
            m = mi.astype(np.float32, copy=False) if isinstance(mi, np.ndarray) else np.asarray(mi, dtype=np.float32)
            item["mask"] = m

        # optional label passthrough (ignored by trainer)
        if self.y is not None:
            item["y"] = self.y[i]

        return item
