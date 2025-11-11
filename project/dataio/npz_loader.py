import numpy as np, torch
from torch.utils.data import Dataset

class NPZDataset(Dataset):
    def __init__(self, path):
        d = np.load(path)
        self.x = torch.tensor(d["x"]).float()   # [N,T,F]
        self.y = torch.tensor(d["y"]).long()    # [N]  (0/1)
        self.mask = torch.tensor(d["mask"]).float() if "mask" in d else None
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i):
        out = {"x": self.x[i], "y": self.y[i]}
        if self.mask is not None: out["mask"] = self.mask[i]
        return out
