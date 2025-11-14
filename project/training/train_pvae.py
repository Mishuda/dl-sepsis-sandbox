#!/usr/bin/env python3
# train_pvae.py

import os, sys, math, time, random, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import contextlib


# Allow importing from the pVAE submodule (external/pvae/*)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUBMOD_PVAE = os.path.join(REPO_ROOT, "external", "pvae")
if SUBMOD_PVAE not in sys.path:
    sys.path.insert(0, SUBMOD_PVAE)

#
# Dataset
# Expect NPZDataset to yield dicts with at least {"x": np.ndarray[T,F]}
# Optionally {"len": int, "mask": np.ndarray[T,F]}. Dtypes can be raw; we cast here.

from project.dataio.npz_loader import NPZDataset



# Reproducibility
def set_seed(s: int = 1337) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)



# Model

class LSTMEncoder(nn.Module):
    def __init__(self, in_dim: int, h: int = 64, z: int = 16,
                 nlayer: int = 1, bidir: bool = False, dropout: float = 0.0):
        super().__init__()
        self.bidir = bidir
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=h,
            num_layers=nlayer,
            batch_first=True,
            bidirectional=bidir,
            dropout=(dropout if nlayer > 1 else 0.0),
        )
        out_h = h * (2 if bidir else 1)
        self.mu = nn.Linear(out_h, z)
        self.lv = nn.Linear(out_h, z)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: [B,T,F], lengths: [B] (CPU or CUDA Long ok for packing on CPU lengths)
        packed = pack_padded_sequence(
            x, lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.rnn(packed)  # hn: [L*num_dir, B, H]
        if self.bidir:
            # last layer forward/backward
            h_last = torch.cat([hn[-2], hn[-1]], dim=-1)
        else:
            h_last = hn[-1]
        return self.mu(h_last), self.lv(h_last)


class MLPDecoder(nn.Module):
    def __init__(self, out_dim: int, z: int = 16, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z, h),
            nn.ReLU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# Losses / Latent utils
def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # mean over batch of per-sample KL to N(0, I)
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()


def mse_masked(xhat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = ((xhat - x) ** 2 * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den


def sample_and_kl(mu: torch.Tensor, lv: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    if args.manifold == "euclid":
        scale = torch.exp(0.5 * lv)
        q = torch.distributions.Normal(mu, scale)
        z = q.rsample()
        kl = kl_normal(mu, lv)
        return z, kl

    # Poincaré    
    else:
        #
        from pvae.manifolds.poincareball import PoincareBall
        from pvae.distributions.wrapped_normal import WrappedNormal

        # M = PoincareBall(mu.size(-1), c=args.curv)
        # mu = mu.float()
        # q = WrappedNormal(mu, torch.exp(0.5 * lv), M)
        M = PoincareBall(dim=mu.shape[-1], c=args.curv)
        mu = M.projx(mu)  # project onto valid Poincaré ball
        q = WrappedNormal(mu, torch.exp(0.5 * lv), M)

        lv = lv.float()

        p = WrappedNormal(torch.zeros_like(mu), torch.ones_like(mu), M)

        z = q.rsample()
        kl = (q.log_prob(z) - p.log_prob(z)).sum(-1).mean()
        return z, kl



# Collate: pad/clip to a fixed global T_pad (computed from TRAIN set)
# If item has its own mask, we use it and pad/clip; else we create mask=1 up to length.

def make_collate_pad(T_pad: int):
    def collate_pad(batch):
        # batch: list of dicts with "x" (np.ndarray[T_i,F]) and optional "len","mask"
        B = len(batch)
        # infer F from first item
        F = int(batch[0]["x"].shape[1])
        xpad = torch.zeros(B, T_pad, F, dtype=torch.float32)
        mpad = torch.zeros(B, T_pad, F, dtype=torch.float32)
        lens = torch.empty(B, dtype=torch.long)

        for i, b in enumerate(batch):
            x_np = b["x"].astype(np.float32)
            Ti = int(x_np.shape[0])
            # clip if longer than T_pad
            t = min(Ti, T_pad)
            xpad[i, :t] = torch.from_numpy(x_np[:t])

            if "mask" in b:
                m_np = b["mask"].astype(np.float32)
                mpad[i, :t] = torch.from_numpy(m_np[:t, :F])
            else:
                mpad[i, :t] = 1.0

            if "len" in b:
                lens[i] = min(int(b["len"]), T_pad)
            else:
                lens[i] = t

        return {"x": xpad, "mask": mpad, "len": lens}
    return collate_pad


# Utility: infer global T_pad and F from the TRAIN dataset
# Works with fixed-length (dense) or object array datasets

def infer_TF(dataset: NPZDataset, scan_limit: int = 10000) -> tuple[int, int]:
    # Try fast path if dataset exposes .x
    if hasattr(dataset, "x"):
        x = dataset.x
        if isinstance(x, np.ndarray) and x.dtype == object:
            max_T = 0
            F = int(x[0].shape[1])
            for i, xi in enumerate(x):
                if i >= scan_limit: break
                max_T = max(max_T, int(xi.shape[0]))
            return int(max_T), F
        elif isinstance(x, np.ndarray) and x.ndim == 3:
            # fixed-length [N,T,F]
            return int(x.shape[1]), int(x.shape[2])

    # Fallback: sample items
    max_T, F = 0, None
    n = len(dataset)
    for i in range(min(n, scan_limit)):
        item = dataset[i]
        Ti, Fi = int(item["x"].shape[0]), int(item["x"].shape[1])
        max_T = max(max_T, Ti)
        F = Fi if F is None else F
    if F is None:
        raise RuntimeError("Could not infer T and F from dataset.")
    return int(max_T), int(F)


# evaluation (ELBO = recon + beta*KL) with masking

@torch.no_grad()
def eval_elbo(enc: nn.Module, dec: nn.Module, loader: DataLoader, device, args) -> float:
    enc.eval(); dec.eval()
    tot, n = 0.0, 0
    for b in loader:

        x = b["x"].to(device, non_blocking=True).float()      # [B,T_pad,F]
        m = b["mask"].to(device, non_blocking=True).float()   # [B,T_pad,F]
        mu, lv = enc(x, b["len"])
        z, kl = sample_and_kl(mu, lv, args)
        # decoder was built for out_dim = T_pad*F
        B, T_pad, F = x.shape
        xhat = dec(z).view(B, T_pad, F)
        recon = mse_masked(xhat, x, m)
        elbo = recon + args.beta * kl
        tot += elbo.item() * B
        n += B
    return tot / max(n, 1)



# Main

def main(args):
    set_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    tr = NPZDataset(args.data)
    va = NPZDataset(args.val)

    # Infer global T_pad from TRAIN, and F
    T_pad, F = infer_TF(tr)
    # Build collate that always pads/clips to train T_pad; validation must fit within it.
    collate_pad = make_collate_pad(T_pad)
    print(f"[data] T_pad={T_pad}, F={F} | train_N={len(tr)}, val_N={len(va)} | device={device.type}")


    # Loaders
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True,
                          collate_fn=collate_pad, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(va, batch_size=args.batch, shuffle=False,
                          collate_fn=collate_pad, pin_memory=(device.type=="cuda"))

    

    # Models
    enc = LSTMEncoder(
        in_dim=F, h=args.hid, z=args.latent,
        nlayer=args.nlayer, bidir=args.bidir, dropout=args.drop
    ).to(device)

    dec = MLPDecoder(out_dim=F * T_pad, z=args.latent, h=args.hid).to(device)
    assert dec.net[-1].out_features == F*T_pad, "Decoder output dim mismatch"

    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=args.lr)


    # Checkpoints
    ckpt_dir = Path("runs/ckpts"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf
    use_amp = (args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else torch.amp.GradScaler(enabled=False)

    # Training loop
    for epoch in range(args.epochs):
        enc.train(); dec.train()
        sum_recon = 0.0; sum_kl = 0.0; sum_loss = 0.0; n_seen = 0

        beta_now = args.beta * min(1.0, (epoch + 1) / max(1, args.kl_warm))

        for b in train_loader:
            # x = b["x"].to(device).float()
            # m = b["mask"].to(device).float()
            # lengths = b["len"]



            # opt.zero_grad(set_to_none=True)
            # amp_ctx = torch.amp.autocast('cuda') if use_amp else contextlib.nullcontext()
            # with amp_ctx:
            #     mu, lv = enc(x, lengths)
            #     z, kl = sample_and_kl(mu, lv, args)
            #     B, T, F_ = x.shape
            #     xhat = dec(z).view(B, T, F_)
            #     recon = mse_masked(xhat, x, m)
            #     loss = recon + beta_now * kl

            #     #
            #     x = b["x"].to(device, non_blocking=True).float()
            #     m = b["mask"].to(device, non_blocking=True).float()
            # device copies (single time, non_blocking)
            x = b["x"].to(device, non_blocking=True).float()
            m = b["mask"].to(device, non_blocking=True).float()
            lengths = b["len"]

            opt.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast('cuda') if use_amp else contextlib.nullcontext()
            with amp_ctx:
                #
                mu, lv = enc(x, lengths)
                z, kl = sample_and_kl(mu, lv, args)
                B, T, F_ = x.shape
                xhat = dec(z).view(B, T, F_)
                recon = mse_masked(xhat, x, m)
                loss = recon + beta_now * kl


            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_([*enc.parameters(), *dec.parameters()], args.clip)
            scaler.step(opt); scaler.update()

            bs = x.size(0)
            n_seen += bs
            sum_recon += recon.item() * bs
            sum_kl    += kl.item() * bs
            sum_loss  += loss.item() * bs

            if torch.isnan(recon) or torch.isnan(kl) or torch.isnan(loss):
                raise RuntimeError("NaN detected in training step.")

        train_recon = sum_recon / n_seen
        train_kl = sum_kl / n_seen
        train_loss = sum_loss / n_seen

        val_elbo = eval_elbo(enc, dec, val_loader, device, args)

        print(f"epoch {epoch:03d} | train: recon={train_recon:.4f} kl={train_kl:.4f} loss={train_loss:.4f} "
              f"| val_elbo={val_elbo:.4f} | beta={beta_now:.3f}")

        # Save last + best
        last_payload = {
            "enc": enc.state_dict(),
            "dec": dec.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "T_pad": T_pad,
            "F": F,
            "val_elbo": val_elbo,
        }
        torch.save(last_payload, ckpt_dir / "last.pt")
        if val_elbo < best_val:
            best_val = val_elbo
            best_payload = dict(last_payload)
            best_payload["best_val"] = best_val
            torch.save(best_payload, ckpt_dir / "best.pt")

    print("Training complete.")

# CLI

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",    type=str, default="data/processed/train.npz")
    p.add_argument("--val",     type=str, default="data/processed/val.npz")
    p.add_argument("--latent",  type=int, default=16)
    p.add_argument("--hid",     type=int, default=64)
    p.add_argument("--epochs",  type=int, default=10)
    p.add_argument("--batch",   type=int, default=64)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--beta",    type=float, default=1.0)
    p.add_argument("--manifold", type=str, default="euclid", choices=["euclid", "poincare"])
    p.add_argument("--curv",    type=float, default=0.7)
    p.add_argument("--nlayer",  type=int, default=1)
    p.add_argument("--bidir",   action="store_true")
    p.add_argument("--drop",    type=float, default=0.0)
    p.add_argument("--clip",    type=float, default=1.0)
    p.add_argument("--amp",     action="store_true")
    p.add_argument("--kl_warm", type=int, default=5, help="epochs to reach target beta")
    args = p.parse_args()
    main(args)
