import argparse, os, sys, torch, torch.nn as nn
from torch.utils.data import DataLoader
from project.dataio.npz_loader import NPZDataset

# allow importing from the pVAE submodule
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUBMOD_PVAE = os.path.join(REPO_ROOT, "external", "pvae")
if SUBMOD_PVAE not in sys.path: sys.path.insert(0, SUBMOD_PVAE)

class LSTMEncoder(nn.Module):
    def __init__(self, in_dim, h=64, z=16):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, h, batch_first=True)
        self.mu  = nn.Linear(h, z)
        self.lv  = nn.Linear(h, z)
    def forward(self, x):
        h,_ = self.rnn(x)          # [B,T,H]
        h = h[:, -1]               # last step
        return self.mu(h), self.lv(h)

class MLPDecoder(nn.Module):
    def __init__(self, out_dim, z=16, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z,h), nn.ReLU(), nn.Linear(h,out_dim))
    def forward(self, z): return self.net(z)

def kl_normal(mu, logvar):
    # per-sample KL to N(0,I), mean over batch
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()

@torch.no_grad()
def eval_elbo(enc, dec, loader, device, args):
    enc.eval(); dec.eval()
    tot = 0; n = 0
    for batch in loader:
        x = batch["x"].to(device).float()       # [B,T,F]
        mu, lv = enc(x)
        if args.manifold == "euclid":
            scale = torch.exp(0.5*lv)
            q = torch.distributions.Normal(mu, scale)
            z = q.rsample()
            kl = kl_normal(mu, lv)
        else:
            from pvae.manifolds.poincareball import PoincareBall
            from pvae.distributions.wrapped_normal import WrappedNormal
            M = PoincareBall(c=args.curv)
            q = WrappedNormal(M, mu, torch.exp(0.5*lv))
            p = WrappedNormal(M, torch.zeros_like(mu), torch.ones_like(mu))
            z = q.rsample()
            kl = (q.log_prob(z) - p.log_prob(z)).sum(-1).mean()
        xhat = dec(z).reshape_as(x)
        recon = ((xhat - x)**2).mean()
        tot += (recon + args.beta*kl).item() * x.size(0); n += x.size(0)
    return tot / max(n,1)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = NPZDataset(args.data); va = NPZDataset(args.val)
    T, F = tr.x.shape[1], tr.x.shape[2]

    enc = LSTMEncoder(F, args.hid, args.latent).to(device)
    dec = MLPDecoder(F*T, args.latent, args.hid).to(device)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=args.lr)

    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(va, batch_size=args.batch)

    for epoch in range(args.epochs):
        enc.train(); dec.train()
        for batch in train_loader:
            x = batch["x"].to(device).float()   # [B,T,F]
            mu, lv = enc(x)
            if args.manifold == "euclid":
                q = torch.distributions.Normal(mu, torch.exp(0.5*lv))
                z = q.rsample()
                kl = kl_normal(mu, lv)
            else:
                from pvae.manifolds.poincareball import PoincareBall
                from pvae.distributions.wrapped_normal import WrappedNormal
                M = PoincareBall(c=args.curv)
                q = WrappedNormal(M, mu, torch.exp(0.5*lv))   # (M, loc, scale)
                p = WrappedNormal(M, torch.zeros_like(mu), torch.ones_like(mu))
                z = q.rsample()
                kl = (q.log_prob(z) - p.log_prob(z)).sum(-1).mean()

            xhat = dec(z).reshape_as(x)
            recon = ((xhat - x)**2).mean()
            loss = recon + args.beta * kl

            opt.zero_grad(); loss.backward(); opt.step()

        val_elbo = eval_elbo(enc, dec, val_loader, device, args)
        print(f"epoch {epoch}: recon={recon.item():.4f} kl={kl.item():.4f} loss={loss.item():.4f}  val_elbo={val_elbo:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="data/processed/train.npz")
    p.add_argument("--val",    default="data/processed/val.npz")
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--hid",    type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch",  type=int, default=64)
    p.add_argument("--lr",     type=float, default=3e-4)
    p.add_argument("--beta",   type=float, default=1.0)
    p.add_argument("--manifold", type=str, default="euclid", choices=["euclid","poincare"])
    p.add_argument("--curv",   type=float, default=0.7)
    args = p.parse_args(); main(args)
