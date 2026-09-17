# Sepsis Representation Learning Sandbox

Experimental PyTorch code for comparing **Euclidean and Poincaré latent representations** of multivariate clinical time series.

This repository was developed as a deep-learning project on sepsis trajectory representation. It is intentionally kept as a **research sandbox** rather than presented as a production pipeline.

## Project idea

The project explores whether a hyperbolic latent space can provide a useful alternative to a conventional Euclidean VAE representation for heterogeneous patient trajectories.

The current training code combines:

- an **LSTM encoder** for variable-length multivariate time series
- a simple MLP decoder
- Euclidean Gaussian latent variables or a **Poincaré-ball wrapped-normal** latent distribution
- masked reconstruction loss for incomplete/padded observations
- KL weighting with optional warm-up
- reproducible random seeds
- train/validation evaluation and checkpointing

## Repository structure

```text
project/
├── dataio/
│   └── npz_loader.py          # NPZ dataset loader
├── scripts/
│   └── preprocess_toy.py      # Generate toy data for pipeline checks
└── training/
    └── train_pvae.py          # Euclidean/Poincaré VAE training

external/
└── pvae/                      # Upstream Poincaré VAE code as a git submodule

envs/
└── pvae-sepsis.yml            # Minimal development environment metadata
```

The Poincaré implementation is linked as a submodule from the original [`emilemathieu/pvae`](https://github.com/emilemathieu/pvae) repository.

## Data format

The training script expects NPZ datasets containing time-series arrays. At minimum, each item must provide:

- `x`: observations with shape `T × F`
- optionally `len`: the valid sequence length
- optionally `mask`: observed/missing-value mask with shape `T × F`

Training and validation paths default to:

```text
data/processed/train.npz
data/processed/val.npz
```

**Clinical/source datasets are not included in this repository.** The `data/` directory and model checkpoints are excluded through `.gitignore`.

## Smoke test with synthetic data

Generate a small toy dataset:

```bash
python project/scripts/preprocess_toy.py
```

Then run the Euclidean model:

```bash
python project/training/train_pvae.py --manifold euclid --epochs 10
```

Or the Poincaré version:

```bash
python project/training/train_pvae.py --manifold poincare --curv 0.7 --epochs 10
```

`--manifold` selects the latent geometry, `--curv` sets the Poincaré-ball curvature parameter, and `--epochs` controls the number of training epochs.

## Main training options

| Argument | Purpose | Default |
| --- | --- | ---: |
| `--latent` | Latent dimension | 16 |
| `--hid` | Hidden dimension | 64 |
| `--batch` | Batch size | 64 |
| `--lr` | Learning rate | 3e-4 |
| `--beta` | KL weight | 1.0 |
| `--manifold` | `euclid` or `poincare` | `euclid` |
| `--curv` | Poincaré curvature | 0.7 |
| `--kl_warm` | Epochs used to warm up the KL weight | 5 |
| `--bidir` | Use a bidirectional LSTM encoder | off |
| `--amp` | Enable mixed precision on CUDA | off |

## Status

This repository captures an experimental stage of the project. It is useful as a record of the model adaptation and training logic, but it should not be interpreted as a validated clinical model or a complete reproducible clinical-data analysis.
