"""Inference: mean of five fold-head probabilities for one seed, then argmax."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .head import CLASS_NAMES, EMBED_DIM, DecoupledMixtureOfExpertsHead
from .preprocess import age_to_band_id, embed_image, load_age_stats, normalize_age

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
FOLDS = (0, 1, 2, 3, 4)


def load_head(path: Path, device: str = "cpu") -> DecoupledMixtureOfExpertsHead:
    head = DecoupledMixtureOfExpertsHead()
    head.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return head.to(device).eval()


def fold_probs(z: torch.Tensor, age: float, seed: int, fold: int, weights: Path, stats: dict, device: str):
    s = stats[str(seed)][str(fold)]
    age_norm = torch.tensor([[normalize_age(age, s["age_mean"], s["age_std"])]], dtype=torch.float32, device=device)
    band = torch.tensor([age_to_band_id(age)], dtype=torch.long, device=device)
    head = load_head(weights / f"seed_{seed}" / f"fold_{fold}" / "best_head.pt", device)
    with torch.no_grad():
        return head(z.to(device), age_norm, band)[1].cpu()


def predict(z: torch.Tensor, age: float, seed: int = 42, weights: Path = DEFAULT_WEIGHTS, device: str = "cpu") -> dict:
    """z: (1, 1152) MedSigLIP embedding. Returns mean probabilities and argmax class."""
    stats = load_age_stats()
    probs = torch.stack([fold_probs(z, age, seed, f, weights, stats, device) for f in FOLDS]).mean(dim=0)[0]
    idx = int(probs.argmax())
    return {
        "seed": seed,
        "age": age,
        "probabilities": {n: float(p) for n, p in zip(CLASS_NAMES, probs)},
        "prediction": CLASS_NAMES[idx],
    }


def load_embedding(path: Path) -> torch.Tensor:
    arr = torch.from_numpy(np.load(path)) if str(path).endswith(".npy") else torch.load(path, weights_only=True)
    z = arr.float().reshape(1, -1)
    if z.shape[1] != EMBED_DIM:
        raise ValueError(f"expected {EMBED_DIM}-d embedding, got {tuple(z.shape)}")
    return z


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Age-DMoE + SAP-Loss inference")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="ultrasound image (runs MedSigLIP-448)")
    src.add_argument("--embedding", type=Path, help="precomputed 1152-d vector (.npy or .pt)")
    p.add_argument("--age", type=float, required=True, help="patient age in years")
    p.add_argument("--seed", type=int, default=42, choices=(42, 43, 44))
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    z = embed_image(Image.open(a.image), a.device) if a.image else load_embedding(a.embedding)
    print(json.dumps(predict(z, a.age, a.seed, a.weights, a.device), indent=2))


if __name__ == "__main__":
    main()
