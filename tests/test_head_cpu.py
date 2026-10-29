"""CPU-only checks: head builds, a copied checkpoint loads with matching keys, inference runs."""
from pathlib import Path

import pytest
import torch

from age_dmoe.head import CLASS_NAMES, EMBED_DIM, DecoupledMixtureOfExpertsHead
from age_dmoe.infer import FOLDS, predict
from age_dmoe.preprocess import CROP_SIDE, age_to_band_id, load_age_stats

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
CKPT = WEIGHTS / "seed_42" / "fold_0" / "best_head.pt"


def test_state_dict_keys_match():
    head = DecoupledMixtureOfExpertsHead()
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    assert set(state.keys()) == set(head.state_dict().keys())
    head.load_state_dict(state)  # strict: shapes must match too


def test_forward_shape_and_probs():
    head = DecoupledMixtureOfExpertsHead().eval()
    head.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    _, p = head(torch.randn(2, EMBED_DIM), torch.zeros(2, 1), torch.tensor([0, 2]))
    assert p.shape == (2, 3)
    assert torch.allclose(p.sum(-1), torch.ones(2), atol=1e-5)


def test_band_and_crop_constants():
    assert [age_to_band_id(a) for a in (35, 36, 49, 50)] == [0, 1, 1, 2]
    assert CROP_SIDE == 358


def test_age_stats_cover_all_checkpoints():
    stats = load_age_stats()
    for seed in (42, 43, 44):
        for fold in FOLDS:
            assert (WEIGHTS / f"seed_{seed}" / f"fold_{fold}" / "best_head.pt").exists()
            assert stats[str(seed)][str(fold)]["age_std"] > 0


def test_predict_from_precomputed_embedding():
    torch.manual_seed(0)
    out = predict(torch.randn(1, EMBED_DIM), age=45.0, seed=42)
    assert out["prediction"] in CLASS_NAMES
    assert sum(out["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
