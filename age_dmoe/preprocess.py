"""Age and image preprocessing (functional helpers)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

IMAGE_SIZE = 448
CROP_SCALE = 0.8
CROP_SIDE = int(IMAGE_SIZE * CROP_SCALE)  # 358
ENCODER_ID = "google/medsiglip-448"
_STATS_PATH = Path(__file__).with_name("age_norm_stats.json")


def age_to_band_id(age: float) -> int:
    """<=35 -> 0, 36-49 -> 1, >=50 -> 2."""
    if age <= 35:
        return 0
    if age >= 50:
        return 2
    return 1


def load_age_stats(path: Path = _STATS_PATH) -> dict:
    return json.loads(Path(path).read_text())


def normalize_age(age: float, mean: float, std: float) -> float:
    """(age - train_mean) / train_std, population std of the fold's training set."""
    return (float(age) - mean) / std


def prepare_image(img: Image.Image) -> Image.Image:
    """Resize full frame to 448, center-crop 358 (=int(448*0.8)), resize back to 448."""
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    left = (IMAGE_SIZE - CROP_SIDE) // 2
    top = (IMAGE_SIZE - CROP_SIDE) // 2
    img = img.crop((left, top, left + CROP_SIDE, top + CROP_SIDE))
    return img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)


def embed_image(img: Image.Image, device: str = "cpu") -> torch.Tensor:
    """Frozen MedSigLIP-448 vision pooler_output, shape (1, 1152). Downloads the encoder."""
    from transformers import AutoImageProcessor, SiglipVisionModel

    processor = AutoImageProcessor.from_pretrained(ENCODER_ID)
    model = SiglipVisionModel.from_pretrained(ENCODER_ID).to(device).eval()
    inputs = processor(images=prepare_image(img), return_tensors="pt").to(device)
    with torch.no_grad():
        return model(**inputs).pooler_output.float().cpu()
