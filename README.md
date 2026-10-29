# Age-DMoE-CystNet

**Age-Decoupled Mixture-of-Experts with Soft Age Prior Loss for Three-Class Ultrasound Classification of Adnexal Masses**

Inference code and trained weights for the manuscript *"Age-Decoupled Mixture-of-Experts with Soft Age Prior Loss for Three-Class Ultrasound Classification of Adnexal Masses: A Single-Center Retrospective Study"* (Choe D, Kong H-J; under review).

The model classifies an adnexal mass on a grayscale B-mode ultrasound image as **Benign**, **Borderline** or **Malignant**, using the image and the patient's age.

> Research code only. It is not a medical device and must not be used for clinical decisions.

## Method

<!-- ASCII diagram mirrors Fig. 3 of the manuscript -->
```
ultrasound image ─► MedSigLIP-448 (frozen) ─► z (1152-d) ─┬─► expert 1 ─┐
                                                          ├─► expert 2 ─┼─► Σ π_k · logits_k ─► softmax ─► p
                                                          └─► expert 3 ─┘          ▲
age ─► [standardized age ; age-band one-hot] ─► e (64-d) ─► router([z ; e]) ─► π ──┘
```

- **Age-DMoE.** The three experts (two-layer MLP heads, 1152 → 256 → 3) receive only the image embedding `z`. Age reaches the model only through the router, which outputs the softmax mixing weights `π`. Age therefore changes *which* image experts are weighted, never the image features themselves.
- **SAP-Loss (Soft Age Prior Loss).** During training, `KL(q ‖ π)` pulls `π` toward a soft target `q = softmax(b / τ)` built from three age bands defined before model development (≤35, 36–49, ≥50 years; τ = 0.75, λ_SAP = 0.2). It is a training-time term only, so the inference head is the plain Age-DMoE head.
- **Encoder.** [`google/medsiglip-448`](https://huggingface.co/google/medsiglip-448) vision tower, frozen (`pooler_output`, 1152-d). Only the head is trained (0.89 M parameters).

## Results (single-center, temporally separated)

Mean ± SD over seeds 42, 43 and 44. The Validation Cohort is scored on the average of the five fold heads; the Derivation Cohort on 5-fold out-of-fold predictions.

| Model | Validation Macro-AUC (n = 59) | Derivation Macro-AUC (n = 1,171) |
|---|---|---|
| Image-Only | 0.803 ± 0.005 | 0.796 ± 0.005 |
| Age-DMoE | 0.868 ± 0.011 | 0.807 ± 0.006 |
| **Age-DMoE + SAP-Loss** | **0.890 ± 0.002** | **0.808 ± 0.007** |

On the Validation Cohort, Age-DMoE + SAP-Loss minus Image-Only = +0.087 (paired t-test over three seeds, p = 0.0022). The Validation Cohort is small (50 benign, 5 borderline, 4 malignant), and there is no external cohort yet.

## Repository layout

```
age_dmoe/
  head.py              Age-DMoE head (experts: z only; router: [z ; age embedding])
  preprocess.py        image crop/resize, MedSigLIP-448 embedding, age band and normalization
  infer.py             fold-ensemble inference and CLI
  age_norm_stats.json  per-seed, per-fold training age mean/std (no patient-level data)
weights/
  seed_{42,43,44}/fold_{0..4}/best_head.pt   15 trained heads
tests/
  test_head_cpu.py     CPU tests (no GPU, no encoder download)
```

## Installation

Python ≥ 3.10.

```bash
git clone https://github.com/dablro12/Age-DMoE-Cystnet.git
cd Age-DMoE-Cystnet
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The image path downloads MedSigLIP-448 from Hugging Face. You may have to accept its license and run `huggingface-cli login` first.

## Usage

Prediction = mean of the five fold-head probabilities for one seed (default 42), then argmax.

```bash
# from an image (runs the frozen encoder)
python -m age_dmoe --image path/to/ultrasound.png --age 45 --seed 42

# from a precomputed 1152-d MedSigLIP-448 embedding (.npy or .pt), no encoder needed
python -m age_dmoe --embedding z.npy --age 45
```

Example output (illustrative values):

```json
{
  "seed": 42,
  "age": 45.0,
  "probabilities": {"Benign": 0.81, "Malignant": 0.07, "Borderline": 0.12},
  "prediction": "Benign"
}
```

Python API:

```python
from age_dmoe.infer import predict
out = predict(z, age=45.0, seed=42)   # z: torch.Tensor of shape (1, 1152)
```

Preprocessing applied before the encoder: full frame (no lesion crop) resized to 448 × 448, center crop of 358 px (80%), resized back to 448 × 448. Logit order is `0 Benign, 1 Malignant, 2 Borderline`.

Patient-level prediction in the paper averages the class probabilities over all images of a patient.

## Tests

```bash
python -m pytest tests -q
```

## Data availability

The ultrasound images are not public because of patient privacy (IRB exemption no. B-2207-766-003, Seoul National University Bundang Hospital). This repository holds no images or patient identifiers.

## Citation

```bibtex
@article{choe2026agedmoe,
  title   = {Age-Decoupled Mixture-of-Experts with Soft Age Prior Loss for Three-Class Ultrasound Classification of Adnexal Masses: A Single-Center Retrospective Study},
  author  = {Choe, Daehyeon and Kong, Hyoun-Joong},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code and head weights: [MIT](LICENSE). The MedSigLIP-448 encoder is not redistributed here and is governed by its own terms of use.

## Contact

Daehyeon Choe — dablro12@snu.ac.kr
Hyoun-Joong Kong (corresponding) — gongcop7@snu.ac.kr
