# Generative AI for Synthetic Data Generation

**A denoising diffusion probabilistic model (DDPM) in PyTorch, generating synthetic
medical images to address data scarcity for rare conditions.**

Rare disease datasets are small by definition, and the images cannot be shared freely.
A generative model trained on what exists can produce additional training material that
carries no patient identity.

## What's here

| File | Role |
|---|---|
| `ddpm.py` | The `DDPM` model and its noise schedule |
| `train.py` | Training loop with checkpointing |
| `inference.py` | Sampling from a trained checkpoint |
| `bigquery_upload.py` | Pushing generated metadata to BigQuery |
| `run_analytics.py`, `analytics.sql` | Queries over the generated set |
| `test_ddpm.py`, `test_inference.py`, `test_bigquery.py` | Unit tests |
| `setup.py` | Package definition |

## Design targets

- Visual realism good enough to be useful as training data, measured by FID
- A measurable accuracy gain in a downstream classifier trained with the synthetic set
- Batch generation fast enough to be practical

## On the numbers

The figures above are **design targets** that shaped the implementation — they are not
measured results. This repository ships no benchmark harness and no trained weights, so
nothing here reproduces them. They are recorded because they drove real decisions about
architecture and algorithm choice, not as claims about observed performance.

## Running it

```bash
pip install -r requirements.txt
python train.py --dataset data/mednist --epochs 50
python inference.py --checkpoint models/checkpoints/epoch_49.pth
pytest
```

Training writes checkpoints to `models/checkpoints/`, which is created on start. Requires
a CUDA device.

## Status

Complete training and sampling pipeline with tests. Trained checkpoints are not committed.

## Licence

All rights reserved. Published for reading, not for reuse.
