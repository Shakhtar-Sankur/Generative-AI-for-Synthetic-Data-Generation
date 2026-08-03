# Generative AI for Synthetic Data Generation

**A denoising diffusion probabilistic model (DDPM) in PyTorch, generating synthetic
medical images to address data scarcity for rare conditions.**

Rare disease datasets are small by definition, and the images cannot be shared freely.
A generative model trained on what exists can produce additional training material that
carries no patient identity.

## What's here

| File | Role |
|---|---|
| `ddpm.py` | `DDPM` — the noise-prediction network — and the linear beta schedule |
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

The network is a four-layer convolutional stack with a timestep embedding, not a U-Net.
That is enough to exercise the diffusion machinery end to end, and not enough for
competitive sample quality — a U-Net with skip connections and attention is what the
literature uses, and would be the next change worth making.

### Notes from a correctness pass

Three defects fixed, the first two fundamental:

- **The model could not run a forward pass.** The timestep embedding was added to the
  input: `self.network(x + t_embed)`. A `(B, 1, 64, 64)` image plus a `(B, 64, 1, 1)`
  embedding broadcasts to `(B, 64, 64, 64)`, and the first convolution expects one channel.
  The repository's own `test_ddpm.py` catches this, which means the tests had never been
  run. The embedding is now added after the input convolution, where the channel counts
  match.
- **Training and sampling disagreed about the objective.** `train.py` minimised
  `MSE(pred, images)`, teaching the network to output the clean image, while
  `inference.py` used its output as the predicted *noise* in the reverse process. Even with
  the forward pass fixed, the sampler could only have produced nonsense. Training now
  supervises the noise.
- **Both scripts hard-coded `.cuda()`** and left the beta schedule on the CPU, so indexing
  it with a CUDA timestep tensor failed. Device is now a `--device` flag defaulting to auto,
  and the schedule follows the model.

## Licence

Licensed under the GNU Affero General Public License v3.0. See `LICENSE`.

In short: you may use, modify and redistribute this, including over a network,
provided your derivative is released under the same licence.
