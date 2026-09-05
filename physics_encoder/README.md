# MoPE-JEPA Physics Encoder

`mope-jepa/` contains the source snapshot used by PhyWAM 5-v3.6 for physical representation learning, event-aware fine-tuning, and online physical-token extraction.

Included:

- MoPE-JEPA model and training implementation
- RoboTwin event dataset and evaluation code
- Physical feature v3.6 extraction pipeline
- Training and validation scripts
- Small configuration and metric files

Excluded from Git:

- Model checkpoints and optimizer states
- Training outputs, logs, and W&B runs
- Generated features, cached datasets, and Python caches

At runtime, set `MOPE_REPO` to this directory and provide the excluded artifacts explicitly:

```bash
export MOPE_REPO="$PWD/physics_encoder/mope-jepa"
export MOPE_CKPT=/path/to/mope-jepa-checkpoint.pth
export PHYS_EVENT_LABEL_PATH=/path/to/event-labels.json
```
