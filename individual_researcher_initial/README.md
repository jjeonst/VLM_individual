Initial VLM-Habitat code.

This folder is reference-only source lineage from the individual researcher
prototype imported before the canonical TopoVLM scaffold was created.

Do not import this folder from canonical TopoVLM modules. Do not submit the
prototype `.slurm` files as project jobs. The scripts use hardcoded local paths,
ad hoc CLIP feature extraction, dummy goal text, and hand-written scheduler
wrappers that are intentionally outside the canonical `train.py`, `validate.py`,
`sweep_wandb.py`, `configs/`, and `slurm/habitat/` contract.
