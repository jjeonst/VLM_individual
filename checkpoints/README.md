# Checkpoints

Checkpoint payloads are ignored by git. Every retained learned run should write a
`checkpoint_manifest.json` next to the checkpoint payloads. Slurm jobs may set
`CHECKPOINT_DIR`; otherwise local training writes below this directory.
