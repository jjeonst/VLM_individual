# Artifacts

This directory stores durable repo-owned records such as contracts, audits, and
small manifests. Training, evaluation, model, and data code must not import from
`artifacts/` or depend on artifact records as runtime implementation modules.

Large payloads, logs, W&B local state, checkpoints, and generated evidence are
ignored by default. Track only reviewed records that are intentionally part of
the repo contract.
