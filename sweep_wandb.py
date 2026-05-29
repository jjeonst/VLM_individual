"""Canonical W&B sweep entrypoint for TopoVLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or inspect a TopoVLM W&B sweep.")
    parser.add_argument("--sweep-config", required=True)
    parser.add_argument("--wandb-contract", required=True)
    parser.add_argument("--role-id", required=True)
    parser.add_argument("--create", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sweep_config = _load_yaml(Path(args.sweep_config))
    contract = _load_json(Path(args.wandb_contract))
    role = _resolve_role(contract, args.role_id)
    _validate_sweep_config(sweep_config, role)
    if not args.create:
        print(json.dumps({"status": "dry_run_ok", "role_id": args.role_id}, sort_keys=True))
        return
    import wandb

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=contract["canonical_entity"],
        project=role["wandb_project"],
    )
    print(json.dumps({"status": "created", "sweep_id": sweep_id}, sort_keys=True))


def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Sweep config must be a mapping: {path}")
    return loaded


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Contract must be a mapping: {path}")
    return loaded


def _resolve_role(contract: dict[str, object], role_id: str) -> dict[str, object]:
    roles = contract.get("roles")
    if not isinstance(roles, list):
        raise ValueError("W&B contract must contain roles list.")
    for role in roles:
        if isinstance(role, dict) and role.get("role_id") == role_id:
            return role
    raise ValueError(f"Missing W&B role: {role_id}")


def _validate_sweep_config(config: dict[str, object], role: dict[str, object]) -> None:
    command = config.get("command")
    if not isinstance(command, list) or "sweep_wandb.py" not in " ".join(
        str(item) for item in command
    ):
        raise ValueError("W&B sweep config command must route through sweep_wandb.py.")
    project = config.get("project") or config.get("run_project")
    if project is not None and project != role.get("wandb_project"):
        raise ValueError("Sweep config project does not match W&B contract role.")


if __name__ == "__main__":
    main()
