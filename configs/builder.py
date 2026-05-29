"""EGO-style YAML composition for TopoVLM configs."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from configs.schema import TopoVLMConfig


REQUIRED_DOMAINS = ("train", "data", "model", "objectives", "eval")


def build_config_from_exp(exp: str, *, debug: bool = False) -> TopoVLMConfig:
    exp_path = _resolve_experiment_path(exp)
    exp_cfg = _load_yaml(exp_path)
    defaults = exp_cfg.pop("defaults", None)
    selections = _parse_defaults(defaults)

    merged: dict[str, Any] = {}
    for domain, name in selections:
        merged = _deep_merge(merged, _load_domain_config(domain, name))
    merged = _deep_merge(merged, exp_cfg)
    merged.setdefault("config_name", _config_name_from_path(exp_path))
    if debug:
        _apply_debug_overrides(merged)
    return _dataclass_from_dict(TopoVLMConfig, merged)


def _base_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_experiment_path(exp: str) -> Path:
    target = Path(exp)
    if target.suffix not in {".yaml", ".yml"}:
        target = target.with_suffix(".yaml")
    candidates = []
    if target.is_absolute():
        candidates.append(target)
    else:
        base = _base_dir() / "exp"
        candidates.extend((Path.cwd() / target, base / target, base / target.name))
        if len(target.parts) == 1:
            candidates.extend(sorted(base.glob(f"**/{target.name}")))
    matches = [candidate.resolve() for candidate in candidates if candidate.exists()]
    unique = []
    for match in matches:
        if match not in unique:
            unique.append(match)
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise ValueError(f"Ambiguous experiment config: {exp}")
    raise FileNotFoundError(f"Missing experiment config: {exp}")


def _resolve_domain_path(domain: str, name: str) -> Path:
    if domain not in REQUIRED_DOMAINS:
        raise ValueError(f"Unsupported config domain: {domain}")
    target = Path(name)
    if target.suffix not in {".yaml", ".yml"}:
        target = target.with_suffix(".yaml")
    base = _base_dir() / domain
    candidates = [base / target, base / target.name]
    if len(target.parts) == 1:
        candidates.extend(sorted(base.glob(f"**/{target.name}")))
    matches = [candidate.resolve() for candidate in candidates if candidate.exists()]
    unique = []
    for match in matches:
        if match not in unique:
            unique.append(match)
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise ValueError(f"Ambiguous {domain} config: {name}")
    raise FileNotFoundError(f"Missing {domain} config: {name}")


def _load_domain_config(domain: str, name: str) -> dict[str, Any]:
    base_default = _base_dir() / domain / "default.yaml"
    configs = []
    if base_default.exists():
        configs.append(_load_yaml(base_default))
    selected_path = _resolve_domain_path(domain, name)
    if selected_path != base_default:
        configs.append(_load_yaml(selected_path))
    merged: dict[str, Any] = {}
    for cfg in configs:
        merged = _deep_merge(merged, cfg)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return loaded


def _parse_defaults(defaults: Any) -> list[tuple[str, str]]:
    if not isinstance(defaults, list):
        raise ValueError("Experiment config must declare a defaults list.")
    seen: set[str] = set()
    selections = []
    for entry in defaults:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError("Each defaults entry must map one domain to one name.")
        domain, name = next(iter(entry.items()))
        if domain in seen:
            raise ValueError(f"Duplicate defaults domain: {domain}")
        if domain not in REQUIRED_DOMAINS:
            raise ValueError(f"Unsupported defaults domain: {domain}")
        seen.add(domain)
        selections.append((domain, str(name)))
    missing = set(REQUIRED_DOMAINS).difference(seen)
    if missing:
        raise ValueError(f"Missing defaults domains: {', '.join(sorted(missing))}")
    return selections


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_debug_overrides(raw: dict[str, Any]) -> None:
    raw["debug"] = True
    raw["wandb"] = False
    raw["max_epochs"] = 1
    raw["save_every_epochs"] = 1
    data = raw.setdefault("data", {})
    data["synthetic_debug"] = True
    data["max_episodes"] = 4
    data["batch_size"] = 2
    data["num_workers"] = 0
    model = raw.setdefault("model", {})
    policy = model.setdefault("policy", {})
    policy["hidden_dim"] = min(int(policy.get("hidden_dim", 128)), 128)
    policy["transformer_layers"] = 1


def _dataclass_from_dict(cls: type[Any], raw: dict[str, Any]) -> Any:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"Expected dataclass type: {cls}")
    fields = {field.name: field for field in dataclasses.fields(cls)}
    type_hints = get_type_hints(cls)
    unknown = set(raw).difference(fields)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name in fields:
        if name in raw:
            kwargs[name] = _coerce_value(type_hints[name], raw[name])
    return cls(**kwargs)


def _coerce_value(field_type: Any, value: Any) -> Any:
    origin = get_origin(field_type)
    args = get_args(field_type)
    if dataclasses.is_dataclass(field_type):
        if not isinstance(value, dict):
            raise ValueError(f"Expected mapping for {field_type.__name__}")
        return _dataclass_from_dict(field_type, value)
    if origin is list and args:
        if value is None:
            return None
        return list(value)
    if origin is dict:
        if value is None:
            return None
        return dict(value)
    return value


def _config_name_from_path(path: Path) -> str:
    exp_root = _base_dir() / "exp"
    try:
        return str(path.relative_to(exp_root).with_suffix(""))
    except ValueError:
        return path.stem
