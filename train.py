"""Canonical training and cache-building entrypoint for TopoVLM."""

from __future__ import annotations

import argparse
import json
import os


MODES = ("train", "build_cache", "preflight")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TopoVLM training or train-owned cache work.")
    parser.add_argument("--exp", required=True, help="Experiment config under configs/exp.")
    parser.add_argument("--mode", choices=MODES, default="train")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--allow-missing-data", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from configs import build_config_from_exp

    cfg = build_config_from_exp(args.exp, debug=args.debug)
    checkpoint_dir = args.checkpoint_dir or os.environ.get("CHECKPOINT_DIR")
    if checkpoint_dir is not None:
        cfg.output_root = checkpoint_dir

    if args.mode == "preflight":
        from evaluation.preflight import run_data_preflight

        result = run_data_preflight(cfg, allow_missing_data=args.allow_missing_data)
    elif args.mode == "build_cache":
        from data.habitat_cache import build_habitat_graph_cache

        result = build_habitat_graph_cache(cfg)
    elif args.mode == "train":
        from training.runner import run_training

        result = run_training(cfg)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
