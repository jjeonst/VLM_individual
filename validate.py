"""Canonical validation, diagnostics, and offline evaluation entrypoint."""

from __future__ import annotations

import argparse
import json


RUNNERS = ("data_preflight", "cache_audit", "offline_policy_eval")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TopoVLM validation and diagnostics.")
    parser.add_argument("--runner", choices=RUNNERS, required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--allow-missing-data", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from configs import build_config_from_exp

    cfg = build_config_from_exp(args.exp, debug=args.debug)
    if args.runner == "data_preflight":
        from evaluation.preflight import run_data_preflight

        result = run_data_preflight(cfg, allow_missing_data=args.allow_missing_data)
    elif args.runner == "cache_audit":
        from evaluation.preflight import run_cache_audit

        result = run_cache_audit(cfg, allow_missing_data=args.allow_missing_data)
    elif args.runner == "offline_policy_eval":
        if args.checkpoint_dir is None:
            raise ValueError("offline_policy_eval requires --checkpoint-dir")
        from evaluation.offline_eval import run_offline_policy_eval

        result = run_offline_policy_eval(cfg, args.checkpoint_dir)
    else:
        raise ValueError(f"Unsupported runner: {args.runner}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
