"""Export HM3D ObjectNav scene top-down trajectory figures."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from analysis.code.hm3d_trajectory_notebook import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_EXP,
    plot_habitat_topdown,
    replay_selected_objectnav_scene_topdowns,
)  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "analysis/results/hm3d_scene_topdown_trajectories/habitat_pr2l_hm3d_bc"
)


def write_scene_topdown_pngs(
    scene_replays: list[dict[str, object]],
    output_dir: str | Path,
    *,
    data_root: str | Path,
    exp: str,
    dpi: int,
    command: list[str],
) -> dict[str, object]:
    output_path = Path(output_dir)
    scene_dir = output_path / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    scene_category_counts = Counter()
    scene_goal_counts = Counter()
    selected_episode_count = 0
    for scene_index, replay in enumerate(scene_replays):
        scene = str(replay["scene"])
        records = list(replay["records"])
        trajectories = list(replay["trajectories"])
        selected_episode_count += len(records)
        category_count = len(_scene_categories(records))
        goal_positions = _scene_goal_positions(trajectories)
        scene_category_counts[category_count] += 1
        scene_goal_counts[len(goal_positions)] += 1

        fig, _ = plot_habitat_topdown(replay)
        png_path = scene_dir / f"{scene_index:03d}_{_safe_stem(scene)}.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        figures.append(
            {
                "scene": scene,
                "png_path": str(png_path.relative_to(output_path)),
                "selected_episode_count": len(records),
                "object_categories": sorted(_scene_categories(records)),
                "unique_goal_position_count": len(goal_positions),
                "unique_goal_positions_grid": goal_positions,
            }
        )
        print(
            json.dumps(
                {
                    "event": "scene_png_written",
                    "scene": scene,
                    "png_path": str(png_path),
                    "selected_episode_count": len(records),
                    "unique_goal_position_count": len(goal_positions),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    manifest = {
        "analysis_name": "hm3d_scene_topdown_trajectories",
        "scope": "habitat_pr2l_hm3d_bc",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git_output("rev-parse", "HEAD"),
        "source_status_short": _git_output("status", "--short"),
        "source_tree_dirty": bool(_git_output("status", "--short")),
        "command": " ".join(shlex.quote(part) for part in command),
        "data_root": str(data_root),
        "exp": exp,
        "output_dir": str(output_path),
        "scene_png_dir": str(scene_dir.relative_to(output_path)),
        "scene_count": len(scene_replays),
        "selected_episode_count": selected_episode_count,
        "category_count_distribution": {
            str(key): scene_category_counts[key] for key in sorted(scene_category_counts)
        },
        "goal_count_distribution": {
            str(key): scene_goal_counts[key] for key in sorted(scene_goal_counts)
        },
        "figures": figures,
    }
    manifest_path = output_path / "result_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export one top-down trajectory PNG per selected HM3D scene."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--exp", default=DEFAULT_EXP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    scene_replays = replay_selected_objectnav_scene_topdowns(args.data_root, exp=args.exp)
    manifest = write_scene_topdown_pngs(
        scene_replays,
        args.output_dir,
        data_root=args.data_root,
        exp=args.exp,
        dpi=args.dpi,
        command=[sys.executable, *sys.argv] if argv is None else [sys.executable, *argv],
    )
    print(
        json.dumps(
            {
                "event": "export_complete",
                "manifest_path": str(Path(args.output_dir) / "result_manifest.json"),
                "scene_count": manifest["scene_count"],
                "selected_episode_count": manifest["selected_episode_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _scene_categories(records: list[dict[str, object]]) -> set[str]:
    return {
        str(record.get("object_category", record.get("goal_text", "unknown")))
        for record in records
    }


def _scene_goal_positions(trajectories: list[dict[str, object]]) -> list[list[int]]:
    goals = set()
    for trajectory in trajectories:
        if "goal_grid_position" not in trajectory:
            raise ValueError("Top-down replay is missing actual Habitat goal_grid_position.")
        goal = np.asarray(trajectory["goal_grid_position"], dtype=np.int64)
        goals.add((int(goal[0]), int(goal[1])))
    return [[row, col] for row, col in sorted(goals)]


def _safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


if __name__ == "__main__":
    main()
