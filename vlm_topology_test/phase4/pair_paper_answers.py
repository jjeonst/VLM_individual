"""Put the paper's Table 4 answers beside ours, situation by situation, and our no-CoT answer beside
our CoT answer to the same image.

The paper prints six CoT answers and no no-CoT answers at all, so the no-CoT column can only be ours.
For each Table 4 answer the situation is (target, yes/no, the room type it names); our CoT answers
from the real bfloat16 encoding are filtered to that situation and one is drawn at random with a
fixed seed, so nothing is hand-picked. Both conditions stored the answers to the first three frames
of every trajectory, so the no-CoT answer shown is to the identical image.
"""
from __future__ import annotations

import glob
import json
import re

import numpy as np

from compare_answers import PAPER

ROOT = "/data/topovlm/habitat/embeddings/pr2l_habitat_web_hd_{}/manifests/*.jsonl"
SITUATIONS = [  # (target, answer, room named) for Table 4 rows 1-6
    ("toilet", "no", "bedroom"),
    ("toilet", "yes", "bathroom"),
    ("bed", "no", "dining room"),
    ("bed", "yes", "bedroom"),
    ("sofa", "no", "kitchen"),
    ("sofa", "yes", "living room"),
]


def load(condition: str) -> dict[str, dict]:
    out = {}
    for path in sorted(glob.glob(ROOT.format(condition))):
        for line in open(path):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[record["episode_id"]] = record
    return out


def main() -> int:
    cot, nocot = load("cot_fixed"), load("nocot")
    rng = np.random.default_rng(0)
    for number, ((target, verdict, room), paper) in enumerate(zip(SITUATIONS, PAPER), start=1):
        pool = []
        for episode, record in cot.items():
            if record.get("object_category") != target or episode not in nocot:
                continue
            for index, answer in enumerate(record.get("sample_answers", [])):
                if (re.match(rf"\s*{verdict}\b", answer, re.I)
                        and re.search(rf"\b{room}\b", answer, re.I)):
                    pool.append((episode, index, answer))
        total = sum(len(r.get("sample_answers", [])) for r in cot.values()
                    if r.get("object_category") == target)
        print(f"\n### {number}. {target} · {verdict.upper()} · {room}   "
              f"(우리 CoT에서 이 상황 {len(pool)}건 / {target} 답 {total}건)")
        print(f"PAPER  {paper}")
        if not pool:
            print("OURS   (해당 상황 없음)")
            continue
        episode, index, answer = pool[int(rng.integers(len(pool)))]
        others = nocot[episode].get("sample_answers", [])
        print(f"COT    {answer}")
        print(f"NOCOT  {others[index] if index < len(others) else '(없음)'}")
        print(f"       [{episode[:24]} 프레임 {index}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
