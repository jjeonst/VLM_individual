"""Appendix C.9: the same 200 frames asked of two Prismatic checkpoints, answer formats side by side.

Both JSON files come from `probe_answer_length.py` -- same frames, same per-frame seeds, same
float16 two-card harness -- so a difference between them is the checkpoint:

  prism-dinosiglip-224px+7b             LLaVA v1.5 + LVIS-Instruct-4V + LRV-Instruct, 2 epochs (ours)
  prism-dinosiglip-224px-controlled+7b  LLaVA v1.5 only, 1 epoch

The question is whether the refusal template that fills our no-CoT answers ("No, it's not mentioned
in the image." ... "the instruction is misleading") belongs to the checkpoint. Metrics match
`compare_answers.py`.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

TARGETS = ("tv_monitor", "toilet", "bed", "sofa", "chair", "plant")
ROOM = re.compile(r"\b(bedroom|bathroom|kitchen|living room|dining room|hallway|corridor|office|"
                  r"laundry|garage|closet|entryway|foyer|lounge|study|basement|staircase|stairway)\b", re.I)
REASON = re.compile(r"\b(because|since|as it|as this|due to|typically|usually|commonly?|likely|suggests?)\b", re.I)
CAPTION = re.compile(r"\b(mentioned|described|description|list(ed)?|information|given)\b", re.I)
MISLEADING = re.compile(r"\b(misleading|instruction)\b", re.I)
NOT_MENTIONED = re.compile(r"not mentioned in the image", re.I)
GARBAGE = re.compile(r"[={};<>\[\]#]|\(\s*\d|\d+\s*,\s*\d+|\.\w+\(|\bv\.\w")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def masked(text: str) -> str:
    for target in TARGETS:
        text = text.replace(target, "x")
    return text


def row(text: str) -> dict:
    norm = [re.sub(r"\W+", " ", s.lower()).strip() for s in sentences(text)]
    long_ = [s for s in norm if len(s.split()) >= 3]
    return {"yesno": bool(re.match(r"\s*(yes|no)\b", text, re.I)), "room": bool(ROOM.search(text)),
            "reason": bool(REASON.search(text)), "caption": bool(CAPTION.search(text)),
            "misleading": bool(MISLEADING.search(text)), "not_mentioned": bool(NOT_MENTIONED.search(text)),
            "garbage": bool(GARBAGE.search(masked(text))), "repeat": len(long_) != len(set(long_)),
            "sentences": len(norm)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=Path, default=Path("/data/topovlm/logs/probe_answer_length.json"))
    parser.add_argument("--controlled", type=Path,
                        default=Path("/data/topovlm/logs/probe_answer_length_controlled.json"))
    args = parser.parse_args()
    runs = {"우리(prism)": json.loads(args.ours.read_text()),
            "controlled": json.loads(args.controlled.read_text())}

    head = (f"  {'모델':12s} {'설정':12s} {'토큰':>5s} {'예/아니오':>8s} {'방종류':>6s} {'이유':>6s} "
            f"{'설명어휘':>8s} {'notMent':>8s} {'misleading':>10s} {'쓰레기':>6s} {'반복':>5s} {'문장':>5s}")
    print(head)
    for key in ("nocot_min32", "cot_min32", "nocot_min0", "cot_min0"):
        for model, data in runs.items():
            rows = [row(x["text"]) for x in data[key]]
            tokens = np.mean([x["tokens"] for x in data[key]])
            m = lambda k: 100 * np.mean([r[k] for r in rows])
            print(f"  {model:12s} {key:12s} {tokens:5.1f} {m('yesno'):7.1f}% {m('room'):5.1f}% "
                  f"{m('reason'):5.1f}% {m('caption'):7.1f}% {m('not_mentioned'):7.1f}% "
                  f"{m('misleading'):9.1f}% {m('garbage'):5.1f}% {m('repeat'):4.1f}% "
                  f"{np.mean([r['sentences'] for r in rows]):5.2f}")
        print()

    print("=== 가장 흔한 첫 문장 (최소 32) ===")
    pattern = re.compile(r"\b(" + "|".join(TARGETS) + r")\b")
    for key in ("nocot_min32", "cot_min32"):
        for model, data in runs.items():
            firsts = Counter(pattern.sub("<T>", (sentences(x["text"]) or [""])[0]) for x in data[key])
            top = ";  ".join(f"{100 * c / len(data[key]):.0f}% {s[:60]}" for s, c in firsts.most_common(3))
            print(f"  {model:12s} {key:12s} {top}")

    print("\n=== 같은 프레임, 두 체크포인트 (최소 32) ===")
    total = len(runs["controlled"]["nocot_min32"])
    for j in list(range(0, total, max(total // 6, 1)))[:6]:
        a = runs["우리(prism)"]["nocot_min32"][j]
        print(f"\n[{a['goal']}] {a['episode_id'][:20]} 프레임 {a['frame']}")
        for key in ("nocot_min32", "cot_min32"):
            for model, data in runs.items():
                print(f"  {model:12s} {key:12s} {data[key][j]['text'][:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
