"""Compare the paper's example VLM answers with the ones our encoding actually produced.

Ours come from the encoding manifests' `sample_answers` (bfloat16, the real run, first three
frames of each trajectory), not from the float16 probes.
"""
import glob
import json
import re
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

PAPER = [
    "No, a toilet is not found in this room as it is a bedroom. Toilets are typically found in bathrooms. "
    "The image shows a large bed with a wooden headboard, nightstands, and lamps.",
    "Yes, a toilet would be found in this room because it is a bathroom. The presence of a toilet in the "
    "image is expected, as bathrooms are designed to accommodate toilets for personal hygiene.",
    "No, a bed would not be found in this room because it is a dining room. The presence of a dining table "
    "and chairs suggests that this room is designed for eating meals and socializing, rather than sleeping",
    "Yes, a bed would be found here because the room is described as a bedroom. It has a bed, a television, "
    "a lamp, and a dresser, all of which are common furnishings in a bedroom.",
    "No, a sofa is not found here because it is a kitchen area with appliances and cabinets. Sofas are "
    "typically found in living rooms or other common areas.",
    "Yes, there is a black leather sofa in the living room. It has a red pillow on it. It is a large "
    "sectional couch.",
    "No, as it's a bedroom. Toilets are usually found in bathrooms.",   # Figure 1, likely abridged
]
TARGETS = ("tv_monitor", "toilet", "bed", "sofa", "chair", "plant")

ROOM = re.compile(r"\b(bedroom|bathroom|kitchen|living room|dining room|hallway|corridor|office|"
                  r"laundry|garage|closet|entryway|foyer|lounge|study|basement|staircase|stairway)\b", re.I)
REASON = re.compile(r"\b(because|since|as it|as this|due to|typically|usually|commonly?|likely|suggests?)\b", re.I)
CAPTION = re.compile(r"\b(mentioned|described|description|list(ed)?|information|given)\b", re.I)
GARBAGE = re.compile(r"[={};<>\[\]#]|\(\s*\d|\d+\s*,\s*\d+|\.\w+\(|\bv\.\w")
NOT_MENTIONED = re.compile(r"not mentioned in the image", re.I)


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def stats(texts, tok, label):
    n = len(texts)
    ids = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    rows = []
    for t in texts:
        clean = t
        for target in TARGETS:
            clean = clean.replace(target, "x")
        parts = sentences(t)
        norm = [re.sub(r"\W+", " ", s.lower()).strip() for s in parts]
        long_ = [s for s in norm if len(s.split()) >= 3]
        rows.append({
            "yesno": bool(re.match(r"\s*(yes|no)\b", t, re.I)),
            "sentences": len(parts),
            "truncated": not t.rstrip().endswith((".", "!", "?", '"')),
            "room": bool(ROOM.search(t)),
            "reason": bool(REASON.search(t)),
            "caption": bool(CAPTION.search(t)),
            "not_mentioned": bool(NOT_MENTIONED.search(t)),
            "garbage": bool(GARBAGE.search(clean)),
            "repeat": len(long_) != len(set(long_)),
            "first_is_all": len(parts) <= 1,
        })
    m = lambda k: 100 * np.mean([r[k] for r in rows])
    print(f"\n[{label}] n={n}")
    print(f"  토큰 평균 {np.mean(ids):.1f} · 중앙 {np.median(ids):.0f} · 범위 {min(ids)}-{max(ids)}"
          f" · 문장 수 평균 {np.mean([r['sentences'] for r in rows]):.2f}")
    print(f"  예/아니오로 시작 {m('yesno'):5.1f}% | 방 종류 언급 {m('room'):5.1f}% | 이유 어휘 {m('reason'):5.1f}%")
    print(f"  '설명/목록/언급' 어휘 {m('caption'):5.1f}% | \"not mentioned in the image\" {m('not_mentioned'):5.1f}%")
    print(f"  문장 도중 끊김 {m('truncated'):5.1f}% | 쓰레기 {m('garbage'):5.1f}% | 문장 반복 {m('repeat'):5.1f}%")
    return rows


def manifest_answers(condition):
    out = []
    for f in sorted(glob.glob(f"/data/topovlm/habitat/embeddings/pr2l_habitat_web_hd_{condition}/manifests/*.jsonl")):
        for line in open(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            out += [(d.get("object_category"), a) for a in d.get("sample_answers", [])]
    return out


def main():
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    stats(PAPER, tok, "논문 예시 (Table 4 + Figure 1)")
    stats(PAPER[:6], tok, "논문 Table 4만")
    cot = manifest_answers("cot_fixed")
    nocot = manifest_answers("nocot")
    stats([a for _, a in cot], tok, "우리 CoT 인코딩 (bf16, 매니페스트)")
    stats([a for _, a in nocot], tok, "우리 no-CoT 인코딩 (bf16, 매니페스트)")

    print("\n=== 가장 흔한 첫 문장 ===")
    for label, data in (("CoT", cot), ("no-CoT", nocot)):
        first = Counter(re.sub(r"\b(" + "|".join(TARGETS) + r")\b", "<T>", sentences(a)[0] if sentences(a) else "")
                        for _, a in data)
        total = len(data)
        print(f"  [{label}]")
        for s, c in first.most_common(6):
            print(f"    {100 * c / total:5.1f}%  {s[:110]}")

    print("\n=== 목표 이름이 답에 그대로 나오는 비율 (tv_monitor) ===")
    for label, data in (("CoT", cot), ("no-CoT", nocot)):
        tv = [a for g, a in data if g == "tv_monitor"]
        lit = np.mean(["tv_monitor" in a for a in tv]) if tv else float("nan")
        print(f"  {label}: tv_monitor 에피소드 답 {len(tv)}개 중 'tv_monitor' 문자열 포함 {100 * lit:.1f}%")

    rng = np.random.default_rng(0)
    print("\n=== 무작위 표본 (CoT 6 / no-CoT 6) ===")
    for label, data in (("CoT", cot), ("no-CoT", nocot)):
        for i in rng.choice(len(data), 6, replace=False):
            g, a = data[i]
            print(f"  [{label}·{g}] {a[:240]}")


if __name__ == "__main__":
    main()
