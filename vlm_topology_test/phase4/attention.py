"""Where does the trained CLS query actually look?

This is the one thing about the policy that the structural checks cannot settle. Every
parameter matches the paper and every shape is right, but a network is free to ignore an input
it was given. If the learned query attends only to the sixteen visual tokens and skips the
question and the model's written answer, then this policy is the image-encoder baseline wearing
PR2L's clothes, and the gap the paper reports has nothing to reproduce.

The frame's tokens come in three stretches, in this order:

    [16 visual] [P prompt] [G generated]      G varies from 32 to 48, P is fixed per condition

`P` is recovered from the data rather than assumed: the shortest frame in a trajectory is the
one where the model generated the minimum 32 tokens, so P = (shortest frame) - 16 - 32.

What is reported is the share of the query's attention that lands on each stretch, next to the
share it would get if attention were spread evenly. A ratio near 1 means the query treats that
stretch like any other tokens; well above 1 means it seeks it out. The visual stretch is only a
fifth of the tokens, so even indifference gives it about a fifth of the mass -- the number to
read is the ratio, not the raw share.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import dataset
from dataset import load_trajectory, read_manifest
from policy import NavigationPolicy

HABITAT_ROOT = Path("/data/topovlm/habitat")
CHECKPOINT_ROOT = Path("/data/topovlm/checkpoints/pr2l_phase4")
VISUAL_TOKENS = 16       # Appendix C.2, general item 5: 256 pooled 4x4
MIN_GENERATED = 32       # Appendix C.2, PR2L item 4


def prompt_length(counts: np.ndarray) -> int:
    """Recover the question's token count from the frames' lengths."""
    return int(counts.min()) - VISUAL_TOKENS - MIN_GENERATED


def capture_attention(policy: NavigationPolicy, tokens: torch.Tensor,
                      padding: torch.Tensor) -> torch.Tensor:
    """Run the summary layer and return the query's attention over each frame's tokens."""
    layer = policy.summary.transformer.decoder.layers[0].multihead_attn
    original, captured = layer.forward, []

    def spy(query, key, value, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        result = original(query, key, value, **kwargs)
        captured.append(result[1].detach())
        return result

    layer.forward = spy
    try:
        with torch.inference_mode():
            policy.summary(tokens, padding)
    finally:
        layer.forward = original
    return captured[0][:, 0]        # [frames, tokens]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="stageA_cot")
    parser.add_argument("--condition", default="cot")
    parser.add_argument("--trajectories", type=int, default=12)
    args = parser.parse_args()

    root = HABITAT_ROOT / "embeddings" / f"pr2l_habitat_web_hd_{args.condition}"
    records = read_manifest(root)[:args.trajectories]

    checkpoint = torch.load(CHECKPOINT_ROOT / f"{args.run_name}.pt", map_location="cpu")
    policy = NavigationPolicy(heading_encoding=checkpoint.get("heading_encoding", "angle"),
                              num_heads=checkpoint.get("num_heads", 1))
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    totals = np.zeros(3)      # visual, prompt, generated
    spans = np.zeros(3)       # how many tokens each stretch holds, for the even-spread baseline
    frames_seen = 0

    for record in records:
        trajectory = load_trajectory(record, root)
        counts = (~trajectory.token_padding).sum(axis=1)
        prompt = prompt_length(counts)
        if prompt <= 0:
            print(f"[attn] {record['episode_id']}: 질문 길이 추정 실패 ({prompt}), 건너뜀")
            continue

        tokens = torch.from_numpy(trajectory.tokens.astype(np.float32)).unsqueeze(0)
        padding = torch.from_numpy(trajectory.token_padding).unsqueeze(0)
        weights = capture_attention(policy, tokens, padding).numpy()

        for row, count in zip(weights, counts):
            visual = row[:VISUAL_TOKENS].sum()
            question = row[VISUAL_TOKENS:VISUAL_TOKENS + prompt].sum()
            generated = row[VISUAL_TOKENS + prompt:count].sum()
            totals += (visual, question, generated)
            spans += (VISUAL_TOKENS, prompt, count - VISUAL_TOKENS - prompt)
            frames_seen += 1

    if not frames_seen:
        print("[attn] 볼 수 있는 프레임이 없다")
        return 1

    share = totals / totals.sum()
    even = spans / spans.sum()
    names = ("시각 16", "질문", "생성 답변")

    print(f"\n=== CLS 질의의 어텐션 분포 ({frames_seen:,} 프레임, 궤적 {len(records)}) ===")
    print(f"{'구간':10s} {'토큰 비중':>10s} {'어텐션 비중':>12s} {'배율':>8s}")
    for index, name in enumerate(names):
        ratio = share[index] / max(even[index], 1e-9)
        print(f"{name:10s} {100*even[index]:9.1f}% {100*share[index]:11.1f}% {ratio:7.2f}x")

    text = share[1] + share[2]
    print(f"\n텍스트(질문+생성) 어텐션 비중: {100*text:.1f}% "
          f"(토큰 비중 {100*(even[1]+even[2]):.1f}%)")
    if text < 0.05:
        print("[attn] 경고 — 질의가 텍스트를 거의 보지 않는다. 이 정책은 사실상 "
              "이미지 인코더 기준선이고, 논문의 격차를 재현할 수 없다.")
        return 1
    print("[attn] 질의가 텍스트 위치를 실제로 사용한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
