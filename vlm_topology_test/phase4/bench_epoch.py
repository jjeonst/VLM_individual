"""How long the arithmetic of one epoch takes on this card, with no disk in the way.

The loader is deliberately absent: this measures only what the GPU does, so that the read time
can be added separately for a node whose page cache is known. Batch shapes follow what
`group_by_length` actually produces at `--frame-budget 384` -- most groups are several short
trajectories padded together, and a handful are one long trajectory alone.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy import NavigationPolicy  # noqa: E402

FRAMES_PER_EPOCH = 1_219_318


def bench(policy, optimiser, device, batch, steps, tokens, repeats, amp):
    make = lambda *s: torch.randn(*s, device=device)
    args = (make(batch, steps, tokens, 2048),
            torch.zeros(batch, steps, tokens, dtype=torch.bool, device=device),
            make(batch, steps, 2), make(batch, steps),
            torch.zeros(batch, steps, 4, device=device),
            torch.zeros(batch, steps, 6, device=device))
    for index in range(repeats + 1):
        if index == 1:
            torch.cuda.synchronize(); started = time.time()
        with torch.autocast("cuda", dtype=amp or torch.bfloat16, enabled=amp is not None):
            logits, _ = policy(*args)
        logits.float().sum().backward()
        optimiser.step(); optimiser.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = (time.time() - started) / repeats
    return elapsed, batch * steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=78, help="78 for CoT, 16 for the image encoder")
    parser.add_argument("--amp", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--summary-chunk", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(0)} | amp {args.amp} | "
          f"토큰 {args.tokens} | 조각 {args.summary_chunk or '없음'}\n")

    policy = NavigationPolicy(token_dim=2048, summary_chunk=args.summary_chunk).to(device)
    policy.train()
    optimiser = torch.optim.AdamW(policy.parameters(), lr=1e-4, eps=1e-5)
    amp = torch.bfloat16 if args.amp == "bf16" else None

    shapes = [(6, 64, "짧은 궤적 6개"), (2, 192, "중간 궤적 2개"), (1, 384, "긴 궤적 1개")]
    rates = []
    for batch, steps, label in shapes:
        try:
            elapsed, frames = bench(policy, optimiser, device, batch, steps,
                                    args.tokens, args.repeats, amp)
            per = 1000 * elapsed / frames
            rates.append(per)
            print(f"  {label:16s} {batch}x{steps:4d} | {elapsed:6.3f}s | 프레임당 {per:5.3f} ms")
        except torch.cuda.OutOfMemoryError:
            print(f"  {label:16s} {batch}x{steps:4d} | OOM")
            policy.zero_grad(set_to_none=True); torch.cuda.empty_cache()

    if rates:
        typical = sum(rates) / len(rates)
        epoch = FRAMES_PER_EPOCH * typical / 1000
        print(f"\n  평균 {typical:.3f} ms/프레임")
        print(f"  epoch당 계산 {epoch:,.0f}초 ({epoch/60:.0f}분)")
        print(f"  31 epoch 계산만 {31*epoch/3600:.1f}시간")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
