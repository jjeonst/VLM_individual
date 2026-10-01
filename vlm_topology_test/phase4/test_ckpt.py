"""Check that `--summary-chunk` changes nothing but memory.

Three things have to hold before the CoT run moves to an 11 GB card:

  1. the chunked path returns exactly what the single-pass one does, in eval mode;
  2. the gradients it produces are the same, in training mode with dropout live -- this is the
     part checkpointing could get wrong, since the backward pass redraws the dropout mask and
     only matches if the RNG state was preserved;
  3. the longest demonstration in the dataset, 2,612 steps, fits.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy import NavigationPolicy  # noqa: E402

LONGEST = 2612          # the longest trajectory in the cot_fixed manifest
TOKENS = 78             # visual + prompt + generated, the CoT condition's width
CHUNK = 256


def inputs(device, steps, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    make = lambda *shape: torch.randn(*shape, generator=generator).to(device)
    return (make(1, steps, TOKENS, 2048),
            torch.zeros(1, steps, TOKENS, dtype=torch.bool, device=device),
            make(1, steps, 2), make(1, steps),
            torch.zeros(1, steps, 4, device=device),
            torch.zeros(1, steps, 6, device=device))


def build(device, chunk):
    torch.manual_seed(0)
    return NavigationPolicy(token_dim=2048, summary_chunk=chunk).to(device)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA 없음"); return 1
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"{name} | {total:.1f} GB\n")

    plain, chunked = build(device, None), build(device, CHUNK)
    chunked.load_state_dict(plain.state_dict())

    print("=== 1. 순전파 동치 (eval, 64스텝) ===")
    plain.eval(); chunked.eval()
    args = inputs(device, 64)
    with torch.no_grad():
        a, _ = plain(*args)
        b, _ = chunked(*args)
    forward_gap = (a - b).abs().max().item()
    print(f"  최대 절대 오차 {forward_gap:.3e}")

    print("\n=== 2. 기울기 동치 (train, dropout 살아 있음, 64스텝) ===")
    plain.train(); chunked.train()
    grads = []
    for model in (plain, chunked):
        torch.manual_seed(1234)          # same dropout draw for both paths
        model.zero_grad(set_to_none=True)
        logits, _ = model(*args)
        logits.float().pow(2).mean().backward()
        grads.append({n: p.grad.detach().clone() for n, p in model.named_parameters()
                      if p.grad is not None})
    shared = sorted(set(grads[0]) & set(grads[1]))
    grad_gap = max((grads[0][n] - grads[1][n]).abs().max().item() for n in shared)
    scale = max(grads[0][n].abs().max().item() for n in shared)
    print(f"  파라미터 {len(shared)}개 | 최대 절대 오차 {grad_gap:.3e} (기울기 크기 {scale:.3e})")

    print(f"\n=== 3. 최장 궤적 {LONGEST}스텝 ===")
    peaks = {}
    for label, model in (("원본", plain), (f"조각 {CHUNK}", chunked)):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model.zero_grad(set_to_none=True)
        try:
            long_args = inputs(device, LONGEST, seed=7)
            started = time.time()
            logits, _ = model(*long_args)
            logits.float().sum().backward()
            torch.cuda.synchronize()
            peaks[label] = torch.cuda.max_memory_allocated() / 1e9
            print(f"  {label:10s} 최대 {peaks[label]:5.2f} GB | {time.time()-started:5.2f}s")
        except torch.cuda.OutOfMemoryError:
            print(f"  {label:10s} OOM")
        finally:
            model.zero_grad(set_to_none=True)
            del long_args
            torch.cuda.empty_cache()

    print("\n=== 판정 ===")
    ok = True
    for label, value, limit in (("순전파 동치", forward_gap, 0.0),
                                ("기울기 동치", grad_gap, 1e-5 * max(scale, 1.0))):
        passed = value <= limit
        ok &= passed
        print(f"  {label}: {'통과' if passed else '실패'} ({value:.3e} ≤ {limit:.3e})")
    fits = peaks.get(f"조각 {CHUNK}", float("inf")) < total * 0.85
    ok &= fits
    print(f"  최장 궤적 적재: {'통과' if fits else '실패'}")
    print("\n" + ("전부 통과 — 이 카드에서 학습해도 된다." if ok else "실패 — 옮기지 말 것."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
