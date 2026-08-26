"""Train the policy to reproduce the actions people took.

This is behaviour cloning: at every step of every demonstration the policy is asked what to do,
and is corrected towards what the person actually did. The paper uses it rather than
reinforcement learning so that its numbers sit alongside the prior work it compares against
(Appendix C.2, opening paragraph).

**Whole trajectories, not slices.** The policy carries a recurrent state, and a slice taken from
the middle of a demonstration would have to start from a state nobody knows. Feeding each
demonstration from its first step means the state is rolled honestly throughout. The paper makes
the same choice and notes the consequence: batches then hold fewer distinct scenes than they
would otherwise, and gradient accumulation is needed to reach the intended batch size anyway
(Appendix C.2, item 3).

**The loss is weighted twice over**, because the demonstrations are dominated by walking
forward. See `dataset.inflection_weights`.

**The learning rate barely moves.** The paper keeps the schedule its predecessor used, which
decays linearly over 400 million transitions, while training for only about a tenth of that
(Appendix C.2, item 4). The rate therefore falls by roughly a tenth across the whole run rather
than annealing to zero. That is faithfully what the paper describes, and it is worth stating
plainly because a decaying schedule that never decays looks like a bug otherwise.

**Trajectories are read from disk as they are needed, never all at once.** Each frame carries
about 80 tokens of 2048 numbers, so a tenth of the dataset is roughly 40 GB and the whole of it
roughly 400 GB -- far past what fits in memory. Reading is also the slower half of the work: an
epoch's arithmetic takes seconds while an epoch's reading takes minutes, so the loading runs in
worker processes that fetch the next group while the current one is on the GPU.

Two modes:

``train`` is the real thing.

``overfit`` takes a handful of demonstrations and trains on them for a long time, asking only
whether the policy can memorise them. It requires no generalisation, so a failure there cannot
be blamed on having too little data -- it means something between the representation and the
optimiser is not connected. It is the cheapest check that the pipeline works at all.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import dataset
import policy as policy_module
from dataset import collate, load_trajectory, read_manifest
from policy import NavigationPolicy

HABITAT_ROOT = Path("/data/topovlm/habitat")
CHECKPOINT_ROOT = Path("/data/topovlm/checkpoints/pr2l_phase4")

# Appendix C.2, item 2, and VC-1 Appendix A.3 for the optimiser the paper adopts.
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
# Appendix C.2, item 3: the same number of transitions per update as the work it follows,
# derived there from 400M transitions over ~25k updates with 512 environments.
TRANSITIONS_PER_UPDATE = 16_384
# Appendix C.2, item 4: the schedule is written for the full run even though training stops
# early, so the rate decays at the same rate as the baseline rather than over our own horizon.
SCHEDULE_TRANSITIONS = 400_000_000
EPOCHS = 40
ACTION_NAMES = ("정지", "전진", "좌회전", "우회전")


def group_by_length(records: list[dict], frame_budget: int) -> list[list[dict]]:
    """Decide which trajectories share a forward pass, using only their recorded lengths.

    Grouping has to happen before anything is read, or the whole dataset would have to be in
    memory to plan the batches. The manifest already records each trajectory's step count, which
    is all the planner needs. Similar lengths go together so that padding a short demonstration
    out to a long one's length does not waste most of the batch.
    """
    ordered = sorted(records, key=lambda r: r["steps"])
    groups, current, longest = [], [], 0
    for record in ordered:
        length = record["steps"]
        if current and (len(current) + 1) * max(longest, length) > frame_budget:
            groups.append(current)
            current, longest = [], 0
        current.append(record)
        longest = max(longest, length)
    if current:
        groups.append(current)
    return groups


def stage_locally(records: list[dict], embedding_root: Path, target: Path,
                  workers: int = 16) -> tuple[Path, Path]:
    """Copy a run's trajectories to the node's own disk and read them from there.

    Forty epochs read the same files forty times. Whether that should come off the shared
    volume or a local one is not a matter of taste but of two numbers, and both were measured
    on the node this runs on, while the rest of the cluster was busy:

        /data     24.6 MB/s        -> 80 GB, forty times: 36 hours
        copy     590.0 MB/s
        /scratch 828.4 MB/s        -> one copy plus forty reads: 1.1 hours

    The gap is not the filesystems' doing so much as the traffic: four encoding jobs were
    reading and writing the shared volume at the time. It also cannot be papered over by the
    page cache, because 80 GB does not fit in the node's 62 GB of memory -- which is exactly
    why Stage A saw none of this, its 37 GB sitting comfortably in a 128 GB node.

    The two per-trajectory arrays that are not embeddings -- actions and pose -- are a few
    kilobytes each and get copied too. Small files are what a network filesystem handles worst,
    and there are two of them per trajectory per epoch.
    """
    from concurrent.futures import ThreadPoolExecutor

    target.mkdir(parents=True, exist_ok=True)
    local_embeddings = target / "embeddings"
    for sub in (local_embeddings / "train", target / "actions" / dataset.RENDER_SET / "train",
                target / "pose" / dataset.RENDER_SET / "train"):
        sub.mkdir(parents=True, exist_ok=True)

    jobs, total = [], 0
    for record in records:
        sources = dataset.trajectory_paths(record, embedding_root)
        targets = dataset.trajectory_paths(record, local_embeddings, target)
        for source, destination in zip(sources, targets):
            if destination.exists() and destination.stat().st_size == source.stat().st_size:
                continue
            jobs.append((source, destination))
            total += source.stat().st_size

    if not jobs:
        print(f"[stage] {target} 에 이미 전부 있다", flush=True)
        return local_embeddings, target

    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda pair: shutil.copyfile(pair[0], pair[1]), jobs))
    elapsed = time.time() - started
    print(f"[stage] {len(jobs):,}개 파일 {total/1e9:.1f} GB → {target} "
          f"({elapsed:.0f}초, {total/1e6/max(elapsed, 1e-9):.0f} MB/s)", flush=True)
    return local_embeddings, target


class GroupDataset(Dataset):
    """One item is one forward pass's worth of trajectories, read when asked for."""

    def __init__(self, groups: list[list[dict]], embedding_root: Path,
                 habitat_root: Path | None = None, zero_tokens: bool = False) -> None:
        self.groups = groups
        self.embedding_root = embedding_root
        self.habitat_root = habitat_root
        self.zero_tokens = zero_tokens

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return collate([load_trajectory(record, self.embedding_root, self.habitat_root,
                                        zero_tokens=self.zero_tokens)
                        for record in self.groups[index]])


def weighted_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor]):
    """Cross entropy over real steps only, scaled by each step's weight.

    Padded steps are *selected out* rather than multiplied by zero. The difference matters: a
    non-finite value at a padded position survives multiplication -- NaN times zero is NaN --
    and one such value turns the whole sum, and then every gradient, into NaN. Selection cannot
    let anything through. `policy.FrameSummary` also avoids producing the NaN in the first
    place; this is the second line of defence, and cheap.
    """
    flat = logits.reshape(-1, logits.shape[-1])
    target = batch["actions"].reshape(-1)
    valid = batch["valid"].reshape(-1)
    weights = batch["weights"].reshape(-1)

    losses = nn.functional.cross_entropy(flat, target, reduction="none") * weights
    losses = torch.where(valid, losses, torch.zeros_like(losses))
    return losses.sum(), valid.sum()


def accuracy_by_action(logits: torch.Tensor, batch: dict[str, torch.Tensor],
                       tally: np.ndarray) -> None:
    """Count correct predictions per action, which is what reveals a collapsed policy."""
    predicted = logits.argmax(dim=-1).reshape(-1)
    target = batch["actions"].reshape(-1)
    valid = batch["valid"].reshape(-1)
    for action in range(dataset.NUM_ACTIONS):
        chosen = (target == action) & valid
        tally[action, 0] += int((predicted[chosen] == action).sum())
        tally[action, 1] += int(chosen.sum())


def run_epoch(policy, optimiser, loader, device, seen: int, rate_at, update_every: int,
              zero_tokens: bool = False):
    """One pass over the data; returns the running totals the caller reports.

    `zero_tokens` blanks the frame representations and leaves everything else -- the pose, the
    heading, the previous action, the goal, the recurrent state -- untouched. It answers a
    question nothing else here can: how much of the policy's behaviour comes from the
    representation the paper is about, and how much from knowing where it stands and what it is
    looking for. A control that scores as well as the real thing means the representation is
    contributing nothing, whatever the structural checks say.
    """
    policy.train()
    tally = np.zeros((dataset.NUM_ACTIONS, 2), dtype=np.int64)
    loss_sum, step_sum, pending, updates = 0.0, 0, 0, 0
    wait, compute = 0.0, 0.0
    optimiser.zero_grad(set_to_none=True)

    mark = time.time()
    for batch in loader:
        wait += time.time() - mark
        mark = time.time()
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        # The loader hands over the half-precision block it read from disk; widening it is a
        # GPU-side cast of a tensor that is already here, rather than a CPU-side one that
        # doubles what has to be carried across. The values are identical either way.
        batch["tokens"] = batch["tokens"].float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = policy(batch["tokens"], batch["token_padding"], batch["gps"],
                               batch["compass"], batch["previous_action"], batch["goal"])
        logits = logits.float()
        total, count = weighted_loss(logits, batch)
        (total / max(int(count), 1)).backward()

        with torch.no_grad():
            accuracy_by_action(logits, batch, tally)
        loss_sum += float(total)
        step_sum += int(count)
        pending += int(count)
        seen += int(count)

        if pending >= update_every:
            for group in optimiser.param_groups:
                group["lr"] = rate_at(seen)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            pending, updates = 0, updates + 1
        compute += time.time() - mark
        mark = time.time()

    if pending:
        for group in optimiser.param_groups:
            group["lr"] = rate_at(seen)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        updates += 1

    return {"loss": loss_sum / max(step_sum, 1), "tally": tally, "updates": updates,
            "seen": seen, "wait": wait, "compute": compute}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "overfit"])
    parser.add_argument("--condition", default="cot")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--frame-budget", type=int, default=384,
                        help="frames per forward pass; lower it if memory runs short")
    parser.add_argument("--workers", type=int, default=8,
                        help="processes reading trajectories ahead of the GPU")
    parser.add_argument("--heading-encoding", default="angle", choices=["angle", "sincos"])
    parser.add_argument("--num-heads", type=int, default=policy_module.NUM_HEADS,
                        help="attention heads in the frame-summary layer; Listing 1 uses 1")
    parser.add_argument("--stage-to", type=Path, default=None,
                        help="copy this run's trajectories to a local disk first and read "
                             "from there; see stage_locally for the measurements")
    parser.add_argument("--prefetch", type=int, default=2,
                        help="batches each worker runs ahead; raise it when the GPU starves")
    parser.add_argument("--resume", type=Path, default=None,
                        help="checkpoint to continue from, for a run the wall clock cut short")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="also keep a copy every N epochs, for tracking how the "
                             "summary query's attention settles; 0 disables")
    parser.add_argument("--zero-tokens", action="store_true",
                        help="blank the frame representations: the control that measures "
                             "how much the VLM actually contributes")
    parser.add_argument("--trajectories", type=int, default=None,
                        help="use only this many trajectories (overfit mode uses 20)")
    args = parser.parse_args()

    overfit = args.mode == "overfit"
    epochs = args.epochs if args.epochs is not None else (200 if overfit else EPOCHS)
    wanted = args.trajectories if args.trajectories is not None else (20 if overfit else None)

    embedding_root = HABITAT_ROOT / "embeddings" / f"pr2l_habitat_web_hd_{args.condition}"
    records = read_manifest(embedding_root)
    if not records:
        print(f"[train] {embedding_root} 에 인코딩된 궤적이 없다")
        return 1
    if wanted is not None:
        records = records[:wanted]

    # The stored width says which condition this is: 2048 for PR2L's two stacked layers, 2176
    # for the image encoder's fused patches. Listing 1 takes it as an argument for this reason.
    widths = {r["width"] for r in records}
    if len(widths) != 1:
        print(f"[train] 매니페스트에 폭이 섞여 있다: {sorted(widths)}")
        return 1
    token_dim = widths.pop()

    habitat_root = None
    if args.stage_to is not None:
        embedding_root, habitat_root = stage_locally(records, embedding_root, args.stage_to,
                                                     workers=args.workers * 2)

    total_steps = sum(r["steps"] for r in records)
    groups = group_by_length(records, args.frame_budget)
    # Memorising a handful of demonstrations is meant to be easy, so the update is applied every
    # pass rather than after the paper's batch size, which those few steps would never reach.
    update_every = 1 if overfit else TRANSITIONS_PER_UPDATE

    print(f"[train] 모드 {args.mode} | 궤적 {len(records)} | 스텝 {total_steps:,} | "
          f"forward {len(groups)}회/epoch | epoch {epochs} | "
          f"헤드 {args.num_heads} | 나침반 {args.heading_encoding} | 폭 {token_dim} | "
          f"워커 {args.workers} prefetch {args.prefetch}"
          + (" | 토큰 0 대조군" if args.zero_tokens else ""), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = NavigationPolicy(heading_encoding=args.heading_encoding,
                              num_heads=args.num_heads,
                              token_dim=token_dim).to(device)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)

    def rate_at(seen: int) -> float:
        return LEARNING_RATE * max(0.0, 1.0 - seen / SCHEDULE_TRANSITIONS)

    # One file descriptor per tensor is how a worker hands a batch to the main process, and a
    # batch here carries nine of them. At 24 workers running 8 batches ahead that is 1,728
    # against a soft limit of 1,024, and the loader does not fail -- it deadlocks, with the disk
    # idle, the GPU idle and the main process parked in a futex. Raising the soft limit toward
    # the hard one costs nothing and removes the ceiling.
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = min(65536, hard)
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            print(f"[train] 파일 디스크립터 한도 {soft} → {want} (하드 {hard})", flush=True)
    except (ImportError, ValueError, OSError) as error:
        print(f"[train] 파일 디스크립터 한도를 올리지 못했다: {error}", flush=True)

    loader = DataLoader(GroupDataset(groups, embedding_root, habitat_root,
                                     zero_tokens=args.zero_tokens),
                        batch_size=None,
                        shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0,
                        prefetch_factor=args.prefetch if args.workers > 0 else None)

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    history, seen, first_epoch = [], 0, 1

    if args.resume:
        # Forty epochs of this condition is longer than any single allocation on this cluster
        # lasts, so the run has to survive being cut off. Restarting from zero each time is not
        # an option: an epoch costs about an hour.
        resumed = torch.load(args.resume, map_location=device)
        policy.load_state_dict(resumed["policy"])
        first_epoch = int(resumed["epoch"]) + 1
        # The schedule is a function of transitions seen, so it can be reconstructed exactly
        # rather than restarted -- otherwise the learning rate would jump back up on resume.
        seen = int(resumed.get("seen", (first_epoch - 1) * total_steps))
        history = resumed.get("history", [])
        if "optimiser" in resumed:
            optimiser.load_state_dict(resumed["optimiser"])
            note = "옵티마이저 상태까지 복원"
        else:
            # Checkpoints written before this option existed hold weights only. AdamW's moments
            # rebuild within a few dozen updates, so the cost is a brief transient, not a
            # restart -- but it is a deviation and belongs in the log.
            note = "옵티마이저 상태 없음 — AdamW 모멘트는 처음부터 다시 쌓인다"
        print(f"[train] {args.resume} 에서 이어받는다: epoch {first_epoch}부터, "
              f"전이 {seen:,} 기준 lr {rate_at(seen):.2e} | {note}", flush=True)
        if first_epoch > epochs:
            print(f"[train] 이미 {epochs} epoch을 마쳤다.")
            return 0

    for epoch in range(first_epoch, epochs + 1):
        started = time.time()
        stats = run_epoch(policy, optimiser, loader, device, seen, rate_at, update_every,
                          zero_tokens=args.zero_tokens)
        seen = stats["seen"]

        share = stats["tally"][:, 0] / np.maximum(stats["tally"][:, 1], 1)
        overall = stats["tally"][:, 0].sum() / max(stats["tally"][:, 1].sum(), 1)
        print(f"[train] epoch {epoch:3d} | loss {stats['loss']:.4f} | 정확도 {100*overall:5.1f}% | "
              f"lr {rate_at(seen):.2e} | 갱신 {stats['updates']:4d} | "
              f"{time.time()-started:5.0f}s (읽기 {stats['wait']:.0f}s 계산 {stats['compute']:.0f}s) | "
              + " ".join(f"{n} {100*s:4.1f}%" for n, s in zip(ACTION_NAMES, share)), flush=True)
        history.append({"epoch": epoch, "loss": stats["loss"], "overall_accuracy": float(overall),
                        "accuracy": {n: float(s) for n, s in zip(ACTION_NAMES, share)},
                        "transitions": seen, "read_seconds": stats["wait"],
                        "compute_seconds": stats["compute"]})

        payload = {"policy": policy.state_dict(), "epoch": epoch,
                   # Saved so that a run cut off by the wall clock resumes where it stopped,
                   # with the same optimiser moments and the same point on the schedule.
                   "optimiser": optimiser.state_dict(), "seen": seen, "history": history,
                   "heading_encoding": args.heading_encoding, "num_heads": args.num_heads,
                   "zero_tokens": args.zero_tokens, "condition": args.condition,
                   "token_dim": token_dim}
        torch.save(payload, CHECKPOINT_ROOT / f"{args.run_name}.pt")
        # Keep a few checkpoints along the way. Where the summary layer's query chooses to look
        # is the one thing about this policy that the parameter counts cannot check, and it can
        # only be read off a trained network -- so it has to be read at several points in a run
        # to tell a preference that is settling from one that is still moving. Each file is
        # 300 MB and there are a handful of them.
        if args.checkpoint_every and epoch % args.checkpoint_every == 0 and epoch != epochs:
            torch.save(payload, CHECKPOINT_ROOT / f"{args.run_name}_epoch{epoch:03d}.pt")

    (CHECKPOINT_ROOT / f"{args.run_name}_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n")

    if overfit:
        final = history[-1]["overall_accuracy"]
        weakest = min(history[-1]["accuracy"].items(), key=lambda kv: kv[1])
        print(f"\n[overfit] 최종 정확도 {100*final:.1f}% | 가장 낮은 행동: "
              f"{weakest[0]} {100*weakest[1]:.1f}%")
        if final < 0.95:
            print("[overfit] FAILED — 궤적 20개조차 외우지 못한다. 데이터 부족으로는 설명되지 "
                  "않으므로 표현·정책·손실·최적화 사이에 문제가 있다.")
            return 1
        print("[overfit] 통과 — 표현부터 최적화까지 연결에 문제가 없다.")
    print(f"[train] {CHECKPOINT_ROOT / f'{args.run_name}.pt'} 저장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
