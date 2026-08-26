"""Find the 1024 directions worth keeping, and measure what keeping only those costs.

A token leaves the language model as 4096 numbers. The paper reduces that to 1024 before the
policy ever sees it (Appendix C.2, PR2L item 2), which cuts both the storage and the width the
policy has to learn over. Principal component analysis picks the directions to keep: it finds
the directions along which the tokens actually differ from one another, keeps the 1024 that
account for the most of that variation, and discards the rest as directions in which every
token looks nearly the same.

**The directions must be found once and then frozen.** If each frame were reduced using
directions computed from that frame, the same scene would land on different numbers depending
on when it was seen, and nothing downstream could learn from them. So the directions are
estimated from a sample -- the paper draws one frame per trajectory -- written to disk, and
applied unchanged to every frame afterwards, including the frames encoded live during
evaluation.

**One basis or two.** The paper says it computes "all resulting tokens' principle component
vectors" and uses "said vectors" to reduce "all tokens", which reads as a single basis shared
by both of the layers being extracted; the paragraph describing the reduction does not mention
layers at all, and the two-layer choice is only introduced in the item after it. So a single
shared basis is what this reproduction uses.

There is a reason to check that choice rather than assume it is harmless. The two layers are
not symmetric: HuggingFace's Llama returns the final layer's hidden state **after** the closing
RMSNorm, while the layer before it is the raw residual stream, so their scales can differ by a
lot. Principal components chase total variance, so a much larger cloud would capture the basis
and the smaller one would be poorly served -- which would quietly undo the point of taking two
layers. This module therefore fits the per-layer bases too, purely to report how much each
layer keeps under each choice. Fitting all three costs one pass over the same sample.

**How the fit is done without holding the sample in memory.** The sample is about 1.2 million
vectors of 4096 numbers, which is far too much to keep. Only a running count, sum, and sum of
outer products are needed to recover the mean and covariance, and those are 4096x4096 -- a
fixed 134 MB per accumulator regardless of how many vectors pass through.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DIM = 4096
KEEP = 1024


@dataclass
class Moments:
    """Running count, sum and sum of outer products for a stream of vectors."""

    count: int = 0
    total: np.ndarray = field(default_factory=lambda: np.zeros(DIM, dtype=np.float64))
    gram: np.ndarray = field(default_factory=lambda: np.zeros((DIM, DIM), dtype=np.float64))

    def update(self, vectors: np.ndarray) -> None:
        block = np.asarray(vectors, dtype=np.float64).reshape(-1, DIM)
        self.count += block.shape[0]
        self.total += block.sum(axis=0)
        self.gram += block.T @ block

    @property
    def mean(self) -> np.ndarray:
        return self.total / max(self.count, 1)

    @property
    def covariance(self) -> np.ndarray:
        mean = self.mean
        return self.gram / max(self.count, 1) - np.outer(mean, mean)


@dataclass
class Basis:
    """The frozen reduction: subtract `mean`, then multiply by `components`."""

    mean: np.ndarray            # [4096]
    components: np.ndarray      # [1024, 4096], rows orthonormal
    eigenvalues: np.ndarray     # [4096], descending

    @property
    def explained(self) -> float:
        return float(self.eigenvalues[:KEEP].sum() / max(self.eigenvalues.sum(), 1e-12))

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        flat = np.asarray(vectors, dtype=np.float32).reshape(-1, DIM)
        return (flat - self.mean.astype(np.float32)) @ self.components.astype(np.float32).T


def fit(moments: Moments) -> Basis:
    """Turn accumulated moments into the frozen reduction."""
    covariance = moments.covariance
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return Basis(mean=moments.mean,
                 components=np.ascontiguousarray(eigenvectors[:, :KEEP].T),
                 eigenvalues=np.maximum(eigenvalues, 0.0))


def retained_fraction(basis: Basis, moments: Moments) -> float:
    """How much of one group's variation survives reduction by a possibly foreign basis.

    Measured as one minus the reconstruction error: the group's vectors are centred by the
    basis's mean, projected onto its 1024 directions, brought back, and compared with where
    they started. Using the basis's mean rather than the group's own is deliberate -- that is
    what happens at encoding time, so a mean that suits one layer and not the other shows up
    here rather than staying hidden.
    """
    components = basis.components
    offset = moments.mean - basis.mean
    spread = moments.covariance + np.outer(offset, offset)

    projected = components @ spread @ components.T
    total = float(np.trace(spread))
    kept = float(np.trace(projected))
    own_variance = float(np.trace(moments.covariance))
    if own_variance <= 0:
        return 0.0
    # Residual is measured against the group's own variance, so an offset mean counts as loss.
    return float(1.0 - (total - kept) / own_variance)


def save(path: Path, basis: Basis, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=basis.mean.astype(np.float32),
             components=basis.components.astype(np.float32),
             eigenvalues=basis.eigenvalues.astype(np.float32),
             metadata=np.array(repr(metadata)))


def load(path: Path) -> Basis:
    payload = np.load(path, allow_pickle=False)
    return Basis(mean=payload["mean"].astype(np.float64),
                 components=payload["components"].astype(np.float64),
                 eigenvalues=payload["eigenvalues"].astype(np.float64))
