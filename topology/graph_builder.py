"""Sequential similarity graph builder for cached VLM embeddings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopologyGraph:
    """Store graph node features, edges, and source frame ranges."""

    nodes: object
    edges: list[tuple[int, int]]
    frame_ranges: list[tuple[int, int]]


def build_sequential_similarity_graph(
    embeddings,
    *,
    similarity_threshold: float,
    max_nodes: int,
    min_segment_len: int,
    normalize: bool,
) -> TopologyGraph:
    import numpy as np

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [frames, dim]")
    if len(embeddings) == 0:
        raise ValueError("embeddings must be non-empty")
    feats = embeddings.astype("float32")
    if normalize:
        norm = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.maximum(norm, 1e-8)

    segments: list[list[int]] = [[0]]
    node_features = [feats[0].copy()]
    for frame_idx in range(1, len(feats)):
        current_feature = node_features[-1]
        similarity = float(np.dot(feats[frame_idx], current_feature))
        if similarity >= similarity_threshold or len(segments) >= max_nodes:
            segments[-1].append(frame_idx)
            node_features[-1] = feats[segments[-1]].mean(axis=0)
            if normalize:
                node_features[-1] = _normalize(node_features[-1])
            continue
        if len(segments[-1]) < min_segment_len and len(segments) > 1:
            segments[-1].append(frame_idx)
            node_features[-1] = feats[segments[-1]].mean(axis=0)
            if normalize:
                node_features[-1] = _normalize(node_features[-1])
            continue
        segments.append([frame_idx])
        node_features.append(feats[frame_idx].copy())

    edges = [(idx, idx + 1) for idx in range(max(0, len(segments) - 1))]
    frame_ranges = [(segment[0], segment[-1]) for segment in segments]
    return TopologyGraph(
        nodes=np.stack(node_features, axis=0),
        edges=edges,
        frame_ranges=frame_ranges,
    )


def _normalize(vector):
    import numpy as np

    return vector / max(float(np.linalg.norm(vector)), 1e-8)
