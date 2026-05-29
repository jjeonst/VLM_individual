"""Build prompt-conditioned VLM and topology graph caches for Habitat episodes."""

from __future__ import annotations

import json
from pathlib import Path

from configs.schema import TopoVLMConfig
from data.habitat_manifest import load_episode_records, resolve_data_path
from encoders import build_vlm_encoder
from topology.graph_builder import build_sequential_similarity_graph


def build_habitat_graph_cache(cfg: TopoVLMConfig) -> dict[str, object]:
    if cfg.model.vlm.backend != "prismatic":
        raise ValueError(f"Unsupported VLM backend: {cfg.model.vlm.backend}")
    if cfg.model.topology.builder != "sequential_similarity":
        raise ValueError(f"Unsupported topology builder: {cfg.model.topology.builder}")

    import numpy as np
    from PIL import Image

    data_root = Path(cfg.data.data_root)
    episode_manifest = resolve_data_path(data_root, cfg.data.episodes_manifest)
    records = load_episode_records(episode_manifest)
    if cfg.data.max_episodes is not None:
        records = records[: cfg.data.max_episodes]
    if not records:
        raise ValueError(f"No Habitat episode records in {episode_manifest}")

    graph_dir = resolve_data_path(data_root, cfg.data.graph_cache_dir)
    embedding_dir = resolve_data_path(data_root, cfg.data.embeddings_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    encoder = build_vlm_encoder(cfg.model.vlm)
    manifest_path = resolve_data_path(data_root, cfg.data.graph_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    written = []

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for record in records:
            rgb_path = resolve_data_path(data_root, record.rgb_path)
            actions_path = resolve_data_path(data_root, record.actions_path)
            if not rgb_path.exists():
                raise FileNotFoundError(rgb_path)
            if not actions_path.exists():
                raise FileNotFoundError(actions_path)

            frames = np.load(rgb_path)
            actions = np.load(actions_path)
            if len(frames) == 0 or len(actions) == 0:
                raise ValueError(f"Empty episode payload: {record.episode_id}")

            embeddings = []
            for frame in frames[:: cfg.data.frame_stride]:
                image = Image.fromarray(frame.astype("uint8"), mode="RGB")
                embeddings.append(encoder.encode_image_goal(image, record.goal_text))
            embedding_array = np.stack(embeddings, axis=0)

            graph = build_sequential_similarity_graph(
                embedding_array,
                similarity_threshold=cfg.model.topology.similarity_threshold,
                max_nodes=cfg.model.topology.max_nodes,
                min_segment_len=cfg.model.topology.min_segment_len,
                normalize=cfg.model.topology.normalize_embeddings,
            )

            embedding_rel = Path(cfg.data.embeddings_dir) / f"{record.episode_id}.npy"
            graph_rel = Path(cfg.data.graph_cache_dir) / f"{record.episode_id}.npz"
            np.save(resolve_data_path(data_root, embedding_rel), embedding_array)
            np.savez_compressed(
                resolve_data_path(data_root, graph_rel),
                nodes=graph.nodes,
                edges=np.asarray(graph.edges, dtype=np.int64),
                frame_ranges=np.asarray(graph.frame_ranges, dtype=np.int64),
                actions=actions,
                target_action=np.asarray(int(actions[-1]), dtype=np.int64),
            )
            graph_record = {
                "episode_id": record.episode_id,
                "split": record.split,
                "scene_id": record.scene_id,
                "goal_text": record.goal_text,
                "graph_path": str(graph_rel),
                "embedding_path": str(embedding_rel),
                "target_action": int(actions[-1]),
                "num_nodes": int(graph.nodes.shape[0]),
            }
            manifest.write(json.dumps(graph_record, sort_keys=True) + "\n")
            written.append(graph_record)

    return {
        "status": "ok",
        "episodes": len(records),
        "manifest": str(manifest_path),
        "graphs_written": len(written),
    }
