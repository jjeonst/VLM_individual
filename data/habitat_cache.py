"""Build prompt-conditioned VLM and topology graph caches for Habitat episodes."""

from __future__ import annotations

import json
from pathlib import Path

from configs.schema import TopoVLMConfig
from data.habitat_manifest import (
    load_episode_records,
    resolve_data_path,
    resolve_materialization_data_root,
)
from encoders import build_vlm_encoder
from topology.graph_builder import build_sequential_similarity_graph


def build_habitat_graph_cache(cfg: TopoVLMConfig) -> dict[str, object]:
    if cfg.data.cache_format == "pr2l_token_trajectory":
        return _build_pr2l_token_trajectory_cache(cfg)
    if cfg.data.cache_format != "single_action_graph":
        raise ValueError(f"Unsupported cache format: {cfg.data.cache_format}")
    if cfg.model.vlm.backend != "prismatic":
        raise ValueError(f"Unsupported VLM backend: {cfg.model.vlm.backend}")
    if cfg.model.topology.builder != "sequential_similarity":
        raise ValueError(f"Unsupported topology builder: {cfg.model.topology.builder}")
    _require_prismatic_auth_ready(cfg)

    import numpy as np
    from PIL import Image

    data_root = Path(cfg.data.data_root)
    output_data_root = resolve_materialization_data_root(cfg.data.data_root)
    episode_manifest = resolve_data_path(data_root, cfg.data.episodes_manifest)
    records = load_episode_records(episode_manifest)
    if cfg.data.max_episodes is not None:
        records = records[: cfg.data.max_episodes]
    if not records:
        raise ValueError(f"No Habitat episode records in {episode_manifest}")

    graph_dir = resolve_data_path(output_data_root, cfg.data.graph_cache_dir)
    embedding_dir = resolve_data_path(output_data_root, cfg.data.embeddings_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    encoder = build_vlm_encoder(cfg.model.vlm)
    manifest_path = resolve_data_path(output_data_root, cfg.data.graph_manifest)
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
            np.save(resolve_data_path(output_data_root, embedding_rel), embedding_array)
            np.savez_compressed(
                resolve_data_path(output_data_root, graph_rel),
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
        "source_data_root": str(data_root),
        "output_data_root": str(output_data_root),
        "graphs_written": len(written),
    }


def _build_pr2l_token_trajectory_cache(cfg: TopoVLMConfig) -> dict[str, object]:
    if cfg.model.vlm.backend != "prismatic":
        raise ValueError(f"Unsupported VLM backend: {cfg.model.vlm.backend}")
    if cfg.model.vlm.representation != "pr2l_visual_tokens_last_two_layers":
        raise ValueError(f"Unsupported PR2L VLM representation: {cfg.model.vlm.representation}")
    if cfg.model.topology.builder != "sequential_similarity":
        raise ValueError(f"Unsupported topology builder: {cfg.model.topology.builder}")
    _require_prismatic_auth_ready(cfg)

    import numpy as np
    from PIL import Image

    data_root = Path(cfg.data.data_root)
    output_data_root = resolve_materialization_data_root(cfg.data.data_root)
    episode_manifest = resolve_data_path(data_root, cfg.data.episodes_manifest)
    records = load_episode_records(episode_manifest)
    if cfg.data.max_episodes is not None:
        records = records[: cfg.data.max_episodes]
    if not records:
        raise ValueError(f"No Habitat episode records in {episode_manifest}")

    graph_dir = resolve_data_path(output_data_root, cfg.data.graph_cache_dir)
    embedding_dir = resolve_data_path(output_data_root, cfg.data.embeddings_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    encoder = build_vlm_encoder(cfg.model.vlm)
    projection = _load_or_fit_projection(cfg, encoder, records, data_root, output_data_root)
    manifest_path = resolve_data_path(output_data_root, cfg.data.graph_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    written = []

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for record in records:
            frame_tokens, actions, metadata = _encode_record_pr2l_tokens(
                cfg, encoder, record, data_root, projection, np, Image
            )
            graph = build_sequential_similarity_graph(
                frame_tokens,
                similarity_threshold=cfg.model.topology.similarity_threshold,
                max_nodes=cfg.model.topology.max_nodes,
                min_segment_len=cfg.model.topology.min_segment_len,
                normalize=cfg.model.topology.normalize_embeddings,
            )
            frame_ranges = np.asarray(graph.frame_ranges, dtype=np.int64)
            node_actions = np.asarray(
                [int(actions[min(int(end), len(actions) - 1)]) for _, end in graph.frame_ranges],
                dtype=np.int64,
            )
            embedding_rel = Path(cfg.data.embeddings_dir) / f"{record.episode_id}.npy"
            graph_rel = Path(cfg.data.graph_cache_dir) / f"{record.episode_id}.npz"
            metadata_rel = Path(cfg.data.graph_cache_dir) / f"{record.episode_id}.metadata.json"
            np.save(resolve_data_path(output_data_root, embedding_rel), frame_tokens)
            np.savez_compressed(
                resolve_data_path(output_data_root, graph_rel),
                nodes=graph.nodes.astype("float32"),
                edges=np.asarray(graph.edges, dtype=np.int64),
                frame_ranges=frame_ranges,
                actions=actions,
                node_actions=node_actions,
                action_mask=np.ones(len(node_actions), dtype=bool),
                target_action=np.asarray(int(actions[-1]), dtype=np.int64),
            )
            metadata.update(
                {
                    "episode_id": record.episode_id,
                    "source_dataset": record.source_dataset,
                    "source_trajectory_id": record.source_trajectory_id,
                    "object_category": record.object_category,
                    "representation_id": _representation_id(cfg),
                    "projection_path": cfg.model.vlm.projection_path,
                }
            )
            resolve_data_path(output_data_root, metadata_rel).write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
                "prediction_target": "nodes",
                "num_tokens": int(graph.nodes.shape[1]),
                "feature_dim": int(graph.nodes.shape[-1]),
                "num_actions": int(len(node_actions)),
                "representation_id": _representation_id(cfg),
                "metadata_path": str(metadata_rel),
            }
            manifest.write(json.dumps(graph_record, sort_keys=True) + "\n")
            written.append(graph_record)

    return {
        "status": "ok",
        "episodes": len(records),
        "manifest": str(manifest_path),
        "source_data_root": str(data_root),
        "output_data_root": str(output_data_root),
        "graphs_written": len(written),
        "cache_format": cfg.data.cache_format,
        "representation_id": _representation_id(cfg),
    }


def _require_prismatic_auth_ready(cfg: TopoVLMConfig) -> None:
    from encoders.prismatic import inspect_prismatic_hf_auth

    audit = inspect_prismatic_hf_auth(cfg.model.vlm)
    if audit["requires_private_hf_auth"] and not audit["token_available"]:
        hf_repo = audit["hf_repo"]
        model_config_path = audit["model_config_path"]
        raise FileNotFoundError(
            f"Missing Hugging Face token for gated Prismatic LLM metadata: "
            f"hf_repo={hf_repo} model_config_path={model_config_path}"
        )


def _encode_record_pr2l_tokens(cfg, encoder, record, data_root, projection, np, Image):
    rgb_path = resolve_data_path(data_root, record.rgb_path)
    actions_path = resolve_data_path(data_root, record.actions_path)
    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)
    if not actions_path.exists():
        raise FileNotFoundError(actions_path)
    frames = np.load(rgb_path)
    actions = np.load(actions_path).astype("int64")
    if len(frames) == 0 or len(actions) == 0:
        raise ValueError(f"Empty episode payload: {record.episode_id}")
    frames = frames[:: cfg.data.frame_stride]
    actions = actions[:: cfg.data.frame_stride]
    frame_tokens = []
    generated_texts = []
    for frame in frames:
        image = Image.fromarray(frame.astype("uint8"), mode="RGB")
        encoded = encoder.encode_image_goal_tokens(image, record.goal_text)
        frame_tokens.append(encoded["tokens"])
        generated_texts.append(encoded["generated_text"])
    token_array = np.stack(frame_tokens, axis=0).astype("float32")
    token_array = _apply_projection(token_array, projection, np).astype("float32")
    metadata = {
        "prompt_template": cfg.model.vlm.prompt_template,
        "include_generated_text": cfg.model.vlm.include_generated_text,
        "generated_texts": generated_texts,
        "hidden_layer_indices": list(cfg.model.vlm.hidden_layer_indices),
        "visual_pool_grid": cfg.model.vlm.visual_pool_grid,
        "visual_bank_reduction": cfg.model.vlm.visual_bank_reduction,
    }
    return token_array, actions, metadata


def _load_or_fit_projection(cfg, encoder, records, data_root, output_data_root):
    if cfg.model.vlm.projection == "none":
        return None
    if cfg.model.vlm.projection != "pca":
        raise ValueError(f"Unsupported projection: {cfg.model.vlm.projection}")
    if cfg.model.vlm.projection_path is None:
        raise ValueError("PCA projection requires model.vlm.projection_path")
    import numpy as np

    projection_path = resolve_data_path(data_root, cfg.model.vlm.projection_path)
    if projection_path.exists():
        return _load_projection(projection_path, np)
    output_projection_path = resolve_data_path(output_data_root, cfg.model.vlm.projection_path)
    return _fit_projection(cfg, encoder, records, data_root, output_projection_path, np)


def _fit_projection(cfg, encoder, records, data_root, projection_path, np):
    from PIL import Image
    from sklearn.decomposition import PCA

    samples = []
    sample_count = 0
    for record in records:
        rgb_path = resolve_data_path(data_root, record.rgb_path)
        if not rgb_path.exists():
            raise FileNotFoundError(rgb_path)
        frames = np.load(rgb_path)[:: cfg.data.frame_stride]
        for frame in frames:
            image = Image.fromarray(frame.astype("uint8"), mode="RGB")
            encoded = encoder.encode_image_goal_tokens(image, record.goal_text)
            tokens = encoded["tokens"].reshape(-1, encoded["tokens"].shape[-1])
            samples.append(tokens.astype("float32"))
            sample_count += int(tokens.shape[0])
            if sample_count >= cfg.model.vlm.projection_fit_max_tokens:
                break
        if sample_count >= cfg.model.vlm.projection_fit_max_tokens:
            break
    if not samples:
        raise ValueError("No PR2L token samples available for PCA projection fit.")
    sample_matrix = np.concatenate(samples, axis=0)
    max_components = min(sample_matrix.shape[0], sample_matrix.shape[1])
    if cfg.model.vlm.projection_dim is None:
        raise ValueError("PCA projection requires model.vlm.projection_dim")
    if cfg.model.vlm.projection_dim > max_components:
        raise ValueError(
            f"projection_dim {cfg.model.vlm.projection_dim} exceeds max PCA rank {max_components}"
        )
    pca = PCA(n_components=cfg.model.vlm.projection_dim, svd_solver="randomized", random_state=0)
    pca.fit(sample_matrix)
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        projection_path,
        mean=pca.mean_.astype("float32"),
        components=pca.components_.astype("float32"),
        explained_variance=pca.explained_variance_.astype("float32"),
        sample_count=np.asarray(sample_matrix.shape[0], dtype=np.int64),
        raw_dim=np.asarray(sample_matrix.shape[1], dtype=np.int64),
        projection_dim=np.asarray(cfg.model.vlm.projection_dim, dtype=np.int64),
        representation_id=np.asarray(_representation_id(cfg)),
    )
    return _load_projection(projection_path, np)


def _load_projection(path, np):
    payload = np.load(path)
    return {
        "mean": payload["mean"].astype("float32"),
        "components": payload["components"].astype("float32"),
    }


def _apply_projection(tokens, projection, np):
    if projection is None:
        return tokens
    original_shape = tokens.shape
    flat = tokens.reshape(-1, original_shape[-1]).astype("float32")
    projected = (flat - projection["mean"]) @ projection["components"].T
    return projected.reshape(*original_shape[:-1], projected.shape[-1])


def _representation_id(cfg: TopoVLMConfig) -> str:
    return (
        f"{cfg.model.vlm.model_id}:"
        f"{cfg.model.vlm.representation}:"
        f"layers={','.join(str(layer) for layer in cfg.model.vlm.hidden_layer_indices)}:"
        f"pool={cfg.model.vlm.visual_pool_grid}:"
        f"projection={cfg.model.vlm.projection}:{cfg.model.vlm.projection_dim}"
    )
