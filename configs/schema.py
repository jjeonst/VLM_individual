"""Typed configuration schema for TopoVLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Describe Habitat data, cache, and loader inputs."""

    dataset_name: str = "habitat_objectnav"
    domain: str = "habitat"
    trajectory_source: str = "objectnav_shortest_path"
    cache_format: str = "single_action_graph"
    data_root: str = "/data/topovlm/habitat"
    habitat_config: str = "configs/habitat/pr2l_objectnav.yaml"
    split: str = "train"
    objectnav_dataset_dir: str = "datasets/objectnav/hm3d/v2/objectnav_hm3d_v2"
    scene_dataset_dir: str = "scene_datasets/hm3d"
    scene_dataset_config: str = "scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    episodes_manifest: str = "episodes/train/manifest.jsonl"
    graph_manifest: str = "graphs/prismatic_objectnav_train/manifest.jsonl"
    graph_cache_dir: str = "graphs/prismatic_objectnav_train"
    embeddings_dir: str = "embeddings/prismatic_objectnav_train"
    vlm_weights_dir: str = "/data/topovlm/vlm_weights/prismatic"
    image_key: str = "rgb"
    image_height: int = 224
    image_width: int = 224
    action_key: str = "action"
    goal_key: str = "goal_text"
    frame_stride: int = 1
    max_episodes: Optional[int] = None
    episode_selection_manifest: Optional[str] = None
    balanced_subset_size: Optional[int] = None
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    synthetic_debug: bool = False
    require_existing_cache: bool = True


@dataclass
class VLMConfig:
    """Describe the VLM adapter and frozen-weight cache source."""

    backend: str = "prismatic"
    model_id: str = "prism-dinosiglip+7b"
    weights_path: str = "/data/topovlm/vlm_weights/prismatic/prism-dinosiglip+7b"
    hf_token_path: Optional[str] = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    frozen: bool = True
    representation: str = "last_token"
    hidden_layer_indices: list[int] = field(default_factory=lambda: [-1])
    visual_pool_grid: int = 1
    visual_bank_reduction: str = "mean"
    projection: str = "none"
    projection_path: Optional[str] = None
    projection_dim: Optional[int] = None
    projection_fit_max_tokens: int = 8192
    include_generated_text: bool = False
    generation_seed: int = 0
    generation_temperature: float = 0.4
    max_new_tokens: int = 48
    output_dim: int = 4096
    prompt_template: str = "You are navigating an indoor scene. Goal: {goal_text}"
    cache_batch_size: int = 1


@dataclass
class TopologyConfig:
    """Describe how frame embeddings are compressed into graph nodes."""

    builder: str = "sequential_similarity"
    similarity_threshold: float = 0.9
    max_nodes: int = 128
    min_segment_len: int = 1
    normalize_embeddings: bool = True


@dataclass
class PolicyConfig:
    """Describe the graph-conditioned behavior-cloning policy."""

    type: str = "graph_transformer_bc"
    input_dim: int = 4096
    hidden_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 2
    dropout: float = 0.1
    num_actions: int = 4
    prediction_target: str = "graph"
    max_positions: int = 2048


@dataclass
class ModelConfig:
    """Group VLM, topology, and policy configuration."""

    vlm: VLMConfig = field(default_factory=VLMConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


@dataclass
class BehaviorCloningConfig:
    """Configure cross-entropy behavior cloning."""

    label_smoothing: float = 0.0
    class_weights: Optional[list[float]] = None
    inflection_weight: float = 1.0
    stop_turn_weight: float = 1.0
    stop_turn_action_ids: list[int] = field(default_factory=lambda: [0, 2, 3])


@dataclass
class ObjectivesConfig:
    """Select training objectives and their parameters."""

    names: list[str] = field(default_factory=lambda: ["behavior_cloning"])
    behavior_cloning: BehaviorCloningConfig = field(default_factory=BehaviorCloningConfig)


@dataclass
class EvalConfig:
    """Describe offline validation and diagnostic settings."""

    num_episodes: int = 16
    max_steps: int = 500
    episode_timeout_seconds: int = 120
    success_distance: float = 0.2
    output_dir: str = "artifacts/eval"
    write_predictions: bool = False


@dataclass
class TopoVLMConfig:
    """Top-level canonical TopoVLM training configuration."""

    config_name: str = "habitat/pr2l_hm3d_bc"
    run_name: str = "pr2l_hm3d_bc"
    seed: int = 42
    debug: bool = False
    wandb: bool = False
    wandb_entity: str = "topovlm"
    wandb_project: str = "TopoVLM"
    wandb_group: Optional[str] = "prismatic_graph_policy"
    wandb_run_name: Optional[str] = None
    wandb_contract_path: str = "artifacts/contracts/wandb_identity_contract.json"
    wandb_contract_role_id: str = "habitat_bc"
    output_root: str = "checkpoints"
    device: str = "cuda"
    max_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    save_every_epochs: int = 1
    gradient_accumulation_steps: int = 1
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    objectives: ObjectivesConfig = field(default_factory=ObjectivesConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
