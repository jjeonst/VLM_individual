"""VLM encoder adapters."""

from configs.schema import VLMConfig
from encoders.base import VLMEncoder
from encoders.prismatic import PrismaticEncoder


def build_vlm_encoder(config: VLMConfig) -> VLMEncoder:
    if config.backend == "prismatic":
        return PrismaticEncoder(config)
    raise ValueError(f"Unsupported VLM backend: {config.backend}")


__all__ = ["VLMEncoder", "PrismaticEncoder", "build_vlm_encoder"]
