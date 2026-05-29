"""Base VLM encoder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class VLMEncoder(ABC):
    """Convert image-goal pairs into fixed-dimensional frozen features."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def encode_image_goal(self, image: object, goal_text: str):
        raise NotImplementedError
