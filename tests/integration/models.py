"""Test model registry for integration tests.

Provides a registry of models to test against. Only lightweight models are
enabled by default. Additional models can be enabled via the NDIF_TEST_MODELS
environment variable.

Usage:
    # Enable extra models at runtime:
    NDIF_TEST_MODELS=pythia-70m,qwen2-0.5b pytest tests/integration/ --run-remote -v
"""

import os
from dataclasses import dataclass


@dataclass
class TestModel:
    name: str  # human-readable name
    model_key: str  # nnsight-formatted model key
    enabled: bool  # whether to run by default


MODEL_REGISTRY: dict[str, TestModel] = {}


def register_model(name: str, model_key: str, enabled: bool = False):
    MODEL_REGISTRY[name] = TestModel(name=name, model_key=model_key, enabled=enabled)


def get_enabled_models() -> list[TestModel]:
    """Return models enabled by default + any enabled via NDIF_TEST_MODELS env var."""
    extra = [
        s.strip()
        for s in os.environ.get("NDIF_TEST_MODELS", "").split(",")
        if s.strip()
    ]
    return [m for m in MODEL_REGISTRY.values() if m.enabled or m.name in extra]


# Default registrations
register_model("gpt2", "openai-community/gpt2", enabled=True)
register_model("pythia-70m", "EleutherAI/pythia-70m", enabled=False)
register_model("qwen2-0.5b", "Qwen/Qwen2-0.5B", enabled=False)
