"""LLM model factory — creates the right model based on environment config.

Auto-detects the provider from environment variables:
- If ``OLLAMA_MODEL`` is set → Ollama
- Otherwise → AWS Bedrock (backward-compatible default)

All provider-specific imports are lazy so that missing dependencies (e.g.
``boto3`` when using Ollama) never cause import-time failures.
"""

from __future__ import annotations

import os

from utils.logging_helpers import get_logger, log_info_event

logger = get_logger(__name__)

# Fallback Bedrock model when env vars are not set.
_DEFAULT_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"


def get_provider() -> str:
    """Return the active LLM provider name: ``"ollama"`` or ``"bedrock"``."""
    return "ollama" if os.getenv("OLLAMA_MODEL") else "bedrock"


def create_model(*, tier: str = "default"):
    """Create an LLM model instance for the active provider.

    Args:
        tier: ``"default"`` for the primary model or ``"small"`` for a
            cheaper / smaller model (used by user_behavior_analysis_agent).

    Returns:
        A Strands-compatible model (``BedrockModel`` or ``OllamaModel``).
    """
    provider = get_provider()

    if provider == "ollama":
        from strands.models.ollama import OllamaModel

        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if tier == "small":
            model_id = os.getenv("OLLAMA_SMALL_MODEL") or os.getenv("OLLAMA_MODEL")
        else:
            model_id = os.getenv("OLLAMA_MODEL")

        log_info_event(
            logger,
            f"Creating OllamaModel (tier={tier}, model={model_id}, host={host})",
            "model_factory.ollama",
            tier=tier,
            model_id=model_id,
            host=host,
        )
        return OllamaModel(host=host, model_id=model_id)

    # Default: Bedrock
    import boto3
    from strands.models.bedrock import BedrockModel

    if tier == "small":
        model_id = os.getenv("BEDROCK_SMALL_MODEL") or os.getenv("BEDROCK_MODEL") or _DEFAULT_BEDROCK_MODEL_ID
    else:
        model_id = os.getenv("BEDROCK_MODEL") or _DEFAULT_BEDROCK_MODEL_ID

    log_info_event(
        logger,
        f"Creating BedrockModel (tier={tier}, model={model_id})",
        "model_factory.bedrock",
        tier=tier,
        model_id=model_id,
    )
    return BedrockModel(
        model_id=model_id,
        boto_session=boto3.Session(),
        streaming=True,
    )
