"""Unit tests for the LLM model factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from utils.model_factory import create_model, get_provider

pytestmark = pytest.mark.unit


class TestGetProvider:
    """Tests for provider auto-detection."""

    def test_returns_bedrock_by_default(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        assert get_provider() == "bedrock"

    def test_returns_ollama_when_ollama_model_set(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "llama3")
        assert get_provider() == "ollama"


class TestCreateModelBedrock:
    """Tests for Bedrock model creation."""

    @pytest.fixture(autouse=True)
    def _use_bedrock(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.setenv("BEDROCK_MODEL", "arn:aws:bedrock:us-east-1:123:model/test")
        monkeypatch.setenv("BEDROCK_SMALL_MODEL", "arn:aws:bedrock:us-east-1:123:model/haiku")

    @patch("strands.models.bedrock.BedrockModel.__init__", return_value=None)
    @patch("boto3.Session")
    def test_creates_default_bedrock_model(self, mock_session_cls, mock_init):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        create_model()

        mock_init.assert_called_once_with(
            model_id="arn:aws:bedrock:us-east-1:123:model/test",
            boto_session=mock_session,
            streaming=True,
        )

    @patch("strands.models.bedrock.BedrockModel.__init__", return_value=None)
    @patch("boto3.Session")
    def test_creates_small_bedrock_model(self, mock_session_cls, mock_init):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        create_model(tier="small")

        mock_init.assert_called_once_with(
            model_id="arn:aws:bedrock:us-east-1:123:model/haiku",
            boto_session=mock_session,
            streaming=True,
        )


class TestCreateModelOllama:
    """Tests for Ollama model creation."""

    @pytest.fixture(autouse=True)
    def _use_ollama(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "llama3")
        monkeypatch.delenv("OLLAMA_SMALL_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)

    @patch("strands.models.ollama.OllamaModel.__init__", return_value=None)
    def test_creates_default_ollama_model(self, mock_init):
        create_model()

        mock_init.assert_called_once_with(
            host="http://localhost:11434",
            model_id="llama3",
        )

    @patch("strands.models.ollama.OllamaModel.__init__", return_value=None)
    def test_creates_small_ollama_model_fallback(self, mock_init):
        """Small tier falls back to OLLAMA_MODEL when OLLAMA_SMALL_MODEL is not set."""
        create_model(tier="small")

        mock_init.assert_called_once_with(
            host="http://localhost:11434",
            model_id="llama3",
        )

    @patch("strands.models.ollama.OllamaModel.__init__", return_value=None)
    def test_creates_small_ollama_model_explicit(self, mock_init, monkeypatch):
        monkeypatch.setenv("OLLAMA_SMALL_MODEL", "phi3")

        create_model(tier="small")

        mock_init.assert_called_once_with(
            host="http://localhost:11434",
            model_id="phi3",
        )

    @patch("strands.models.ollama.OllamaModel.__init__", return_value=None)
    def test_uses_custom_host(self, mock_init, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://remote:11434")

        create_model()

        mock_init.assert_called_once_with(
            host="http://remote:11434",
            model_id="llama3",
        )
