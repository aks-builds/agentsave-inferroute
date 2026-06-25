"""Shared pytest fixtures for InferRoute tests."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def turn1_messages():
    """A single user message — Turn 1."""
    return [{"role": "user", "content": "Hello, who are you?"}]


@pytest.fixture
def turn2_messages():
    """User + assistant + user — Turn 2+."""
    return [
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I am an AI assistant."},
        {"role": "user", "content": "Tell me more."},
    ]


@pytest.fixture
def turn2_messages_minimal():
    """Minimum Turn 2+: one exchange completed, new user message."""
    return [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What day is it?"},
    ]


@pytest.fixture
def chat_request_turn1(turn1_messages):
    """Full chat completions request body, Turn 1."""
    return {
        "model": "meta-llama/Llama-3-8b-instruct",
        "messages": turn1_messages,
        "stream": False,
    }


@pytest.fixture
def chat_request_turn2(turn2_messages):
    """Full chat completions request body, Turn 2+."""
    return {
        "model": "meta-llama/Llama-3-8b-instruct",
        "messages": turn2_messages,
        "stream": False,
    }
