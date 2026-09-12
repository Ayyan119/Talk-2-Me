"""Module: test_sliding_window_memory
Layer: Testing
Purpose: Comprehensive unit test suite for SlidingWindowMemory implementation.
Dependencies: pytest, src.core.types, src.core.exceptions, src.memory.sliding_window_memory
"""

import pytest

from src.core.exceptions import ConversationMemoryError
from src.core.types import Message, MessageRole
from src.memory.sliding_window_memory import SlidingWindowMemory


def test_add_and_retrieve_messages() -> None:
    """Tests adding messages to memory and retrieving all messages."""
    memory = SlidingWindowMemory()
    assert memory.get_all_messages() == []

    msg1 = Message(role=MessageRole.USER, content="Hello")
    msg2 = Message(role=MessageRole.ASSISTANT, content="Hi there!")

    memory.add_message(msg1)
    memory.add_message(msg2)

    all_messages = memory.get_all_messages()
    assert len(all_messages) == 2
    assert all_messages[0] == msg1
    assert all_messages[1] == msg2


def test_sliding_window_limits_turns() -> None:
    """Tests that max_turns limits dialogue history to the requested number of turn pairs."""
    memory = SlidingWindowMemory()

    # Add 4 turns (8 messages)
    for i in range(1, 5):
        memory.add_message(Message(role=MessageRole.USER, content=f"Question {i}"))
        memory.add_message(Message(role=MessageRole.ASSISTANT, content=f"Answer {i}"))

    # Request last 2 turns (4 messages: Q3, A3, Q4, A4)
    context = memory.get_context(max_turns=2)
    assert len(context) == 4
    assert [m.content for m in context] == [
        "Question 3",
        "Answer 3",
        "Question 4",
        "Answer 4",
    ]

    # Request 1 turn (2 messages: Q4, A4)
    context_1 = memory.get_context(max_turns=1)
    assert len(context_1) == 2
    assert [m.content for m in context_1] == ["Question 4", "Answer 4"]


def test_system_prompt_preserved_in_sliding_window() -> None:
    """Tests that system prompt is always retained at the start of context regardless of window size."""
    memory = SlidingWindowMemory()

    system_msg = Message(
        role=MessageRole.SYSTEM, content="You are a helpful voice assistant."
    )
    memory.add_message(system_msg)

    # Add 5 dialogue turns
    for i in range(1, 6):
        memory.add_message(Message(role=MessageRole.USER, content=f"User msg {i}"))
        memory.add_message(Message(role=MessageRole.ASSISTANT, content=f"Bot resp {i}"))

    # When requesting only 1 turn, system prompt must still be present at index 0
    context = memory.get_context(max_turns=1)
    assert len(context) == 3  # System message + 1 turn (2 dialogue messages)
    assert context[0] == system_msg
    assert context[1].content == "User msg 5"
    assert context[2].content == "Bot resp 5"

    # When requesting 0 turns, only system prompt is returned
    context_0 = memory.get_context(max_turns=0)
    assert len(context_0) == 1
    assert context_0[0] == system_msg


def test_clear_memory() -> None:
    """Tests that clear() resets all stored conversation messages."""
    memory = SlidingWindowMemory()
    memory.add_message(Message(role=MessageRole.USER, content="Test message"))
    assert len(memory.get_all_messages()) == 1

    memory.clear()
    assert memory.get_all_messages() == []
    assert memory.get_context() == []


def test_validation_errors() -> None:
    """Tests that invalid inputs raise ConversationMemoryError."""
    # Invalid default_max_turns
    with pytest.raises(ConversationMemoryError, match="must be positive"):
        SlidingWindowMemory(default_max_turns=0)

    with pytest.raises(ConversationMemoryError, match="must be positive"):
        SlidingWindowMemory(default_max_turns=-5)

    memory = SlidingWindowMemory()

    # Invalid message object
    with pytest.raises(ConversationMemoryError, match="Expected Message instance"):
        memory.add_message("invalid message string")  # type: ignore[arg-type]

    # Negative max_turns in get_context
    with pytest.raises(ConversationMemoryError, match="must be non-negative"):
        memory.get_context(max_turns=-1)
