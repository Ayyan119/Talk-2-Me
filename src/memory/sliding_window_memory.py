"""Module: sliding_window_memory
Layer: Adapter/Implementation (Layer 2)
Purpose: In-memory sliding window conversation memory maintaining bounded dialogue turns.
Dependencies: src.core.types, src.core.exceptions, src.memory.base
"""

from src.core.exceptions import ConversationMemoryError
from src.core.types import Message, MessageRole
from src.memory.base import ConversationMemory


class SlidingWindowMemory(ConversationMemory):
    """In-memory sliding window implementation of conversation memory.

    Stores dialogue turns chronologically and retrieves a bounded context window
    while always preserving any initial system prompt instructions.

    Attributes:
        default_max_turns: Default number of dialogue turns (user + assistant pairs)
            to retain in context when max_turns is not explicitly specified.
    """

    def __init__(self, default_max_turns: int | None = 10) -> None:
        """Initializes an empty sliding window conversation memory.

        Args:
            default_max_turns: Default maximum number of dialogue turns to return in context.
                Must be greater than 0 if specified.

        Raises:
            ConversationMemoryError: If default_max_turns is less than or equal to 0.
        """
        if default_max_turns is not None and default_max_turns <= 0:
            raise ConversationMemoryError(
                f"default_max_turns must be positive, got {default_max_turns}"
            )
        self.default_max_turns = default_max_turns
        self._messages: list[Message] = []

    def add_message(self, message: Message) -> None:
        """Appends a new validated Message instance to in-memory history.

        Args:
            message: The Message instance to append to the conversation history.

        Raises:
            ConversationMemoryError: If message is not a valid Message instance.
        """
        if not isinstance(message, Message):
            raise ConversationMemoryError(
                f"Expected Message instance, got {type(message).__name__}"
            )
        self._messages.append(message)

    def get_context(self, max_turns: int | None = None) -> list[Message]:
        """Retrieves recent conversation messages bounded by the requested number of turns.

        Preserves any system messages at the start of context, while truncating
        older conversational dialogue turns to fit within max_turns.

        Args:
            max_turns: Optional limit on dialogue turns (user + assistant pairs).
                If None, uses self.default_max_turns. If 0, returns only system messages.

        Returns:
            A list of Message objects in chronological order suitable for LLM prompt context.

        Raises:
            ConversationMemoryError: If max_turns is negative.
        """
        effective_turns = self.default_max_turns if max_turns is None else max_turns
        if effective_turns < 0:
            raise ConversationMemoryError(
                f"max_turns must be non-negative, got {effective_turns}"
            )

        if not self._messages:
            return []

        # Separate system messages (e.g. initial prompt) from dialogue messages
        system_messages: list[Message] = [
            msg for msg in self._messages if msg.role == MessageRole.SYSTEM
        ]
        dialogue_messages: list[Message] = [
            msg for msg in self._messages if msg.role != MessageRole.SYSTEM
        ]

        if effective_turns == 0:
            return list(system_messages)

        # Each turn consists of up to 2 messages (user + assistant)
        max_messages = effective_turns * 2
        windowed_dialogue = dialogue_messages[-max_messages:]

        return system_messages + windowed_dialogue

    def clear(self) -> None:
        """Clears all stored conversation messages from in-memory history."""
        self._messages.clear()

    def get_all_messages(self) -> list[Message]:
        """Retrieves a copy of the entire message sequence recorded in this session.

        Returns:
            Full chronological list of all Message instances recorded in memory.
        """
        return list(self._messages)
