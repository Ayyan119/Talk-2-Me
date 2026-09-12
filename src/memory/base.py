"""Module: base
Layer: Core/Domain
Purpose: Defines the abstract interface contract for short-term conversation memory.
Dependencies: abc, src.core.types, src.core.exceptions
"""

from abc import ABC, abstractmethod

from src.core.types import Message


class ConversationMemory(ABC):
    """Abstract base contract for managing short-term conversation context and history."""

    @abstractmethod
    def add_message(self, message: Message) -> None:
        """Appends a new conversation message to memory storage.

        Args:
            message: The Message instance to store in conversation history.

        Raises:
            ConversationMemoryError: If message validation fails or cannot be saved.
        """
        raise NotImplementedError

    @abstractmethod
    def get_context(self, max_turns: int | None = None) -> list[Message]:
        """Retrieves recent conversation messages bounded by turn or token limits.

        Args:
            max_turns: Optional limit on the number of dialogue turns (pairs) to return.
                       If None, default context window strategy is used.

        Returns:
            A list of Message objects in chronological order suitable for LLM prompt context.

        Raises:
            ConversationMemoryError: If message retrieval or formatting encounters an error.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Clears all stored conversation messages from memory.

        Raises:
            ConversationMemoryError: If the memory reset operation fails.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_messages(self) -> list[Message]:
        """Retrieves the complete sequence of all messages stored in this session.

        Returns:
            Full list of chronological Message objects recorded in memory.

        Raises:
            ConversationMemoryError: If history extraction fails.
        """
        raise NotImplementedError
