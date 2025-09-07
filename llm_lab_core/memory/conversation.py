"""
Conversation history management.
Corresponds to section 6.1 of the detailed design document.
"""
from typing import List
from llm_lab_core.core.types import Message

class ConversationMemory:
    """
    Manages the conversation history for a single session, including turn limits
    and summarization logic.
    """
    def __init__(self, max_turns: int = 8) -> None:
        """
        Initializes the memory.

        Args:
            max_turns: The maximum number of recent turns to keep in full.
        """
        self.history: List[Message] = []
        self.max_turns = max_turns
        self._summary: str = ""

    def append(self, msg: Message) -> None:
        """
        Appends a message to the history.

        Args:
            msg: The message to add.
        """
        self.history.append(msg)

    def summarize(self) -> str:
        """
        Generates a summary of the conversation history.
        This is a placeholder for a more complex summarization algorithm.
        A real implementation could use a smaller LLM or a rule-based approach.
        """
        if len(self.history) > self.max_turns * 2: # Assuming user/assistant pairs
            # Simple summarization: just take the first user message
            # and a count of turns.
            first_user_message = next((m.content for m in self.history if m.role == 'user'), "")
            self._summary = f"Summary: The conversation started with the user asking about '{first_user_message[:30]}...'. There have been {len(self.history)} turns."
        return self._summary

    def build_context(self) -> List[Message]:
        """
        Builds the context for the LLM, combining a summary (if any) with the
        most recent turns.

        Returns:
            A list of messages to be used as context for the LLM.
        """
        self.summarize() # Update summary if needed

        context: List[Message] = []
        if self._summary:
            context.append(Message(role='system', content=f"Conversation Summary:\n{self._summary}"))

        # Add the most recent turns
        num_messages_to_keep = self.max_turns * 2 # user + assistant
        context.extend(self.history[-num_messages_to_keep:])
        
        return context

    def clear(self) -> None:
        """Clears the conversation history and summary."""
        self.history = []
        self._summary = ""
