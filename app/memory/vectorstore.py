"""
Abstract interface for vector stores for Retrieval-Augmented Generation (RAG).
Corresponds to section 6.2 of the detailed design document.
"""
from typing import Protocol, List, Tuple, Dict, Any

class VectorStore(Protocol):
    """
    Protocol defining the interface for a vector store.
    This allows for different backend implementations (e.g., FAISS, ChromaDB)
    to be used interchangeably.
    """
    def add(self, doc_id: str, text: str, meta: Dict[str, Any]) -> None:
        """
        Adds a document to the vector store.

        Args:
            doc_id: A unique identifier for the document.
            text: The text content of the document to be embedded.
            meta: A dictionary of metadata associated with the document.
        """
        ...

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Searches the vector store for the most similar documents.

        Args:
            query: The query text.
            k: The number of results to return.

        Returns:
            A list of tuples, where each tuple contains:
            (document_id, similarity_score, metadata).
        """
        ...

    def load(self, path: str) -> None:
        """Loads the vector store index from a file."""
        ...

    def save(self, path: str) -> None:
        """Saves the vector store index to a file."""
        ...
