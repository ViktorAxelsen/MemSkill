"""
HotpotQA dataset processor.

HotpotQA is a multi-hop question answering dataset where each sample contains
a context (multiple documents) and a question that requires reasoning across
the documents to answer.

This processor supports fixed-length chunking of the context.
"""
from typing import List, Dict, Any, Optional
import tiktoken

from .base import DataProcessor, ChunkMode, register_processor


@register_processor("hotpotqa")
class HotpotQAProcessor(DataProcessor):
    """
    Data processor for HotpotQA dataset.

    HotpotQA format:
    {
        "context": "Document 1:\\n...\\nDocument 2:\\n...",
        "input": "question string",
        "num_docs": 50,
        "index": 1,
        "answers": ["answer1", "answer2", ...]
    }

    This processor supports fixed-length chunking where the context is split
    into chunks of approximately equal token length.
    """

    supported_chunk_modes = [
        ChunkMode.FIXED_LENGTH,
        ChunkMode.PARAGRAPH,
        ChunkMode.FULL_SESSION  # Treat entire context as one chunk
    ]

    def __init__(
        self,
        chunk_mode: str = "fixed-length",
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        tokenizer_name: str = "cl100k_base",
        show_progress: bool = False
    ):
        """
        Initialize HotpotQA processor.

        Args:
            chunk_mode: How to split context into chunks
            chunk_size: Target size of each chunk in tokens (for fixed-length mode)
            chunk_overlap: Number of overlapping tokens between chunks
            tokenizer_name: Tiktoken tokenizer name for token counting
            show_progress: Whether to show progress bars during processing
        """
        super().__init__(chunk_mode)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.show_progress = show_progress

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        except Exception:
            # Fallback to cl100k_base if specified tokenizer not found
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def extract_chunks(self, data: Dict) -> List[str]:
        """
        Extract text chunks from HotpotQA context data.

        Args:
            data: HotpotQA data dict containing 'context'

        Returns:
            List of text chunks
        """
        context = data.get('context', '')

        if not context:
            return []

        if self.chunk_mode == ChunkMode.FIXED_LENGTH:
            return self._split_fixed_length(context)
        elif self.chunk_mode == ChunkMode.PARAGRAPH:
            return self._split_by_paragraph(context)
        elif self.chunk_mode == ChunkMode.FULL_SESSION:
            return [context]
        else:
            return [context]

    def _split_fixed_length(self, text: str) -> List[str]:
        """
        Split text into fixed-length chunks based on token count.

        Args:
            text: Input text to split

        Returns:
            List of text chunks, each approximately chunk_size tokens
        """
        if not text:
            return []

        # Encode text to tokens
        tokens = self.tokenizer.encode(text)

        if len(tokens) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(tokens):
            # Get chunk of tokens
            end = start + self.chunk_size

            # Get the token slice
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Clean up the chunk (handle partial words at boundaries)
            chunk_text = self._clean_chunk_boundaries(chunk_text, text, start == 0, end >= len(tokens))

            if chunk_text.strip():
                chunks.append(chunk_text.strip())

            # Move start position with overlap
            start = end - self.chunk_overlap

            # Prevent infinite loop
            if start >= len(tokens) - self.chunk_overlap:
                break

        return chunks

    def _clean_chunk_boundaries(
        self,
        chunk_text: str,
        full_text: str,
        is_first: bool,
        is_last: bool
    ) -> str:
        """
        Clean chunk boundaries to avoid cutting words in the middle.

        Args:
            chunk_text: The chunk text
            full_text: The full original text
            is_first: Whether this is the first chunk
            is_last: Whether this is the last chunk

        Returns:
            Cleaned chunk text
        """
        # For simplicity, just trim whitespace
        # In production, you might want to find word boundaries
        return chunk_text.strip()

    def _split_by_paragraph(self, text: str) -> List[str]:
        """
        Split text by paragraph boundaries (double newlines or document markers).

        Args:
            text: Input text to split

        Returns:
            List of paragraph chunks
        """
        # Split by document markers or double newlines
        import re

        # Try to split by "Document X:" pattern first
        doc_pattern = r'(?=Document \d+:)'
        parts = re.split(doc_pattern, text)

        # Filter empty parts and strip
        chunks = [p.strip() for p in parts if p.strip()]

        if len(chunks) <= 1:
            # Fallback to paragraph splitting
            chunks = [p.strip() for p in text.split('\n\n') if p.strip()]

        return chunks if chunks else [text]

    def get_sample_id(self, data: Dict) -> str:
        """Get sample ID from HotpotQA data."""
        # Use index if available, otherwise generate from hash
        if 'index' in data:
            return f"hotpotqa_{data['index']}"
        return f"hotpotqa_{hash(data.get('input', ''))}"

    def get_qa_list(self, data: Dict) -> List[Dict[str, Any]]:
        """
        Extract QA list from HotpotQA data.

        Each HotpotQA sample has one question (input) and one or more answers.

        Returns:
            List containing a single QA dict with:
            - question: str
            - answer: str (first answer)
            - answers: List[str] (all valid answers)
        """
        question = data.get('input', '')
        answers = data.get('answers', [])

        if not question:
            return []

        # Handle case where answers might be a string
        if isinstance(answers, str):
            answers = [answers]

        # Use first answer as primary, keep all for evaluation
        primary_answer = answers[0] if answers else ''

        return [{
            'question': question,
            'answer': primary_answer,
            'answers': answers,  # Keep all valid answers for flexible matching
        }]

    def get_metadata(self, data: Dict) -> Dict[str, Any]:
        """
        Extract additional metadata.

        Returns:
            Dict with document count and other metadata
        """
        return {
            'num_docs': data.get('num_docs', 0),
            'index': data.get('index', -1),
            'chunk_size': self.chunk_size,
            'chunk_mode': self.chunk_mode.value,
        }


# Helper function for creating processor with common configurations
def create_hotpotqa_processor(
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    **kwargs
) -> HotpotQAProcessor:
    """
    Create a HotpotQA processor with common configuration.

    Args:
        chunk_size: Target size of each chunk in tokens
        chunk_overlap: Number of overlapping tokens between chunks
        **kwargs: Additional arguments passed to HotpotQAProcessor

    Returns:
        Configured HotpotQAProcessor instance
    """
    return HotpotQAProcessor(
        chunk_mode="fixed-length",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )
