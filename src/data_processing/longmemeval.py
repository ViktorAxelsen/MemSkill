"""
LongMemEval dataset processor.

LongMemEval is a long-context memory evaluation dataset with chat history sessions.
Each sample contains multiple haystack sessions (chat history) and a question-answer pair.
"""
from typing import List, Dict, Any
from tqdm import tqdm
import tiktoken

from .base import DataProcessor, ChunkMode, register_processor


@register_processor("longmemeval")
class LongMemEvalProcessor(DataProcessor):
    """
    Data processor for LongMemEval dataset.

    LongMemEval format:
    {
        "question": "...",
        "answer": "...",
        "question_date": "...",
        "haystack_sessions": [
            [{"role": "user/assistant", "content": "..."}...],
            ...
        ],
        "haystack_dates": ["date1", "date2", ...]
    }
    """

    supported_chunk_modes = [
        ChunkMode.TURN,
        ChunkMode.TURN_PAIR,
        ChunkMode.FULL_SESSION,
        ChunkMode.FIXED_LENGTH
    ]

    def __init__(
        self,
        chunk_mode: str = "turn-pair",
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        tokenizer_name: str = "cl100k_base",
        show_progress: bool = False
    ):
        """
        Initialize LongMemEval processor.

        Args:
            chunk_mode: How to split chat history into chunks
            chunk_size: Target size of each chunk in tokens (fixed-length mode)
            chunk_overlap: Number of overlapping tokens between chunks
            tokenizer_name: Tiktoken tokenizer name for token counting
            show_progress: Whether to show progress bars during processing
        """
        super().__init__(chunk_mode)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.show_progress = show_progress
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def extract_chunks(self, data: Dict) -> List[str]:
        """
        Extract text chunks from LongMemEval haystack sessions.

        Args:
            data: LongMemEval data dict containing 'haystack_sessions'

        Returns:
            List of text chunks with optional date prefixes
        """
        chunks = []
        haystack_sessions = data.get('haystack_sessions', [])
        haystack_dates = data.get('haystack_dates', [''] * len(haystack_sessions))

        session_iter = zip(haystack_sessions, haystack_dates)
        if self.show_progress:
            session_iter = tqdm(list(session_iter), desc='Extracting sessions', leave=False)

        for session_entry, session_date in session_iter:
            if not isinstance(session_entry, list):
                continue

            session_chunks = self._extract_session_chunks(
                session_entry, session_date
            )
            chunks.extend(session_chunks)

        return chunks

    def _extract_session_chunks(self, turns: List[Dict],
                                date_str: str) -> List[str]:
        """
        Extract chunks from a single session based on chunk mode.

        Args:
            turns: List of turn dicts with 'role' and 'content'
            date_str: Date string for this session

        Returns:
            List of text chunks
        """
        chunks = []

        if self.chunk_mode == ChunkMode.TURN:
            for turn in turns:
                text = f"{turn['role']}: {turn['content']}"
                if date_str:
                    text = f"Session Date: {date_str}\n{text}"
                chunks.append(text)

        elif self.chunk_mode == ChunkMode.TURN_PAIR:
            for j in range(0, len(turns), 2):
                pair_texts = []
                for k in range(j, min(j + 2, len(turns))):
                    pair_texts.append(f"{turns[k]['role']}: {turns[k]['content']}")

                text = "\n\n".join(pair_texts)
                if date_str:
                    text = f"Session Date: {date_str}\n\n{text}"
                chunks.append(text)

        elif self.chunk_mode == ChunkMode.FULL_SESSION:
            session_text = "\n\n".join([
                f"{turn['role']}: {turn['content']}"
                for turn in turns
            ])
            if date_str:
                session_text = f"Session Date: {date_str}\n\n{session_text}"
            chunks.append(session_text)
        elif self.chunk_mode == ChunkMode.FIXED_LENGTH:
            session_text = "\n\n".join([
                f"{turn['role']}: {turn['content']}"
                for turn in turns
            ])
            fixed_chunks = self._split_fixed_length(session_text)
            for chunk in fixed_chunks:
                text = chunk
                if date_str:
                    text = f"Session Date: {date_str}\n\n{text}"
                chunks.append(text)

        return chunks

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

        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.chunk_size:
            return [text]

        overlap = min(self.chunk_overlap, max(self.chunk_size - 1, 0))
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(chunk_text)
            start = end - overlap
            if start >= len(tokens) - overlap:
                break
        return chunks

    def get_sample_id(self, data: Dict) -> str:
        """Get sample ID from LongMemEval data."""
        question_id = data.get('question_id')
        if question_id:
            return str(question_id)
        # LongMemEval may not have explicit sample_id, generate from question hash
        if 'sample_id' in data:
            return data['sample_id']
        question = data.get('question', '')
        return f"lme_{hash(question) % 10**8}"

    def get_qa_list(self, data: Dict) -> List[Dict[str, Any]]:
        """
        Extract QA list from LongMemEval data.

        LongMemEval has a single question-answer pair per sample.

        Returns:
            List with single QA dict containing:
            - question: str
            - answer: str
            - question_date: str (optional)
        """
        if 'question' not in data or 'answer' not in data:
            return []

        return [{
            'question': data['question'],
            'answer': data['answer'],
            'question_date': data.get('question_date', '')
        }]

    def get_metadata(self, data: Dict) -> Dict[str, Any]:
        """
        Extract additional metadata.

        Returns:
            Dict with session count and other info
        """
        haystack_sessions = data.get('haystack_sessions', [])
        return {
            'num_sessions': len(haystack_sessions),
            'question_date': data.get('question_date', '')
        }
