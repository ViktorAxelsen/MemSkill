"""
LoCoMo dataset processor.

LoCoMo is a long-context conversation memory dataset with multi-session dialogues.
Each sample contains multiple sessions with dialogue turns and associated QA pairs.
"""
from typing import List, Dict, Any
from tqdm import tqdm
import tiktoken

from .base import DataProcessor, ChunkMode, register_processor


@register_processor("locomo")
class LoCoMoProcessor(DataProcessor):
    """
    Data processor for LoCoMo dataset.

    LoCoMo format:
    {
        "sample_id": "...",
        "conversation": {
            "session_1": [{"speaker": "A", "text": "...", "blip_caption": "..."}...],
            "session_1_date_time": "...",
            "session_2": [...],
            ...
        },
        "qa": [
            {"question": "...", "answer": "...", "category": 1, "evidence": [...]}
            ...
        ]
    }

    Categories:
    - 1: Multi-hop
    - 2: Temporal
    - 3: Open-domain
    - 4: Single-hop
    - 5: Adversarial (skipped in evaluation)
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
        Initialize LoCoMo processor.

        Args:
            chunk_mode: How to split dialogues into chunks
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
        Extract text chunks from LoCoMo conversation data.

        Args:
            data: LoCoMo data dict containing 'conversation'

        Returns:
            List of text chunks with optional date prefixes
        """
        chunks = []
        conv_data = data.get('conversation', {})

        # Find all session numbers
        session_nums = [
            int(k.split('_')[-1])
            for k in conv_data.keys()
            if 'session' in k and 'date_time' not in k
        ]

        if not session_nums:
            return chunks

        session_range = range(min(session_nums), max(session_nums) + 1)
        iterator = tqdm(session_range, desc='Extracting sessions', leave=False) \
            if self.show_progress else session_range

        for i in iterator:
            session_key = f"session_{i}"
            datetime_key = f"session_{i}_date_time"

            if session_key not in conv_data:
                print("Skipping session {}".format(session_key))
                continue

            date_str = conv_data.get(datetime_key, '')
            session_dialogs = conv_data[session_key]

            session_chunks = self._extract_session_chunks(
                session_dialogs, date_str
            )
            chunks.extend(session_chunks)

        return chunks

    def _extract_session_chunks(self, dialogs: List[Dict],
                                date_str: str) -> List[str]:
        """
        Extract chunks from a single session based on chunk mode.

        Args:
            dialogs: List of dialogue turns
            date_str: Date string for this session

        Returns:
            List of text chunks
        """
        chunks = []

        if self.chunk_mode == ChunkMode.TURN:
            for dialog in dialogs:
                text = self._format_turn(dialog)
                if date_str:
                    text = f"DATE: {date_str}\n{text}"
                chunks.append(text)

        elif self.chunk_mode == ChunkMode.TURN_PAIR:
            for j in range(0, len(dialogs), 2):
                pair_texts = []
                for k in range(j, min(j + 2, len(dialogs))):
                    pair_texts.append(self._format_turn(dialogs[k]))

                text = "\n".join(pair_texts)
                if date_str:
                    text = f"DATE: {date_str}\nCONVERSATION:\n{text}"
                chunks.append(text)

        elif self.chunk_mode == ChunkMode.FULL_SESSION:
            session_texts = [self._format_turn(d) for d in dialogs]
            text = "\n".join(session_texts)
            if date_str:
                text = f"DATE: {date_str}\nCONVERSATION:\n{text}"
            chunks.append(text)
        elif self.chunk_mode == ChunkMode.FIXED_LENGTH:
            session_texts = [self._format_turn(d) for d in dialogs]
            base_text = "\n".join(session_texts)
            fixed_chunks = self._split_fixed_length(base_text)
            for chunk in fixed_chunks:
                text = chunk
                if date_str:
                    text = f"DATE: {date_str}\nCONVERSATION:\n{text}"
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

    def _format_turn(self, dialog: Dict) -> str:
        """
        Format a single dialogue turn.

        Args:
            dialog: Dialogue turn dict with 'speaker', 'text', and optional 'blip_caption'

        Returns:
            Formatted turn string
        """
        text = f'{dialog["speaker"]} said, "{dialog["text"]}"'
        if "blip_caption" in dialog:
            text += f' and shared {dialog["blip_caption"]}.'
        return text

    def get_sample_id(self, data: Dict) -> str:
        """Get sample ID from LoCoMo data."""
        return data.get('sample_id', str(id(data)))

    def get_qa_list(self, data: Dict) -> List[Dict[str, Any]]:
        """
        Extract QA list from LoCoMo data.

        Returns:
            List of QA dicts, each containing:
            - question: str
            - answer: str
            - category: int (1-5)
            - evidence: List[str] (optional)
        """
        return data.get('qa', [])

    def get_metadata(self, data: Dict) -> Dict[str, Any]:
        """
        Extract additional metadata.

        Returns:
            Dict with speakers and other metadata
        """
        conv_data = data.get('conversation', {})
        speakers = set()

        # Extract unique speakers from all sessions
        for key, value in conv_data.items():
            if 'session' in key and 'date_time' not in key:
                for dialog in value:
                    if 'speaker' in dialog:
                        speakers.add(dialog['speaker'])

        return {
            'speakers': list(speakers),
            'num_sessions': len([k for k in conv_data.keys()
                               if 'session' in k and 'date_time' not in k])
        }
