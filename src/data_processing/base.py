"""
Base classes for data processing.

This module provides abstract interfaces for data processing that can be
extended to support different datasets. The core abstraction is chunk-based
processing where long context is split into chunks for sequential reading.

Key concepts:
- Chunk: A unit of text to be processed (could be a turn, session, paragraph, etc.)
- ChunkMode: How to split data into chunks (turn, turn-pair, full-session, etc.)
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm


class ChunkMode(Enum):
    """Defines how to split data into chunks."""
    TURN = "turn"                   # Each utterance/turn is a chunk
    TURN_PAIR = "turn-pair"         # Pairs of turns (e.g., user-assistant)
    FULL_SESSION = "full-session"   # Entire session as one chunk
    PARAGRAPH = "paragraph"         # Split by paragraphs
    FIXED_LENGTH = "fixed-length"   # Fixed token/char length chunks


@dataclass
class DataSample:
    """
    Standardized data sample for training/evaluation.

    This provides a unified interface regardless of the original dataset format.
    Different datasets will map their data to this structure.
    """
    sample_id: str
    chunks: List[str]               # List of text chunks for sequential processing
    qa_list: List[Dict[str, Any]]   # List of QA items for evaluation
    metadata: Dict[str, Any]        # Additional dataset-specific metadata

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DataProcessor(ABC):
    """
    Abstract base class for data processing.

    Subclasses should implement methods for extracting chunks and QA data
    from dataset-specific formats. This enables:
    1. Different datasets to be processed uniformly
    2. Multi-dataset training by combining processors
    3. Easy extension for new datasets
    """

    # Supported chunk modes for this processor
    supported_chunk_modes: List[ChunkMode] = [
        ChunkMode.TURN,
        ChunkMode.TURN_PAIR,
        ChunkMode.FULL_SESSION
    ]

    def __init__(self, chunk_mode: str = "turn-pair"):
        """
        Initialize the data processor.

        Args:
            chunk_mode: How to split data into chunks
        """
        self.chunk_mode = self._parse_chunk_mode(chunk_mode)

    def is_interactive(self) -> bool:
        """
        Whether this dataset is interactive (observations revealed step by step).

        Interactive datasets should override this and provide iter_chunks() that
        yields observations without exposing future steps.
        """
        return False

    def supports_parallel_env(self) -> bool:
        """
        Whether interactive episodes can run in parallel safely.
        """
        return False

    def _parse_chunk_mode(self, mode: str) -> ChunkMode:
        """Parse chunk mode string to enum."""
        mode_map = {
            "turn": ChunkMode.TURN,
            "turn-pair": ChunkMode.TURN_PAIR,
            "full-session": ChunkMode.FULL_SESSION,
            "paragraph": ChunkMode.PARAGRAPH,
            "fixed-length": ChunkMode.FIXED_LENGTH,
        }
        if mode not in mode_map:
            raise ValueError(f"Unknown chunk mode: {mode}. Supported: {list(mode_map.keys())}")
        chunk_mode = mode_map[mode]
        if chunk_mode not in self.supported_chunk_modes:
            raise ValueError(f"Chunk mode {mode} not supported by {self.__class__.__name__}")
        return chunk_mode

    @abstractmethod
    def extract_chunks(self, data: Dict) -> List[str]:
        """
        Extract text chunks from raw data.

        This is the core method that converts dataset-specific format
        into a list of text chunks for sequential processing.

        Args:
            data: Raw data dict (dataset-specific format)

        Returns:
            List of text chunks
        """
        pass

    def iter_chunks(self, data: Dict):
        """
        Yield chunks sequentially.

        Default behavior yields from extract_chunks(). Interactive datasets
        should override to stream observations step-by-step.
        """
        return iter(self.extract_chunks(data))

    def get_episode_length(self, data: Dict) -> Optional[int]:
        """
        Return episode length if known. Interactive datasets may return None.
        """
        if self.is_interactive():
            for key in ("episode_length", "num_steps", "max_steps", "trajectory_length"):
                value = data.get(key) if isinstance(data, dict) else None
                if isinstance(value, int) and value > 0:
                    return value
            return None
        return len(self.extract_chunks(data))

    @abstractmethod
    def get_sample_id(self, data: Dict) -> str:
        """
        Get unique identifier for a data sample.

        Args:
            data: Raw data dict

        Returns:
            Sample ID string
        """
        pass

    @abstractmethod
    def get_qa_list(self, data: Dict) -> List[Dict[str, Any]]:
        """
        Extract QA list from raw data.

        Args:
            data: Raw data dict

        Returns:
            List of QA dicts, each containing at least 'question' and 'answer'
        """
        pass

    def get_metadata(self, data: Dict) -> Dict[str, Any]:
        """
        Extract additional metadata from raw data.

        Override in subclasses to provide dataset-specific metadata.

        Args:
            data: Raw data dict

        Returns:
            Metadata dict
        """
        return {}

    def process(self, data: Dict) -> DataSample:
        """
        Process raw data into standardized DataSample.

        Args:
            data: Raw data dict

        Returns:
            Standardized DataSample
        """
        return DataSample(
            sample_id=self.get_sample_id(data),
            chunks=self.extract_chunks(data),
            qa_list=self.get_qa_list(data),
            metadata=self.get_metadata(data)
        )

    def process_batch(self, data_list: List[Dict],
                      show_progress: bool = True) -> List[DataSample]:
        """
        Process a batch of raw data.

        Args:
            data_list: List of raw data dicts
            show_progress: Whether to show progress bar

        Returns:
            List of DataSamples
        """
        iterator = tqdm(data_list, desc="Processing data") if show_progress else data_list
        return [self.process(data) for data in iterator]


class MultiDatasetProcessor:
    """
    Processor for multi-dataset training.

    Combines multiple DataProcessors to handle data from different datasets.
    Supports weighted sampling for balanced training.
    """

    def __init__(self):
        self.processors: Dict[str, DataProcessor] = {}
        self.weights: Dict[str, float] = {}

    def register(self, name: str, processor: DataProcessor, weight: float = 1.0):
        """
        Register a dataset processor.

        Args:
            name: Dataset name
            processor: DataProcessor instance
            weight: Sampling weight for this dataset
        """
        self.processors[name] = processor
        self.weights[name] = weight

    def get_processor(self, name: str) -> DataProcessor:
        """Get processor by name."""
        if name not in self.processors:
            raise ValueError(f"Unknown dataset: {name}. Registered: {list(self.processors.keys())}")
        return self.processors[name]

    def process(self, data: Dict, dataset_name: str) -> DataSample:
        """
        Process data from a specific dataset.

        Args:
            data: Raw data dict
            dataset_name: Name of the dataset

        Returns:
            DataSample with dataset name in metadata
        """
        processor = self.get_processor(dataset_name)
        sample = processor.process(data)
        sample.metadata['dataset'] = dataset_name
        return sample


# Global registry for data processors
_PROCESSOR_REGISTRY: Dict[str, type] = {}


def register_processor(name: str):
    """
    Decorator to register a data processor class.

    Usage:
        @register_processor("my_dataset")
        class MyDataProcessor(DataProcessor):
            ...
    """
    def decorator(cls):
        _PROCESSOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_processor(name: str, **kwargs) -> DataProcessor:
    """
    Get a data processor instance by name.

    Args:
        name: Registered processor name
        **kwargs: Arguments to pass to processor constructor

    Returns:
        DataProcessor instance
    """
    if name not in _PROCESSOR_REGISTRY:
        raise ValueError(f"Unknown processor: {name}. Available: {list(_PROCESSOR_REGISTRY.keys())}")
    return _PROCESSOR_REGISTRY[name](**kwargs)


def list_processors() -> List[str]:
    """List all registered processor names."""
    return list(_PROCESSOR_REGISTRY.keys())
