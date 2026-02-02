"""
Data processing module for Agentic Memory.

This module provides data processors for different datasets with a unified interface.
The core abstraction is chunk-based processing where long context is split into
chunks for sequential reading and memory operations.

Usage:
    from src.data_processing import get_processor

    # Get processor for a specific dataset
    processor = get_processor("locomo", chunk_mode="turn-pair")

    # Process a single sample
    sample = processor.process(raw_data)

    # Access processed data
    chunks = sample.chunks  # List of text chunks
    qa_list = sample.qa_list  # List of QA items
"""
from .base import (
    DataProcessor,
    DataSample,
    ChunkMode,
    MultiDatasetProcessor,
    get_processor,
    register_processor,
    list_processors,
)
from .locomo import LoCoMoProcessor
from .longmemeval import LongMemEvalProcessor
from .hotpotqa import HotpotQAProcessor
from .alfworld import ALFWorldOfflineDataset, chunk_trajectories_by_tokens


__all__ = [
    # Base classes
    "DataProcessor",
    "DataSample",
    "ChunkMode",
    "MultiDatasetProcessor",
    # Registry functions
    "get_processor",
    "register_processor",
    "list_processors",
    # Concrete processors
    "LoCoMoProcessor",
    "LongMemEvalProcessor",
    "HotpotQAProcessor",
    "ALFWorldOfflineDataset",
    "chunk_trajectories_by_tokens",
]
