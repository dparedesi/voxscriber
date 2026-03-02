"""VoxScriber - Local speaker diarization for macOS (MLX) and Linux (CUDA/CPU)."""

from .pipeline import DiarizationPipeline, PipelineConfig

__all__ = ["DiarizationPipeline", "PipelineConfig"]
__version__ = "0.2.5"
