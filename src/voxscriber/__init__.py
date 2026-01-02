"""VoxScriber - Local speaker diarization for Apple Silicon."""

# Apply macOS fixes on import
from .fix_path import setup_macos_environment
setup_macos_environment()

from .pipeline import DiarizationPipeline, PipelineConfig

__all__ = ["DiarizationPipeline", "PipelineConfig"]
__version__ = "0.1.9"
