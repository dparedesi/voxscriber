"""
MLX Whisper Transcriber Module

Uses Apple's MLX-optimized Whisper for fast, accurate transcription
with word-level timestamps on Apple Silicon.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Word:
    """Represents a transcribed word with timing."""
    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class Segment:
    """Represents a transcription segment."""
    text: str
    start: float
    end: float
    words: List[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [asdict(w) for w in self.words]
        }


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str
    segments: List[Segment]
    language: str
    duration: float

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "duration": self.duration
        }


class Transcriber:
    """MLX Whisper-based transcriber optimized for Apple Silicon."""

    # Available models from smallest to largest
    MODELS = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        # Quantized versions for faster inference
        "large-4bit": "mlx-community/whisper-large-v3-mlx-4bit",
        "large-8bit": "mlx-community/whisper-large-v3-mlx-8bit",
    }

    def __init__(
        self,
        model: str = "large-v3-turbo",
        language: Optional[str] = None,
    ):
        """
        Initialize transcriber.

        Args:
            model: Model name (see MODELS) or HuggingFace repo path
            language: Force specific language (e.g., "en", "es"). None for auto-detect.
        """
        self.model_path = self.MODELS.get(model, model)
        self.language = language
        self._mlx_whisper = None

    def _ensure_loaded(self):
        """Lazy load mlx_whisper module."""
        if self._mlx_whisper is None:
            try:
                import mlx_whisper
                self._mlx_whisper = mlx_whisper
            except ImportError:
                raise ImportError(
                    "mlx-whisper not found. Install with: pip install mlx-whisper"
                )

    def transcribe(
        self,
        audio_path: Path,
        word_timestamps: bool = True,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file (WAV recommended, 16kHz mono)
            word_timestamps: Whether to include word-level timestamps
            verbose: Print progress information

        Returns:
            TranscriptionResult with segments and word timings
        """
        self._ensure_loaded()
        audio_path = Path(audio_path)

        if verbose:
            print(f"Transcribing with model: {self.model_path}")

        # Build transcription options
        options = {
            "path_or_hf_repo": self.model_path,
            "word_timestamps": word_timestamps,
            "verbose": verbose,
        }

        if self.language:
            options["language"] = self.language

        # Run transcription
        result = self._mlx_whisper.transcribe(str(audio_path), **options)

        # Parse results into structured format
        segments = []
        for seg in result.get("segments", []):
            words = []
            for word_data in seg.get("words", []):
                words.append(Word(
                    text=word_data.get("word", "").strip(),
                    start=word_data.get("start", 0.0),
                    end=word_data.get("end", 0.0),
                    confidence=word_data.get("probability", 1.0),
                ))

            segments.append(Segment(
                text=seg.get("text", "").strip(),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                words=words,
            ))

        # Calculate duration from last segment
        duration = segments[-1].end if segments else 0.0

        return TranscriptionResult(
            text=result.get("text", "").strip(),
            segments=segments,
            language=result.get("language", "unknown"),
            duration=duration,
        )

    def transcribe_with_vad(
        self,
        audio_path: Path,
        word_timestamps: bool = True,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe with Voice Activity Detection to reduce hallucinations.

        This uses the diarization VAD results to mask non-speech regions.
        For now, delegates to standard transcription.
        """
        # TODO: Integrate with Silero VAD for hallucination reduction
        return self.transcribe(audio_path, word_timestamps, verbose)

    @classmethod
    def list_models(cls) -> List[str]:
        """List available model names."""
        return list(cls.MODELS.keys())
