"""
Transcriber Module

Supports three backends:
- Parakeet TDT v3 via parakeet-mlx (Apple Silicon, macOS) - default, fastest, 25 languages
- MLX Whisper (Apple Silicon, macOS)
- faster-whisper (Linux/CUDA/CPU) - CTranslate2-based, works everywhere

Backend is derived from the model name and platform, or can be forced via `backend`.
"""

import gc
import platform
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


PARAKEET_MODELS = {
    "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
}

# Chunking keeps Parakeet's attention memory flat on long recordings.
PARAKEET_CHUNK_SECONDS = 60.0
PARAKEET_OVERLAP_SECONDS = 15.0
# One encoder frame. TDT can emit zero-duration tokens; the aligner drops zero-length words.
PARAKEET_MIN_WORD_SECONDS = 0.08


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _detect_backend(model: str) -> str:
    """Pick the backend for a model name on the current platform."""
    if model in PARAKEET_MODELS:
        if not is_apple_silicon():
            raise ValueError(
                f"Model '{model}' needs Apple Silicon (MLX). Use a Whisper model on this platform."
            )
        return "parakeet"
    if is_apple_silicon():
        return "mlx"
    return "faster-whisper"


def _tokens_to_words(tokens) -> List[Word]:
    """Merge Parakeet sub-word tokens into words; a leading space starts a new word."""
    words: List[Word] = []
    text, start, end, confs = "", 0.0, 0.0, []

    def flush():
        words.append(Word(
            text=text.strip(),
            start=start,
            end=max(end, start + PARAKEET_MIN_WORD_SECONDS),
            confidence=min(confs),
        ))

    for tok in tokens:
        if tok.text.startswith(" ") and text.strip():
            flush()
            text, confs = "", []
        if not text.strip():
            start = tok.start
        text += tok.text
        end = tok.end
        confs.append(tok.confidence)
    if text.strip():
        flush()
    return words


class Transcriber:
    """Speech-to-text transcriber with pluggable backend."""

    # MLX model paths (Apple Silicon only)
    MLX_MODELS = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        "large-4bit": "mlx-community/whisper-large-v3-mlx-4bit",
        "large-8bit": "mlx-community/whisper-large-v3-mlx-8bit",
    }

    # faster-whisper model names (CTranslate2)
    FASTER_MODELS = {
        "tiny": "tiny",
        "base": "base",
        "small": "small",
        "medium": "medium",
        "large": "large-v3",
        "large-v3-turbo": "large-v3-turbo",
        "large-4bit": "large-v3",
        "large-8bit": "large-v3",
    }

    def __init__(
        self,
        model: str = "large-v3-turbo",
        language: Optional[str] = None,
        backend: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize transcriber.

        Args:
            model: Model name (see PARAKEET_MODELS / MLX_MODELS / FASTER_MODELS) or HF repo path
            language: Force specific language (e.g., "en", "es"). None for auto-detect.
                      Parakeet always auto-detects, so this is ignored for it.
            backend: "parakeet", "mlx" or "faster-whisper". None for auto-detect.
            device: Device for faster-whisper ("auto", "cuda", "cpu"). Ignored for MLX.
        """
        self.model_name = model
        self.language = language
        self.backend = backend or _detect_backend(model)
        self.device = device
        self._engine = None

    def _ensure_loaded(self):
        """Lazy load the transcription engine."""
        if self._engine is not None:
            return

        if self.backend == "parakeet":
            try:
                from parakeet_mlx import from_pretrained
            except ImportError:
                raise ImportError(
                    "parakeet-mlx not found. Install with: pip install parakeet-mlx\n"
                    "Note: Parakeet (MLX) only works on Apple Silicon Macs."
                )
            self._engine = from_pretrained(
                PARAKEET_MODELS.get(self.model_name, self.model_name)
            )
        elif self.backend == "mlx":
            try:
                import mlx_whisper
                self._engine = mlx_whisper
            except ImportError:
                raise ImportError(
                    "mlx-whisper not found. Install with: pip install mlx-whisper\n"
                    "Note: MLX only works on Apple Silicon Macs."
                )
        else:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError(
                    "faster-whisper not found. Install with: pip install faster-whisper\n"
                    "For GPU support, ensure CUDA 12 and cuDNN 9 are installed."
                )

            model_id = self.FASTER_MODELS.get(self.model_name, self.model_name)

            # Determine device and compute type
            device = self.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            compute_type = "float16" if device == "cuda" else "int8"

            # Quantized model variants
            if self.model_name == "large-4bit":
                compute_type = "int8"
            elif self.model_name == "large-8bit":
                compute_type = "int8"

            self._engine = WhisperModel(model_id, device=device, compute_type=compute_type)

    def _transcribe_parakeet(self, audio_path: Path, word_timestamps: bool):
        """Transcribe using Parakeet TDT (parakeet-mlx) backend."""
        result = self._engine.transcribe(
            str(audio_path),
            chunk_duration=PARAKEET_CHUNK_SECONDS,
            overlap_duration=PARAKEET_OVERLAP_SECONDS,
        )

        segments = []
        for sentence in result.sentences:
            words = _tokens_to_words(sentence.tokens) if word_timestamps else []
            segments.append(Segment(
                text=sentence.text.strip(),
                start=sentence.start,
                end=sentence.end,
                words=words,
            ))

        duration = segments[-1].end if segments else 0.0
        return TranscriptionResult(
            text=result.text,
            segments=segments,
            language=self.language or "auto",
            duration=duration,
        )

    def _transcribe_mlx(self, audio_path: Path, word_timestamps: bool, verbose: bool):
        """Transcribe using MLX Whisper backend."""
        model_path = self.MLX_MODELS.get(self.model_name, self.model_name)

        options = {
            "path_or_hf_repo": model_path,
            "word_timestamps": word_timestamps,
            "verbose": verbose,
        }
        if self.language:
            options["language"] = self.language

        result = self._engine.transcribe(str(audio_path), **options)

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

        duration = segments[-1].end if segments else 0.0
        return TranscriptionResult(
            text=result.get("text", "").strip(),
            segments=segments,
            language=result.get("language", "unknown"),
            duration=duration,
        )

    def _transcribe_faster(self, audio_path: Path, word_timestamps: bool, verbose: bool):
        """Transcribe using faster-whisper backend."""
        options = {"word_timestamps": word_timestamps, "beam_size": 5}
        if self.language:
            options["language"] = self.language

        raw_segments, info = self._engine.transcribe(str(audio_path), **options)

        segments = []
        full_text_parts = []
        for seg in raw_segments:
            words = []
            if word_timestamps and seg.words:
                for w in seg.words:
                    words.append(Word(
                        text=w.word.strip(),
                        start=w.start,
                        end=w.end,
                        confidence=w.probability,
                    ))
            segments.append(Segment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                words=words,
            ))
            full_text_parts.append(seg.text.strip())

        duration = segments[-1].end if segments else 0.0
        return TranscriptionResult(
            text=" ".join(full_text_parts),
            segments=segments,
            language=info.language,
            duration=duration,
        )

    def transcribe(
        self,
        audio_path: Path,
        word_timestamps: bool = True,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe audio file using the configured backend.

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
            print(f"Transcribing with backend: {self.backend}")

        if self.backend == "parakeet":
            return self._transcribe_parakeet(audio_path, word_timestamps)
        if self.backend == "mlx":
            return self._transcribe_mlx(audio_path, word_timestamps, verbose)
        return self._transcribe_faster(audio_path, word_timestamps, verbose)

    def unload(self):
        """Release the model and return its memory to the OS before the next stage runs."""
        if self.backend == "mlx" and self._engine is not None:
            from mlx_whisper.transcribe import ModelHolder
            ModelHolder.model = None
            ModelHolder.model_path = None
        self._engine = None
        gc.collect()
        if self.backend in ("parakeet", "mlx"):
            try:
                import mlx.core as mx
                mx.clear_cache()
            except ImportError:
                pass

    def transcribe_with_vad(
        self,
        audio_path: Path,
        word_timestamps: bool = True,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe with Voice Activity Detection to reduce hallucinations.

        faster-whisper has built-in Silero VAD support.
        Other backends delegate to standard transcription.
        """
        if self.backend == "faster-whisper":
            self._ensure_loaded()
            options = {
                "word_timestamps": word_timestamps,
                "beam_size": 5,
                "vad_filter": True,
            }
            if self.language:
                options["language"] = self.language

            raw_segments, info = self._engine.transcribe(str(audio_path), **options)

            segments = []
            full_text_parts = []
            for seg in raw_segments:
                words = []
                if word_timestamps and seg.words:
                    for w in seg.words:
                        words.append(Word(
                            text=w.word.strip(),
                            start=w.start,
                            end=w.end,
                            confidence=w.probability,
                        ))
                segments.append(Segment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    words=words,
                ))
                full_text_parts.append(seg.text.strip())

            duration = segments[-1].end if segments else 0.0
            return TranscriptionResult(
                text=" ".join(full_text_parts),
                segments=segments,
                language=info.language,
                duration=duration,
            )

        return self.transcribe(audio_path, word_timestamps, verbose)

    @classmethod
    def list_models(cls) -> List[str]:
        """List available model names."""
        return list(PARAKEET_MODELS) + list(cls.MLX_MODELS)
