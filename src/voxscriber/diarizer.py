"""
Pyannote Speaker Diarization Module

Uses pyannote.audio 3.1 for state-of-the-art speaker diarization
with MPS acceleration on Apple Silicon.

NOTE: Pyannote models require a one-time download from Hugging Face.
You need to:
1. Create a Hugging Face account
2. Accept the model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
3. Get your token from: https://huggingface.co/settings/tokens
4. Set HF_TOKEN environment variable or pass token to Diarizer

After initial download, all processing happens 100% locally.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class SpeakerSegment:
    """Represents a speaker segment."""
    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, start: float, end: float) -> float:
        """Calculate overlap duration with a time range."""
        overlap_start = max(self.start, start)
        overlap_end = min(self.end, end)
        return max(0, overlap_end - overlap_start)


@dataclass
class DiarizationResult:
    """Complete diarization result."""
    segments: List[SpeakerSegment]
    num_speakers: int
    speaker_labels: List[str]

    def get_speaker_at(self, time: float) -> Optional[str]:
        """Get the speaker at a specific time."""
        for seg in self.segments:
            if seg.start <= time < seg.end:
                return seg.speaker
        return None

    def get_dominant_speaker(self, start: float, end: float) -> Optional[str]:
        """Get the speaker with most overlap in a time range."""
        speaker_overlaps = {}
        for seg in self.segments:
            overlap = seg.overlaps(start, end)
            if overlap > 0:
                speaker_overlaps[seg.speaker] = speaker_overlaps.get(seg.speaker, 0) + overlap

        if not speaker_overlaps:
            return None
        return max(speaker_overlaps, key=speaker_overlaps.get)

    def to_dict(self) -> dict:
        return {
            "segments": [
                {"speaker": s.speaker, "start": s.start, "end": s.end}
                for s in self.segments
            ],
            "num_speakers": self.num_speakers,
            "speaker_labels": self.speaker_labels,
        }


class Diarizer:
    """Pyannote-based speaker diarizer optimized for Apple Silicon."""

    MODEL_ID = "pyannote/speaker-diarization-3.1"

    def __init__(
        self,
        hf_token: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        device: str = "mps",
    ):
        """
        Initialize diarizer.

        Args:
            hf_token: Hugging Face token. If None, uses HF_TOKEN env var.
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            device: Device to use ("mps" for Apple Silicon, "cpu" for fallback)
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.device = device
        self._pipeline = None

    def _ensure_loaded(self):
        """Lazy load the diarization pipeline."""
        if self._pipeline is not None:
            return

        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError(
                "pyannote.audio not found. Install with: pip install pyannote.audio"
            )

        if not self.hf_token:
            raise ValueError(
                "Hugging Face token required for pyannote models.\n"
                "1. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "2. Get token at: https://huggingface.co/settings/tokens\n"
                "3. Set HF_TOKEN environment variable or pass token to Diarizer"
            )

        # Load pipeline
        # pyannote.audio 3.1 uses use_auth_token, not token
        self._pipeline = Pipeline.from_pretrained(
            self.MODEL_ID,
            use_auth_token=self.hf_token,
        )

        # Move to appropriate device
        if self.device == "mps" and torch.backends.mps.is_available():
            self._pipeline.to(torch.device("mps"))
        elif self.device == "cuda" and torch.cuda.is_available():
            self._pipeline.to(torch.device("cuda"))
        # else: stays on CPU

    def diarize(
        self,
        audio_path: Path,
        verbose: bool = False,
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file (16kHz mono WAV recommended)
            verbose: Print progress information

        Returns:
            DiarizationResult with speaker segments
        """
        self._ensure_loaded()
        audio_path = Path(audio_path)

        if verbose:
            print(f"Diarizing: {audio_path}")

        # Build diarization options
        options = {}
        if self.num_speakers is not None:
            options["num_speakers"] = self.num_speakers
        if self.min_speakers is not None:
            options["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            options["max_speakers"] = self.max_speakers

        # Run diarization
        result = self._pipeline(str(audio_path), **options)

        # Pyannote 4.x returns DiarizeOutput, extract the Annotation
        if hasattr(result, 'speaker_diarization'):
            diarization = result.speaker_diarization
        else:
            # Fallback for older versions
            diarization = result

        # Parse results
        segments = []
        speaker_set = set()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                speaker=speaker,
                start=turn.start,
                end=turn.end,
            ))
            speaker_set.add(speaker)

        # Sort by start time
        segments.sort(key=lambda s: s.start)

        # Create readable speaker labels
        speaker_labels = sorted(list(speaker_set))

        return DiarizationResult(
            segments=segments,
            num_speakers=len(speaker_labels),
            speaker_labels=speaker_labels,
        )

    def diarize_with_embeddings(
        self,
        audio_path: Path,
        reference_embeddings: Optional[dict] = None,
        verbose: bool = False,
    ) -> DiarizationResult:
        """
        Diarize with optional speaker embeddings for identification.

        This allows matching detected speakers to known identities
        if reference embeddings are provided.

        Args:
            audio_path: Path to audio file
            reference_embeddings: Dict mapping speaker names to embeddings
            verbose: Print progress

        Returns:
            DiarizationResult with identified speakers
        """
        # For now, use standard diarization
        # TODO: Implement speaker identification with reference embeddings
        result = self.diarize(audio_path, verbose)

        if reference_embeddings:
            # Would match embeddings here
            pass

        return result

    @staticmethod
    def check_mps_available() -> bool:
        """Check if MPS (Metal Performance Shaders) is available."""
        try:
            import torch
            return torch.backends.mps.is_available()
        except Exception:
            return False
