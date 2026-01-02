"""
VoxScriber Pipeline

Orchestrates the complete speaker diarization workflow:
1. Audio preprocessing
2. Parallel transcription and diarization
3. Alignment
4. Output formatting
"""

import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from .preprocessor import AudioPreprocessor
from .transcriber import Transcriber, TranscriptionResult
from .diarizer import Diarizer, DiarizationResult
from .aligner import Aligner, AlignedTranscript
from .formatters import OutputFormatter, TranscriptPrinter


@dataclass
class PipelineConfig:
    """Configuration for the diarization pipeline."""
    # Transcription settings
    whisper_model: str = "large-v3-turbo"
    language: Optional[str] = None

    # Diarization settings
    hf_token: Optional[str] = None
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    device: str = "mps"

    # Alignment settings
    merge_same_speaker: bool = True
    merge_gap_threshold: float = 1.5

    # Output settings
    speaker_names: Optional[Dict[str, str]] = None

    # Processing
    parallel: bool = True
    verbose: bool = True
    cache_dir: Optional[Path] = None


class DiarizationPipeline:
    """
    Complete speaker diarization pipeline.

    Combines MLX Whisper transcription with Pyannote diarization
    for professional-grade speaker-attributed transcripts.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or PipelineConfig()
        self._setup_components()

    def _setup_components(self):
        """Initialize pipeline components."""
        self.preprocessor = AudioPreprocessor(
            cache_dir=self.config.cache_dir
        )

        self.transcriber = Transcriber(
            model=self.config.whisper_model,
            language=self.config.language,
        )

        self.diarizer = Diarizer(
            hf_token=self.config.hf_token or os.environ.get("HF_TOKEN"),
            num_speakers=self.config.num_speakers,
            min_speakers=self.config.min_speakers,
            max_speakers=self.config.max_speakers,
            device=self.config.device,
        )

        self.aligner = Aligner(
            merge_same_speaker=self.config.merge_same_speaker,
            merge_gap_threshold=self.config.merge_gap_threshold,
        )

        self.formatter = OutputFormatter(
            speaker_names=self.config.speaker_names
        )

        self.printer = TranscriptPrinter()

    def _log(self, message: str):
        """Print log message if verbose."""
        if self.config.verbose:
            print(f"[VoxScriber] {message}")

    def process(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None,
        output_formats: List[str] = ["json", "txt", "srt"],
    ) -> AlignedTranscript:
        """
        Process audio file through complete pipeline.

        Args:
            audio_path: Path to input audio file
            output_dir: Directory for output files. If None, uses audio file directory.
            output_formats: List of output formats to generate

        Returns:
            AlignedTranscript with speaker-attributed segments
        """
        audio_path = Path(audio_path)
        output_dir = output_dir or audio_path.parent
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_start = time.time()

        # Step 1: Preprocess audio
        # Whisper needs 16kHz mono WAV
        # Diarization needs mono WAV (original sample rate is fine, mono is crucial)
        self._log(f"Preprocessing: {audio_path.name}")
        preprocess_start = time.time()
        whisper_audio = self.preprocessor.process(audio_path)
        diarize_audio = self.preprocessor.process_for_diarization(audio_path)
        duration = self.preprocessor.get_duration(audio_path)
        self._log(f"Audio duration: {duration:.1f}s (preprocessed in {time.time() - preprocess_start:.1f}s)")

        # Step 2: Run transcription and diarization
        # Transcription uses preprocessed audio (16kHz mono for Whisper)
        # Diarization uses mono audio (original sample rate preserves speaker characteristics)
        if self.config.parallel:
            transcription, diarization = self._process_parallel(whisper_audio, diarize_audio)
        else:
            transcription, diarization = self._process_sequential(whisper_audio, diarize_audio)

        # Step 3: Align results
        self._log("Aligning transcription with speakers...")
        align_start = time.time()
        aligned = self.aligner.align(transcription, diarization)
        self._log(f"Alignment complete ({time.time() - align_start:.1f}s)")

        # Step 4: Save outputs
        base_name = audio_path.stem
        for fmt in output_formats:
            output_path = output_dir / f"{base_name}.{fmt}"
            self.formatter.save(aligned, output_path)
            self._log(f"Saved: {output_path}")

        total_time = time.time() - total_start
        rtf = total_time / duration if duration > 0 else 0
        self._log(f"Complete! Total time: {total_time:.1f}s (RTF: {rtf:.2f}x)")
        self._log(f"Detected {len(aligned.speakers)} speakers: {', '.join(aligned.speakers)}")

        return aligned

    def _process_parallel(
        self,
        whisper_audio_path: Path,
        diarize_audio_path: Path,
    ) -> tuple[TranscriptionResult, DiarizationResult]:
        """Run transcription and diarization in parallel.

        Args:
            whisper_audio_path: Preprocessed audio (16kHz mono) for Whisper
            diarize_audio_path: Original audio for diarization (preserves speaker characteristics)
        """
        self._log("Running transcription and diarization in parallel...")

        transcription = None
        diarization = None
        transcription_time = 0
        diarization_time = 0

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks with appropriate audio paths
            future_transcribe = executor.submit(
                self._timed_transcribe, whisper_audio_path
            )
            future_diarize = executor.submit(
                self._timed_diarize, diarize_audio_path
            )

            # Collect results
            for future in as_completed([future_transcribe, future_diarize]):
                result, elapsed, task_name = future.result()
                if task_name == "transcribe":
                    transcription = result
                    transcription_time = elapsed
                else:
                    diarization = result
                    diarization_time = elapsed

        self._log(f"Transcription: {transcription_time:.1f}s | Diarization: {diarization_time:.1f}s")
        return transcription, diarization

    def _process_sequential(
        self,
        whisper_audio_path: Path,
        diarize_audio_path: Path,
    ) -> tuple[TranscriptionResult, DiarizationResult]:
        """Run transcription and diarization sequentially.

        Args:
            whisper_audio_path: Preprocessed audio (16kHz mono) for Whisper
            diarize_audio_path: Original audio for diarization (preserves speaker characteristics)
        """
        self._log("Transcribing...")
        transcription, t_time, _ = self._timed_transcribe(whisper_audio_path)
        self._log(f"Transcription complete ({t_time:.1f}s)")

        self._log("Diarizing...")
        diarization, d_time, _ = self._timed_diarize(diarize_audio_path)
        self._log(f"Diarization complete ({d_time:.1f}s)")

        return transcription, diarization

    def _timed_transcribe(self, audio_path: Path) -> tuple:
        """Transcribe with timing."""
        start = time.time()
        result = self.transcriber.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False,
        )
        return result, time.time() - start, "transcribe"

    def _timed_diarize(self, audio_path: Path) -> tuple:
        """Diarize with timing."""
        start = time.time()
        result = self.diarizer.diarize(
            audio_path,
            verbose=False,
        )
        return result, time.time() - start, "diarize"

    def print_transcript(
        self,
        transcript: AlignedTranscript,
        speaker_names: Optional[Dict[str, str]] = None,
    ):
        """Print transcript to console with colors."""
        names = speaker_names or self.config.speaker_names
        self.printer.print(transcript, speaker_names=names)

    @classmethod
    def create_simple(
        cls,
        hf_token: Optional[str] = None,
        num_speakers: Optional[int] = None,
        language: Optional[str] = None,
    ) -> "DiarizationPipeline":
        """
        Create pipeline with simple configuration.

        Args:
            hf_token: Hugging Face token
            num_speakers: Number of speakers if known
            language: Force specific language

        Returns:
            Configured pipeline
        """
        config = PipelineConfig(
            hf_token=hf_token,
            num_speakers=num_speakers,
            language=language,
        )
        return cls(config)
