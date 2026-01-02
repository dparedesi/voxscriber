"""
Audio Preprocessor Module

Handles audio conversion to the format required by diarization models:
- 16kHz sample rate
- Mono channel
- WAV format (PCM 16-bit)
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import os


class AudioPreprocessor:
    """Preprocesses audio files for diarization pipeline."""

    REQUIRED_SAMPLE_RATE = 16000
    REQUIRED_CHANNELS = 1

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize preprocessor.

        Args:
            cache_dir: Directory to cache processed files. If None, uses temp directory.
        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "diarization_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_audio_info(self, audio_path: Path) -> dict:
        """Get audio file information using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        import json
        return json.loads(result.stdout)

    def _needs_conversion(self, audio_path: Path) -> bool:
        """Check if audio needs conversion."""
        info = self._get_audio_info(audio_path)

        for stream in info.get("streams", []):
            if stream.get("codec_type") == "audio":
                sample_rate = int(stream.get("sample_rate", 0))
                channels = int(stream.get("channels", 0))
                codec = stream.get("codec_name", "")

                if (sample_rate == self.REQUIRED_SAMPLE_RATE and
                    channels == self.REQUIRED_CHANNELS and
                    codec == "pcm_s16le"):
                    return False
        return True

    def process(self, audio_path: Path, force: bool = False) -> Path:
        """
        Process audio file for diarization.

        Args:
            audio_path: Path to input audio file
            force: Force reprocessing even if cached version exists

        Returns:
            Path to processed WAV file
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Generate cache filename based on input file
        cache_name = f"{audio_path.stem}_16khz_mono.wav"
        output_path = self.cache_dir / cache_name

        # Check if already processed
        if output_path.exists() and not force:
            # Verify the cached file is valid
            try:
                if not self._needs_conversion(output_path):
                    return output_path
            except Exception:
                pass  # Reprocess if cache validation fails

        # Convert using ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", str(self.REQUIRED_SAMPLE_RATE),  # 16kHz
            "-ac", str(self.REQUIRED_CHANNELS),  # Mono
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Audio conversion failed: {result.stderr}")

        return output_path

    def process_for_diarization(self, audio_path: Path, force: bool = False) -> Path:
        """
        Process audio file for diarization (mono only, keeps original sample rate).

        Pyannote works best with mono audio but can handle various sample rates.
        Multi-channel audio (e.g., 3.0 channel layout) can cause issues.

        Args:
            audio_path: Path to input audio file
            force: Force reprocessing even if cached version exists

        Returns:
            Path to processed WAV file (mono, original sample rate)
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Check if already mono WAV
        info = self._get_audio_info(audio_path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "audio":
                channels = int(stream.get("channels", 0))
                codec = stream.get("codec_name", "")
                # If already mono WAV, use as-is
                if channels == 1 and codec in ["pcm_s16le", "pcm_f32le"]:
                    return audio_path

        # Generate cache filename for diarization version
        cache_name = f"{audio_path.stem}_mono_diarize.wav"
        output_path = self.cache_dir / cache_name

        # Check if already processed
        if output_path.exists() and not force:
            return output_path

        # Convert to mono WAV, keeping original sample rate
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ac", "1",  # Mono (crucial for pyannote)
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Audio conversion for diarization failed: {result.stderr}")

        return output_path

    def get_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        info = self._get_audio_info(audio_path)
        return float(info.get("format", {}).get("duration", 0))

    def cleanup(self, audio_path: Optional[Path] = None):
        """
        Clean up cached files.

        Args:
            audio_path: Specific file to clean up. If None, cleans all cached files.
        """
        if audio_path:
            if audio_path.exists():
                os.remove(audio_path)
        else:
            for f in self.cache_dir.glob("*.wav"):
                os.remove(f)
