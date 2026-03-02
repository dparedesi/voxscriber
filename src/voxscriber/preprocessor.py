"""
Audio Preprocessor Module

Handles audio conversion to the format required by diarization models:
- 16kHz sample rate
- Mono channel
- WAV format (PCM 16-bit)

Uses ffmpeg CLI if available, falls back to PyAV (bundled with faster-whisper)
for environments without system ffmpeg.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


def _has_ffmpeg() -> bool:
    """Check if ffmpeg CLI is available."""
    return shutil.which("ffmpeg") is not None


def _has_ffprobe() -> bool:
    """Check if ffprobe CLI is available."""
    return shutil.which("ffprobe") is not None


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
        self._use_ffmpeg = _has_ffmpeg()

    def _get_audio_info(self, audio_path: Path) -> dict:
        """Get audio file information using ffprobe or PyAV."""
        if _has_ffprobe():
            return self._get_audio_info_ffprobe(audio_path)
        return self._get_audio_info_pyav(audio_path)

    def _get_audio_info_ffprobe(self, audio_path: Path) -> dict:
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

    def _get_audio_info_pyav(self, audio_path: Path) -> dict:
        """Get audio file information using PyAV."""
        import av

        container = av.open(str(audio_path))
        stream = container.streams.audio[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        info = {
            "streams": [{
                "codec_type": "audio",
                "codec_name": stream.codec_context.name,
                "sample_rate": str(stream.sample_rate),
                "channels": str(stream.channels),
            }],
            "format": {"duration": str(duration)},
        }
        container.close()
        return info

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

    def _convert_with_pyav(
        self, audio_path: Path, output_path: Path, target_sr: Optional[int], mono: bool
    ) -> Path:
        """Convert audio using PyAV (no system ffmpeg needed)."""
        import av

        container = av.open(str(audio_path))
        stream = container.streams.audio[0]
        src_sr = stream.sample_rate

        # Decode all audio frames
        frames = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            # arr shape: (channels, samples) for planar, (samples,) for packed
            if arr.ndim == 1:
                arr = arr[None, :]
            frames.append(arr)
        container.close()

        if not frames:
            raise RuntimeError(f"No audio frames decoded from {audio_path}")

        audio = np.concatenate(frames, axis=1).astype(np.float32)

        # Mix to mono if needed
        if mono and audio.shape[0] > 1:
            audio = audio.mean(axis=0, keepdims=True)

        # Resample if needed
        out_sr = target_sr or src_sr
        if out_sr != src_sr:
            from fractions import Fraction
            ratio = Fraction(out_sr, src_sr)
            new_length = int(audio.shape[1] * ratio)
            indices = np.linspace(0, audio.shape[1] - 1, new_length)
            audio = np.stack([np.interp(indices, np.arange(audio.shape[1]), ch) for ch in audio])

        # Write as 16-bit PCM WAV
        # soundfile expects (samples, channels)
        data = audio.T
        sf.write(str(output_path), data, out_sr, subtype="PCM_16")
        return output_path

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

        if self._use_ffmpeg:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(audio_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(self.REQUIRED_SAMPLE_RATE),
                "-ac", str(self.REQUIRED_CHANNELS),
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Audio conversion failed: {result.stderr}")
        else:
            self._convert_with_pyav(
                audio_path, output_path, target_sr=self.REQUIRED_SAMPLE_RATE, mono=True
            )

        return output_path

    def process_for_diarization(self, audio_path: Path, force: bool = False) -> Path:
        """
        Process audio file for diarization (mono only, keeps original sample rate).

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
                if channels == 1 and codec in ["pcm_s16le", "pcm_f32le"]:
                    return audio_path

        # Generate cache filename for diarization version
        cache_name = f"{audio_path.stem}_mono_diarize.wav"
        output_path = self.cache_dir / cache_name

        # Check if already processed
        if output_path.exists() and not force:
            return output_path

        if self._use_ffmpeg:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(audio_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Audio conversion for diarization failed: {result.stderr}")
        else:
            self._convert_with_pyav(audio_path, output_path, target_sr=None, mono=True)

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
