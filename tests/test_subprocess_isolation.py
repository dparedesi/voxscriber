"""Integration tests for subprocess isolation of MLX Whisper and PyTorch/pyannote.

These tests verify that transcription and diarization run in separate
subprocesses, preventing the OpenMP runtime and Metal GPU conflicts that
cause segfaults when both libraries are loaded in the same process.

If the subprocess isolation is removed (e.g. reverted to ThreadPoolExecutor),
these tests will segfault instead of passing.

Run with: pytest tests/test_subprocess_isolation.py -m integration
"""

import math
import os
import struct
import wave

import pytest

_missing = []
try:
    import mlx_whisper  # noqa: F401
except ImportError:
    _missing.append("mlx_whisper")
try:
    import torch  # noqa: F401
    import pyannote.audio  # noqa: F401
except ImportError:
    _missing.append("pyannote.audio")

skip_reason = None
if _missing:
    skip_reason = f"requires {', '.join(_missing)}"
elif not os.environ.get("HF_TOKEN"):
    try:
        from huggingface_hub import get_token
        if not get_token():
            skip_reason = "requires HF_TOKEN or huggingface-cli login"
    except Exception:
        skip_reason = "requires HF_TOKEN or huggingface-cli login"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(skip_reason is not None, reason=skip_reason or ""),
]


def _generate_wav(path, duration=3.0, sample_rate=16000, frequency=440.0):
    """Generate a mono 16kHz 16-bit WAV with a sine tone."""
    n_samples = int(duration * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = b"".join(
            struct.pack("<h", int(32767 * 0.5 * math.sin(2 * math.pi * frequency * i / sample_rate)))
            for i in range(n_samples)
        )
        wf.writeframes(frames)


def test_parallel_subprocess_isolation(tmp_path):
    """Full pipeline in parallel mode completes without segfault.

    Both MLX Whisper (transcription) and PyTorch/pyannote (diarization)
    run in separate spawned subprocesses. If isolation is removed, the
    conflicting OpenMP runtimes will crash the process.
    """
    from voxscriber.pipeline import DiarizationPipeline, PipelineConfig

    wav_path = tmp_path / "tone.wav"
    _generate_wav(wav_path, duration=3.0)

    config = PipelineConfig(
        whisper_model="tiny",
        language="en",
        device="mps",
        parallel=True,
        verbose=False,
    )
    pipeline = DiarizationPipeline(config)
    # The test passes if process() completes without segfault.
    # A synthetic tone has no speech, so we don't assert on transcript content.
    result = pipeline.process(wav_path, output_formats=[])

    assert result is not None


def test_sequential_subprocess_isolation(tmp_path):
    """Full pipeline in sequential mode completes without segfault.

    Same isolation guarantee as parallel, but workers run one at a time.
    """
    from voxscriber.pipeline import DiarizationPipeline, PipelineConfig

    wav_path = tmp_path / "tone.wav"
    _generate_wav(wav_path, duration=3.0)

    config = PipelineConfig(
        whisper_model="tiny",
        language="en",
        device="mps",
        parallel=False,
        verbose=False,
    )
    pipeline = DiarizationPipeline(config)
    # The test passes if process() completes without segfault.
    result = pipeline.process(wav_path, output_formats=[])

    assert result is not None
