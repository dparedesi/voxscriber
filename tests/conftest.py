import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to python path to allow imports without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_transcript():
    """Return a sample transcript structure similar to MLX Whisper output."""
    return {
        "text": " Hello world. This is speaker A. And this is speaker B.",
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 2.0,
                "text": " Hello world.",
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": -0.5,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.9},
                    {"word": "world.", "start": 0.5, "end": 1.0, "probability": 0.9},
                ]
            },
            {
                "id": 1,
                "seek": 0,
                "start": 2.0,
                "end": 4.0,
                "text": " This is speaker A.",
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": -0.5,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
                "words": [
                    {"word": "This", "start": 2.0, "end": 2.5, "probability": 0.9},
                    {"word": "is", "start": 2.5, "end": 3.0, "probability": 0.9},
                    {"word": "speaker", "start": 3.0, "end": 3.5, "probability": 0.9},
                    {"word": "A.", "start": 3.5, "end": 4.0, "probability": 0.9},
                ]
            },
            {
                "id": 2,
                "seek": 0,
                "start": 4.0,
                "end": 6.0,
                "text": " And this is speaker B.",
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": -0.5,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
                "words": [
                    {"word": "And", "start": 4.0, "end": 4.5, "probability": 0.9},
                    {"word": "this", "start": 4.5, "end": 5.0, "probability": 0.9},
                    {"word": "is", "start": 5.0, "end": 5.5, "probability": 0.9},
                    {"word": "speaker", "start": 5.5, "end": 5.8, "probability": 0.9},
                    {"word": "B.", "start": 5.8, "end": 6.0, "probability": 0.9},
                ]
            }
        ]
    }

@pytest.fixture
def sample_diarization_segments():
    """Return a sample list of diarization segments."""
    # Format: (segment, track, label) where segment has start/end
    # Mocking the pyannote.core.Segment object structure
    class MockSegment:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    return [
        (MockSegment(0.0, 1.5), None, "SPEAKER_00"),
        (MockSegment(2.0, 4.0), None, "SPEAKER_00"),
        (MockSegment(4.0, 6.0), None, "SPEAKER_01"),
    ]

@pytest.fixture
def mock_audio_file():
    """Create a temporary dummy audio file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Just create an empty file, we'll mock the processing
        f.write(b"RIFF" + b"\x00" * 32) # Minimal header
        path = f.name

    yield path

    # Cleanup
    if os.path.exists(path):
        os.unlink(path)

@pytest.fixture
def mock_pipeline_components(mocker):
    """Mock out the heavy ML components."""
    # Mock MLX Whisper
    mocker.patch("voxscriber.transcriber.transcribe", return_value={"text": "mock", "segments": []})

    # Mock Pyannote
    mock_pipeline = MagicMock()
    mock_pipeline.return_value.itertracks.return_value = []
    mocker.patch("pyannote.audio.Pipeline.from_pretrained", return_value=mock_pipeline)

    return mock_pipeline
