from pathlib import Path
from unittest.mock import patch

import pytest

from voxscriber.aligner import AlignedTranscript, AttributedSegment, AttributedWord
from voxscriber.diarizer import DiarizationResult, SpeakerSegment
from voxscriber.pipeline import DiarizationPipeline, PipelineConfig
from voxscriber.transcriber import Segment, TranscriptionResult, Word

# --- Fixtures ---

@pytest.fixture
def mock_transcription_result():
    return TranscriptionResult(
        text="Hello world",
        segments=[
            Segment(
                text="Hello world",
                start=0.0,
                end=2.0,
                words=[
                    Word(text="Hello", start=0.0, end=1.0),
                    Word(text="world", start=1.0, end=2.0)
                ]
            )
        ],
        language="en",
        duration=2.0
    )

@pytest.fixture
def mock_diarization_result():
    return DiarizationResult(
        segments=[
            SpeakerSegment(speaker="SPEAKER_01", start=0.0, end=2.0)
        ],
        num_speakers=1,
        speaker_labels=["SPEAKER_01"]
    )

@pytest.fixture
def mock_aligned_transcript():
    return AlignedTranscript(
        segments=[
            AttributedSegment(
                speaker="SPEAKER_01",
                text="Hello world",
                start=0.0,
                end=2.0,
                words=[
                    AttributedWord(text="Hello", start=0.0, end=1.0, speaker="SPEAKER_01"),
                    AttributedWord(text="world", start=1.0, end=2.0, speaker="SPEAKER_01")
                ]
            )
        ],
        speakers=["SPEAKER_01"],
        duration=2.0,
        language="en"
    )

@pytest.fixture
def mock_components():
    with patch("voxscriber.pipeline.AudioPreprocessor") as MockPreprocessor, \
         patch("voxscriber.pipeline.Transcriber") as MockTranscriber, \
         patch("voxscriber.pipeline.Diarizer") as MockDiarizer, \
         patch("voxscriber.pipeline.Aligner") as MockAligner, \
         patch("voxscriber.pipeline.OutputFormatter") as MockFormatter, \
         patch("voxscriber.pipeline.TranscriptPrinter") as MockPrinter:

        # Setup Preprocessor
        preprocessor = MockPreprocessor.return_value
        preprocessor.process.return_value = Path("processed_whisper.wav")
        preprocessor.process_for_diarization.return_value = Path("processed_diarize.wav")
        preprocessor.get_duration.return_value = 10.0

        # Setup Transcriber
        transcriber = MockTranscriber.return_value

        # Setup Diarizer
        diarizer = MockDiarizer.return_value

        # Setup Aligner
        aligner = MockAligner.return_value

        yield {
            "preprocessor": preprocessor,
            "transcriber": transcriber,
            "diarizer": diarizer,
            "aligner": aligner,
            "formatter": MockFormatter.return_value,
            "printer": MockPrinter.return_value,
            "MockPreprocessor": MockPreprocessor,
            "MockTranscriber": MockTranscriber,
            "MockDiarizer": MockDiarizer,
            "MockAligner": MockAligner,
        }

# --- Tests for PipelineConfig ---

def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.whisper_model == "large-v3-turbo"
    assert config.parallel is True
    assert config.device == "mps"
    assert config.language is None
    assert config.hf_token is None

def test_pipeline_config_custom():
    config = PipelineConfig(
        whisper_model="tiny",
        num_speakers=2,
        parallel=False,
        hf_token="test_token"
    )
    assert config.whisper_model == "tiny"
    assert config.num_speakers == 2
    assert config.parallel is False
    assert config.hf_token == "test_token"

# --- Tests for DiarizationPipeline Initialization ---

def test_pipeline_init_defaults(mock_components):
    pipeline = DiarizationPipeline()
    assert isinstance(pipeline.config, PipelineConfig)

    # Verify components initialized with defaults
    mock_components["MockTranscriber"].assert_called_with(
        model="large-v3-turbo",
        language=None
    )
    mock_components["MockDiarizer"].assert_called_with(
        hf_token=None,  # Or os.environ.get("HF_TOKEN")
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
        device="mps"
    )

def test_pipeline_init_custom_config(mock_components):
    config = PipelineConfig(
        whisper_model="medium",
        language="es",
        hf_token="secret",
        num_speakers=3,
        device="cpu",
        cache_dir=Path("/tmp/cache")
    )
    DiarizationPipeline(config)  # noqa: instantiation triggers component creation

    mock_components["MockTranscriber"].assert_called_with(
        model="medium",
        language="es"
    )
    mock_components["MockDiarizer"].assert_called_with(
        hf_token="secret",
        num_speakers=3,
        min_speakers=None,
        max_speakers=None,
        device="cpu"
    )
    mock_components["MockPreprocessor"].assert_called_with(
        cache_dir=Path("/tmp/cache")
    )

def test_create_simple_factory(mock_components):
    pipeline = DiarizationPipeline.create_simple(
        hf_token="token123",
        num_speakers=4,
        language="fr"
    )

    assert pipeline.config.hf_token == "token123"
    assert pipeline.config.num_speakers == 4
    assert pipeline.config.language == "fr"
    assert pipeline.config.parallel is True  # default

# --- Tests for Processing ---

def test_process_parallel_flow(mock_components, mock_transcription_result, mock_diarization_result, mock_aligned_transcript):
    """Test the complete parallel processing flow."""
    # Setup mocks
    mock_components["transcriber"].transcribe.return_value = mock_transcription_result
    mock_components["diarizer"].diarize.return_value = mock_diarization_result
    mock_components["aligner"].align.return_value = mock_aligned_transcript

    pipeline = DiarizationPipeline(PipelineConfig(parallel=True, verbose=False))

    # Run process
    input_path = Path("test.m4a")
    result = pipeline.process(input_path, output_formats=["txt"])

    # Verify results
    assert result == mock_aligned_transcript

    # Verify preprocessing calls
    mock_components["preprocessor"].process.assert_called_once_with(input_path)
    mock_components["preprocessor"].process_for_diarization.assert_called_once_with(input_path)
    mock_components["preprocessor"].get_duration.assert_called_once_with(input_path)

    # Verify parallel execution paths (passed the preprocessed paths)
    whisper_path = mock_components["preprocessor"].process.return_value
    diarize_path = mock_components["preprocessor"].process_for_diarization.return_value

    mock_components["transcriber"].transcribe.assert_called_once_with(
        whisper_path, word_timestamps=True, verbose=False
    )
    mock_components["diarizer"].diarize.assert_called_once_with(
        diarize_path, verbose=False
    )

    # Verify alignment
    mock_components["aligner"].align.assert_called_once_with(
        mock_transcription_result, mock_diarization_result
    )

    # Verify output
    mock_components["formatter"].save.assert_called_once()
    args, _ = mock_components["formatter"].save.call_args
    assert args[0] == mock_aligned_transcript
    assert str(args[1]).endswith(".txt")

def test_process_sequential_flow(mock_components, mock_transcription_result, mock_diarization_result, mock_aligned_transcript):
    """Test the sequential processing flow."""
    mock_components["transcriber"].transcribe.return_value = mock_transcription_result
    mock_components["diarizer"].diarize.return_value = mock_diarization_result
    mock_components["aligner"].align.return_value = mock_aligned_transcript

    pipeline = DiarizationPipeline(PipelineConfig(parallel=False, verbose=False))

    input_path = Path("test.m4a")
    pipeline.process(input_path, output_formats=[])

    whisper_path = mock_components["preprocessor"].process.return_value
    diarize_path = mock_components["preprocessor"].process_for_diarization.return_value

    # Check calls occurred (order is harder to verify strictly without side effects,
    # but sequential implementation calls them one by one)
    mock_components["transcriber"].transcribe.assert_called_once_with(
        whisper_path, word_timestamps=True, verbose=False
    )
    mock_components["diarizer"].diarize.assert_called_once_with(
        diarize_path, verbose=False
    )

def test_process_outputs(mock_components, mock_transcription_result, mock_diarization_result, mock_aligned_transcript):
    """Test multiple output formats."""
    mock_components["transcriber"].transcribe.return_value = mock_transcription_result
    mock_components["diarizer"].diarize.return_value = mock_diarization_result
    mock_components["aligner"].align.return_value = mock_aligned_transcript

    pipeline = DiarizationPipeline()

    with patch("pathlib.Path.mkdir"):  # Mock directory creation
        pipeline.process(
            Path("input/test.wav"),
            output_dir=Path("output"),
            output_formats=["json", "srt", "vtt"]
        )

    # Should have called save 3 times
    assert mock_components["formatter"].save.call_count == 3

    # Check extensions
    calls = mock_components["formatter"].save.call_args_list
    extensions = [str(call.args[1]).split('.')[-1] for call in calls]
    assert set(extensions) == {"json", "srt", "vtt"}

# --- Error Handling Tests ---

def test_transcription_failure(mock_components):
    """Test handling of transcription failure."""
    mock_components["transcriber"].transcribe.side_effect = Exception("Whisper error")

    pipeline = DiarizationPipeline(PipelineConfig(parallel=False))

    with pytest.raises(Exception) as excinfo:
        pipeline.process(Path("test.wav"))

    assert "Whisper error" in str(excinfo.value)

def test_diarization_failure(mock_components, mock_transcription_result):
    """Test handling of diarization failure."""
    mock_components["transcriber"].transcribe.return_value = mock_transcription_result
    mock_components["diarizer"].diarize.side_effect = Exception("Pyannote error")

    pipeline = DiarizationPipeline(PipelineConfig(parallel=False))

    with pytest.raises(Exception) as excinfo:
        pipeline.process(Path("test.wav"))

    assert "Pyannote error" in str(excinfo.value)

def test_parallel_execution_error(mock_components):
    """Test that errors propagate correctly from thread pool."""
    mock_components["transcriber"].transcribe.side_effect = RuntimeError("Thread error")

    pipeline = DiarizationPipeline(PipelineConfig(parallel=True))

    with pytest.raises(RuntimeError) as excinfo:
        pipeline.process(Path("test.wav"))

    assert "Thread error" in str(excinfo.value)

# --- Integration/Logic Tests ---

def test_custom_output_directory(mock_components, mock_transcription_result, mock_diarization_result, mock_aligned_transcript):
    """Test that output directory is respected and created."""
    mock_components["transcriber"].transcribe.return_value = mock_transcription_result
    mock_components["diarizer"].diarize.return_value = mock_diarization_result
    mock_components["aligner"].align.return_value = mock_aligned_transcript

    pipeline = DiarizationPipeline()
    output_dir = Path("custom_output")

    # We patch pathlib.Path.mkdir to verify it's called and avoid FS operations
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        pipeline.process(
            Path("test.wav"),
            output_dir=output_dir,
            output_formats=["json"]
        )

        mock_mkdir.assert_called_with(parents=True, exist_ok=True)

        # Verify save called with path inside custom_output
        # The formatter.save is called with (transcript, output_path)
        save_call_args = mock_components["formatter"].save.call_args
        saved_path = save_call_args[0][1]
        assert saved_path.parent == output_dir

def test_print_transcript(mock_components, mock_aligned_transcript):
    """Test print_transcript delegates to printer."""
    pipeline = DiarizationPipeline()

    custom_names = {"SPEAKER_01": "Alice"}
    pipeline.print_transcript(mock_aligned_transcript, speaker_names=custom_names)

    mock_components["printer"].print.assert_called_once_with(
        mock_aligned_transcript,
        speaker_names=custom_names
    )

def test_log_verbose(capsys, mock_components):
    """Test logging output."""
    # Verbose = True
    pipeline = DiarizationPipeline(PipelineConfig(verbose=True))
    pipeline._log("Test message")
    captured = capsys.readouterr()
    assert "[VoxScriber] Test message" in captured.out

    # Verbose = False
    pipeline = DiarizationPipeline(PipelineConfig(verbose=False))
    pipeline._log("Should not see this")
    captured = capsys.readouterr()
    assert "Should not see this" not in captured.out
