import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voxscriber.preprocessor import AudioPreprocessor


@pytest.fixture
def preprocessor(tmp_path):
    return AudioPreprocessor(cache_dir=tmp_path / "cache")

@pytest.fixture
def mock_audio_file(tmp_path):
    p = tmp_path / "test_audio.m4a"
    p.touch()
    return p

def test_init_creates_cache_dir(tmp_path):
    cache_dir = tmp_path / "custom_cache"
    AudioPreprocessor(cache_dir=cache_dir)
    assert cache_dir.exists()
    assert cache_dir.is_dir()

def test_get_audio_info_success(preprocessor, mock_audio_file):
    expected_info = {
        "streams": [{"codec_type": "audio", "sample_rate": "44100"}],
        "format": {"duration": "10.5"}
    }

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(expected_info),
            stderr=""
        )

        info = preprocessor._get_audio_info(mock_audio_file)
        assert info == expected_info

        # Verify call arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "ffprobe"
        assert str(mock_audio_file) in args

def test_get_audio_info_failure(preprocessor, mock_audio_file):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error reading file"
        )

        with pytest.raises(RuntimeError, match="ffprobe failed"):
            preprocessor._get_audio_info(mock_audio_file)

def test_needs_conversion_true(preprocessor, mock_audio_file):
    # Case 1: Wrong sample rate
    info_wrong_rate = {
        "streams": [{
            "codec_type": "audio",
            "sample_rate": "44100",
            "channels": 1,
            "codec_name": "pcm_s16le"
        }]
    }

    # Case 2: Wrong channels
    info_wrong_channels = {
        "streams": [{
            "codec_type": "audio",
            "sample_rate": "16000",
            "channels": 2,
            "codec_name": "pcm_s16le"
        }]
    }

    # Case 3: Wrong codec
    info_wrong_codec = {
        "streams": [{
            "codec_type": "audio",
            "sample_rate": "16000",
            "channels": 1,
            "codec_name": "aac"
        }]
    }

    for info in [info_wrong_rate, info_wrong_channels, info_wrong_codec]:
        with patch.object(preprocessor, "_get_audio_info", return_value=info):
            assert preprocessor._needs_conversion(mock_audio_file) is True

def test_needs_conversion_false(preprocessor, mock_audio_file):
    correct_info = {
        "streams": [{
            "codec_type": "audio",
            "sample_rate": "16000",
            "channels": 1,
            "codec_name": "pcm_s16le"
        }]
    }

    with patch.object(preprocessor, "_get_audio_info", return_value=correct_info):
        assert preprocessor._needs_conversion(mock_audio_file) is False

def test_process_file_not_found(preprocessor):
    with pytest.raises(FileNotFoundError):
        preprocessor.process(Path("nonexistent.wav"))

def test_process_conversion_success(preprocessor, mock_audio_file):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        output_path = preprocessor.process(mock_audio_file)

        assert output_path.parent == preprocessor.cache_dir
        assert output_path.name == f"{mock_audio_file.stem}_16khz_mono.wav"

        # Verify ffmpeg call
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert "-acodec" in args
        assert "pcm_s16le" in args
        assert "-ar" in args
        assert "16000" in args
        assert "-ac" in args
        assert "1" in args

def test_process_cached_valid(preprocessor, mock_audio_file):
    # Setup cached file
    cached_file = preprocessor.cache_dir / f"{mock_audio_file.stem}_16khz_mono.wav"
    cached_file.touch()

    with patch.object(preprocessor, "_needs_conversion", return_value=False) as mock_check:
        with patch("subprocess.run") as mock_run:
            result = preprocessor.process(mock_audio_file)

            assert result == cached_file
            mock_run.assert_not_called()
            mock_check.assert_called_once_with(cached_file)

def test_process_force_reprocess(preprocessor, mock_audio_file):
    # Setup cached file
    cached_file = preprocessor.cache_dir / f"{mock_audio_file.stem}_16khz_mono.wav"
    cached_file.touch()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        preprocessor.process(mock_audio_file, force=True)

        mock_run.assert_called_once()

def test_process_conversion_failure(preprocessor, mock_audio_file):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Conversion failed"
        )

        with pytest.raises(RuntimeError, match="Audio conversion failed"):
            preprocessor.process(mock_audio_file)

def test_process_for_diarization_already_mono(preprocessor, mock_audio_file):
    info_mono = {
        "streams": [{
            "codec_type": "audio",
            "channels": 1,
            "codec_name": "pcm_s16le"
        }]
    }

    with patch.object(preprocessor, "_get_audio_info", return_value=info_mono):
        result = preprocessor.process_for_diarization(mock_audio_file)
        assert result == mock_audio_file

def test_process_for_diarization_needs_conversion(preprocessor, mock_audio_file):
    info_stereo = {
        "streams": [{
            "codec_type": "audio",
            "channels": 2,
            "codec_name": "pcm_s16le"
        }]
    }

    with patch.object(preprocessor, "_get_audio_info", return_value=info_stereo):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            output_path = preprocessor.process_for_diarization(mock_audio_file)

            assert output_path.name == f"{mock_audio_file.stem}_mono_diarize.wav"

            # Verify ffmpeg args for diarization conversion
            args = mock_run.call_args[0][0]
            assert "-ac" in args
            assert "1" in args
            # Should NOT set sample rate (keep original)
            assert "-ar" not in args

def test_process_for_diarization_file_not_found(preprocessor):
    with pytest.raises(FileNotFoundError):
        preprocessor.process_for_diarization(Path("nonexistent.wav"))

def test_process_for_diarization_failure(preprocessor, mock_audio_file):
    info_stereo = {
        "streams": [{
            "codec_type": "audio",
            "channels": 2,
            "codec_name": "pcm_s16le"
        }]
    }

    with patch.object(preprocessor, "_get_audio_info", return_value=info_stereo):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stderr="Diarization conversion failed"
            )

            with pytest.raises(RuntimeError, match="Audio conversion for diarization failed"):
                preprocessor.process_for_diarization(mock_audio_file)

def test_process_cached_validation_error_reprocesses(preprocessor, mock_audio_file):
    # Setup cached file
    cached_file = preprocessor.cache_dir / f"{mock_audio_file.stem}_16khz_mono.wav"
    cached_file.touch()

    # Simulate validation error (e.g., corrupt file causes _needs_conversion to crash or return True)
    # The code catches Exception and reprocesses
    with patch.object(preprocessor, "_needs_conversion", side_effect=Exception("Corrupt")):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = preprocessor.process(mock_audio_file)

            # Should have reprocessed despite cache existing
            mock_run.assert_called_once()
            assert result == cached_file

def test_process_cached_invalid_reprocesses(preprocessor, mock_audio_file):
    # Setup cached file that exists but has wrong format
    cached_file = preprocessor.cache_dir / f"{mock_audio_file.stem}_16khz_mono.wav"
    cached_file.touch()

    # Mock _needs_conversion to return True (indicating file is invalid/needs conversion)
    with patch.object(preprocessor, "_needs_conversion", return_value=True):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = preprocessor.process(mock_audio_file)

            # Should have reprocessed because cache was invalid
            mock_run.assert_called_once()
            assert result == cached_file

def test_process_for_diarization_cached(preprocessor, mock_audio_file):
    # Setup cached file
    cached_file = preprocessor.cache_dir / f"{mock_audio_file.stem}_mono_diarize.wav"
    cached_file.touch()

    # Mock _get_audio_info to simulate input file needs conversion (stereo)
    # so we reach the cache check logic
    info_stereo = {
        "streams": [{
            "codec_type": "audio",
            "channels": 2,
            "codec_name": "pcm_s16le"
        }]
    }

    with patch.object(preprocessor, "_get_audio_info", return_value=info_stereo):
        with patch("subprocess.run") as mock_run:
            result = preprocessor.process_for_diarization(mock_audio_file)
            assert result == cached_file
            mock_run.assert_not_called()

def test_process_for_diarization_already_mono_f32le(preprocessor, mock_audio_file):
    info_mono = {
        "streams": [{
            "codec_type": "audio",
            "channels": 1,
            "codec_name": "pcm_f32le"
        }]
    }

    with patch.object(preprocessor, "_get_audio_info", return_value=info_mono):
        result = preprocessor.process_for_diarization(mock_audio_file)
        assert result == mock_audio_file

def test_process_for_diarization_force_reprocess(preprocessor, mock_audio_file):
    # Setup cached file
    cached_file = preprocessor.cache_dir / f"{mock_audio_file.stem}_mono_diarize.wav"
    cached_file.touch()

    info_stereo = {
        "streams": [{
            "codec_type": "audio",
            "channels": 2,
            "codec_name": "pcm_s16le"
        }]
    }

    with patch.object(preprocessor, "_get_audio_info", return_value=info_stereo):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            preprocessor.process_for_diarization(mock_audio_file, force=True)

            mock_run.assert_called_once()

def test_needs_conversion_no_audio_streams(preprocessor, mock_audio_file):
    info_no_audio = {
        "streams": [{
            "codec_type": "video",
            "width": 1920
        }]
    }

    with patch.object(preprocessor, "_get_audio_info", return_value=info_no_audio):
        assert preprocessor._needs_conversion(mock_audio_file) is True

def test_needs_conversion_multistream_one_valid(preprocessor, mock_audio_file):
    # Stream 1: Bad audio
    # Stream 2: Valid audio
    # Should return False (no conversion needed)
    info = {
        "streams": [
            {
                "codec_type": "audio",
                "sample_rate": "44100",
                "channels": 2,
                "codec_name": "aac"
            },
            {
                "codec_type": "audio",
                "sample_rate": "16000",
                "channels": 1,
                "codec_name": "pcm_s16le"
            }
        ]
    }

    with patch.object(preprocessor, "_get_audio_info", return_value=info):
        assert preprocessor._needs_conversion(mock_audio_file) is False

def test_get_duration(preprocessor, mock_audio_file):
    info = {"format": {"duration": "123.45"}}

    with patch.object(preprocessor, "_get_audio_info", return_value=info):
        duration = preprocessor.get_duration(mock_audio_file)
        assert duration == 123.45

def test_get_audio_info_json_error(preprocessor, mock_audio_file):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Not JSON",
            stderr=""
        )

        with pytest.raises(json.JSONDecodeError):
            preprocessor._get_audio_info(mock_audio_file)

def test_process_input_already_valid_creates_copy(preprocessor, mock_audio_file):
    # Even if input is valid 16kHz mono, process() creates a cached copy
    # This documents current behavior, contrasting with process_for_diarization which returns input

    # We mock run to success
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        output_path = preprocessor.process(mock_audio_file)

        # It should still run ffmpeg to create the cached file
        mock_run.assert_called_once()
        assert output_path != mock_audio_file
        assert output_path.parent == preprocessor.cache_dir

def test_process_for_diarization_empty_streams(preprocessor, mock_audio_file):
    # If streams list is empty, it should proceed to conversion
    info = {"streams": []}

    with patch.object(preprocessor, "_get_audio_info", return_value=info):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            output_path = preprocessor.process_for_diarization(mock_audio_file)

            # Should have converted
            mock_run.assert_called_once()
            assert output_path.name == f"{mock_audio_file.stem}_mono_diarize.wav"


def test_cleanup_specific_file(preprocessor):
    f = preprocessor.cache_dir / "test.wav"
    f.touch()
    assert f.exists()

    preprocessor.cleanup(f)
    assert not f.exists()

def test_cleanup_all(preprocessor):
    f1 = preprocessor.cache_dir / "test1.wav"
    f2 = preprocessor.cache_dir / "test2.wav"
    f3 = preprocessor.cache_dir / "keep.txt"

    f1.touch()
    f2.touch()
    f3.touch()

    preprocessor.cleanup()

    assert not f1.exists()
    assert not f2.exists()
    assert f3.exists()  # Should only clean .wav files
