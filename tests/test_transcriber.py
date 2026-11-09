import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from voxscriber import transcriber as tr
from voxscriber.transcriber import Transcriber, _tokens_to_words


@dataclass
class Tok:
    text: str
    start: float
    end: float
    confidence: float = 1.0


def test_tokens_to_words_merges_subword_tokens():
    tokens = [
        Tok(" Hel", 0.0, 0.2, 0.9),
        Tok("lo", 0.2, 0.4, 0.8),
        Tok(",", 0.4, 0.45),
        Tok(" world", 0.5, 0.9, 0.7),
        Tok(".", 0.9, 0.95),
    ]
    words = _tokens_to_words(tokens)
    assert [w.text for w in words] == ["Hello,", "world."]
    assert (words[0].start, words[0].end) == (0.0, 0.45)
    assert (words[1].start, words[1].end) == (0.5, 0.95)
    assert words[0].confidence == 0.8


def test_tokens_to_words_first_token_without_space():
    words = _tokens_to_words([Tok("Hola", 1.0, 1.3), Tok(" amigo", 1.4, 1.8)])
    assert [w.text for w in words] == ["Hola", "amigo"]


def test_tokens_to_words_zero_duration_word_gets_min_length():
    words = _tokens_to_words([Tok(" when", 145.04, 145.04)])
    assert words[0].end > words[0].start
    assert words[0].end == pytest.approx(145.04 + tr.PARAKEET_MIN_WORD_SECONDS)


def test_tokens_to_words_empty():
    assert _tokens_to_words([]) == []


def test_parakeet_backend_selected_on_apple_silicon():
    with patch.object(tr, "is_apple_silicon", return_value=True):
        assert Transcriber(model="parakeet-v3").backend == "parakeet"
        assert Transcriber(model="large-v3-turbo").backend == "mlx"


def test_parakeet_rejected_off_apple_silicon():
    with patch.object(tr, "is_apple_silicon", return_value=False):
        with pytest.raises(ValueError, match="Apple Silicon"):
            Transcriber(model="parakeet-v3")
        assert Transcriber(model="small").backend == "faster-whisper"


def test_transcribe_parakeet_maps_sentences_and_words():
    sentence = SimpleNamespace(
        text=" Hello world.",
        start=0.0,
        end=0.95,
        tokens=[Tok(" Hello", 0.0, 0.4), Tok(" world", 0.5, 0.9), Tok(".", 0.9, 0.95)],
    )
    engine = MagicMock()
    engine.transcribe.return_value = SimpleNamespace(text="Hello world.", sentences=[sentence])

    t = Transcriber(model="parakeet-v3", backend="parakeet")
    t._engine = engine
    result = t.transcribe("audio.wav")

    engine.transcribe.assert_called_once_with(
        "audio.wav",
        chunk_duration=tr.PARAKEET_CHUNK_SECONDS,
        overlap_duration=tr.PARAKEET_OVERLAP_SECONDS,
    )
    assert result.text == "Hello world."
    assert result.language == "auto"
    assert result.duration == 0.95
    assert [w.text for w in result.segments[0].words] == ["Hello", "world."]


def test_unload_releases_engine_and_clears_mlx_cache():
    fake_mx = MagicMock()
    t = Transcriber(model="parakeet-v3", backend="parakeet")
    t._engine = object()
    with patch.dict(sys.modules, {"mlx": MagicMock(core=fake_mx), "mlx.core": fake_mx}):
        t.unload()
    assert t._engine is None
    fake_mx.clear_cache.assert_called_once()


def test_unload_clears_mlx_whisper_model_holder():
    holder = SimpleNamespace(model=object(), model_path="x")
    fake_mx = MagicMock()
    t = Transcriber(model="large-v3-turbo", backend="mlx")
    t._engine = object()
    with patch.dict(sys.modules, {
        "mlx_whisper": MagicMock(),
        "mlx_whisper.transcribe": SimpleNamespace(ModelHolder=holder),
        "mlx": MagicMock(core=fake_mx),
        "mlx.core": fake_mx,
    }):
        t.unload()
    assert holder.model is None and holder.model_path is None
