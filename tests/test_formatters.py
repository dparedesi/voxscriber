import json

import pytest

from voxscriber.aligner import AlignedTranscript, AttributedSegment, AttributedWord
from voxscriber.formatters import (
    OutputFormatter,
    format_timestamp_simple,
    format_timestamp_srt,
    format_timestamp_vtt,
)


@pytest.fixture
def sample_transcript():
    segments = [
        AttributedSegment(
            speaker="SPEAKER_00",
            text="Hello world.",
            start=0.0,
            end=1.5,
            words=[
                AttributedWord("Hello", 0.0, 0.5, "SPEAKER_00"),
                AttributedWord("world.", 0.6, 1.5, "SPEAKER_00")
            ]
        ),
        AttributedSegment(
            speaker="SPEAKER_01",
            text="Hi there.",
            start=2.0,
            end=3.5,
            words=[
                AttributedWord("Hi", 2.0, 2.5, "SPEAKER_01"),
                AttributedWord("there.", 2.6, 3.5, "SPEAKER_01")
            ]
        )
    ]
    return AlignedTranscript(
        segments=segments,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        duration=3.5,
        language="en"
    )

@pytest.fixture
def empty_transcript():
    return AlignedTranscript(
        segments=[],
        speakers=[],
        duration=0.0,
        language="en"
    )

@pytest.fixture
def long_transcript():
    # Create a segment with long text to test line wrapping
    long_text = "This is a very long sentence that should definitely be wrapped when formatted as subtitles because it exceeds the character limit per line."
    words = []
    # Simplified word generation for testing
    for i, word in enumerate(long_text.split()):
        words.append(AttributedWord(word, i*0.5, (i+1)*0.5, "SPEAKER_00"))

    segments = [
        AttributedSegment(
            speaker="SPEAKER_00",
            text=long_text,
            start=0.0,
            end=len(words)*0.5,
            words=words
        )
    ]
    return AlignedTranscript(
        segments=segments,
        speakers=["SPEAKER_00"],
        duration=len(words)*0.5,
        language="en"
    )

def test_timestamp_formatting():
    assert format_timestamp_srt(0.0) == "00:00:00,000"
    assert format_timestamp_srt(61.5) == "00:01:01,500"
    assert format_timestamp_srt(3661.001) == "01:01:01,001"

    assert format_timestamp_vtt(0.0) == "00:00:00.000"
    assert format_timestamp_vtt(61.5) == "00:01:01.500"

    assert format_timestamp_simple(0.0) == "00:00"
    assert format_timestamp_simple(65.0) == "01:05"

def test_speaker_mapping():
    mapping = {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
    formatter = OutputFormatter(speaker_names=mapping)
    assert formatter._get_speaker_name("SPEAKER_00") == "Alice"
    assert formatter._get_speaker_name("SPEAKER_01") == "Bob"
    assert formatter._get_speaker_name("SPEAKER_99") == "SPEAKER_99"

def test_to_json(sample_transcript):
    formatter = OutputFormatter()
    json_str = formatter.to_json(sample_transcript)
    data = json.loads(json_str)

    assert data["duration"] == 3.5
    assert len(data["segments"]) == 2
    assert data["segments"][0]["text"] == "Hello world."
    assert data["segments"][0]["speaker"] == "SPEAKER_00"
    assert "words" in data["segments"][0]
    assert len(data["segments"][0]["words"]) == 2

def test_to_json_with_mapping(sample_transcript):
    mapping = {"SPEAKER_00": "Alice"}
    formatter = OutputFormatter(speaker_names=mapping)
    json_str = formatter.to_json(sample_transcript, include_words=False)
    data = json.loads(json_str)

    assert data["segments"][0]["speaker"] == "Alice"
    assert "words" not in data["segments"][0]

def test_to_txt(sample_transcript):
    formatter = OutputFormatter()
    txt = formatter.to_txt(sample_transcript)

    lines = txt.splitlines()
    assert len(lines) == 2
    assert "[00:00] SPEAKER_00: Hello world." in lines[0]
    assert "[00:02] SPEAKER_01: Hi there." in lines[1]

def test_to_txt_no_timestamps(sample_transcript):
    formatter = OutputFormatter()
    txt = formatter.to_txt(sample_transcript, include_timestamps=False)

    assert "SPEAKER_00: Hello world." in txt
    assert "[00:00]" not in txt

def test_to_txt_paragraph_gap():
    # Create segments with large gap
    segments = [
        AttributedSegment(
            speaker="SPEAKER_00",
            text="First.",
            start=0.0,
            end=1.0,
            words=[]
        ),
        AttributedSegment(
            speaker="SPEAKER_00",
            text="Second.",
            start=5.0, # 4s gap
            end=6.0,
            words=[]
        )
    ]
    transcript = AlignedTranscript(segments, ["SPEAKER_00"], 6.0, "en")

    formatter = OutputFormatter()
    txt = formatter.to_txt(transcript, paragraph_gap=3.0)

    # Should have an empty line between segments
    assert "\n\n" in txt

def test_to_srt(sample_transcript):
    formatter = OutputFormatter()
    srt = formatter.to_srt(sample_transcript)

    entries = srt.strip().split("\n\n")
    assert len(entries) == 2

    entry1 = entries[0].split("\n")
    assert entry1[0] == "1"
    assert "00:00:00,000 --> 00:00:01,500" in entry1[1]
    assert "[SPEAKER_00] Hello world." in entry1[2]

def test_to_srt_wrapping(long_transcript):
    formatter = OutputFormatter()
    # Use small limit to force wrapping
    srt = formatter.to_srt(long_transcript, max_chars_per_line=40)

    entry = srt.strip()
    lines = entry.split("\n")
    # Header + Time + Text Lines. Text lines should be > 1
    assert len(lines) > 3

def test_to_vtt(sample_transcript):
    formatter = OutputFormatter()
    vtt = formatter.to_vtt(sample_transcript)

    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt
    assert "<v SPEAKER_00>Hello world." in vtt

def test_to_md(sample_transcript):
    formatter = OutputFormatter()
    md = formatter.to_md(sample_transcript, title="Test Transcript")

    assert "# Test Transcript" in md
    assert "**SPEAKER_00:** Hello world." in md

def test_empty_transcript(empty_transcript):
    formatter = OutputFormatter()
    assert formatter.to_json(empty_transcript)
    assert formatter.to_txt(empty_transcript) == ""
    assert formatter.to_srt(empty_transcript) == ""
    assert formatter.to_vtt(empty_transcript).strip() == "WEBVTT"
    assert formatter.to_md(empty_transcript) == ""

def test_save(sample_transcript, tmp_path):
    formatter = OutputFormatter()

    # Test JSON
    json_path = tmp_path / "test.json"
    formatter.save(sample_transcript, json_path)
    assert json_path.exists()
    assert json.loads(json_path.read_text())["duration"] == 3.5

    # Test auto-detection
    txt_path = tmp_path / "test.txt"
    formatter.save(sample_transcript, txt_path)
    assert txt_path.exists()
    assert "Hello world." in txt_path.read_text()

    # Test explicit format
    custom_path = tmp_path / "test.custom"
    formatter.save(sample_transcript, custom_path, format="md")
    assert "**SPEAKER_00:**" in custom_path.read_text()

    # Test invalid format
    with pytest.raises(ValueError):
        formatter.save(sample_transcript, tmp_path / "bad.ext", format="invalid")
