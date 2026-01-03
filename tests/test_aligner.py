import pytest

from voxscriber.aligner import AlignedTranscript, Aligner, AttributedSegment, AttributedWord
from voxscriber.diarizer import DiarizationResult, SpeakerSegment
from voxscriber.transcriber import Segment, TranscriptionResult, Word

# Fixtures

@pytest.fixture
def aligner():
    return Aligner()

@pytest.fixture
def sample_transcription():
    # A simple conversation: "Hello world. How are you?"
    return TranscriptionResult(
        text="Hello world How are you",
        segments=[
            Segment(
                text="Hello world",
                start=0.0,
                end=1.5,
                words=[
                    Word(text="Hello", start=0.0, end=0.5, confidence=0.9),
                    Word(text="world", start=0.6, end=1.5, confidence=0.9),
                ]
            ),
            Segment(
                text="How are you",
                start=2.0,
                end=3.5,
                words=[
                    Word(text="How", start=2.0, end=2.3, confidence=0.9),
                    Word(text="are", start=2.4, end=2.8, confidence=0.9),
                    Word(text="you", start=2.9, end=3.5, confidence=0.9),
                ]
            )
        ],
        language="en",
        duration=3.5
    )

@pytest.fixture
def sample_diarization():
    return DiarizationResult(
        segments=[
            # Speaker A says "Hello world"
            SpeakerSegment(speaker="SPEAKER_01", start=0.0, end=1.6),
            # Speaker B says "How are you"
            SpeakerSegment(speaker="SPEAKER_02", start=1.9, end=3.6),
        ],
        num_speakers=2,
        speaker_labels=["SPEAKER_01", "SPEAKER_02"]
    )

# Tests

def test_word_overlaps_speech(aligner, sample_diarization):
    # Word completely inside speech segment
    word_in = Word(text="test", start=0.5, end=1.0)
    assert aligner._word_overlaps_speech(word_in, sample_diarization) is True

    # Word completely outside speech segment
    word_out = Word(text="test", start=1.7, end=1.8) # Gap is 1.6 to 1.9
    assert aligner._word_overlaps_speech(word_out, sample_diarization) is False

    # Word partially overlapping (enough)
    # Segment ends at 1.6. Word 1.5-1.7. Overlap 0.1. Duration 0.2. Ratio 0.5 >= 0.3 (default)
    word_partial = Word(text="test", start=1.5, end=1.7)
    assert aligner._word_overlaps_speech(word_partial, sample_diarization) is True

def test_get_speaker_for_word(aligner, sample_diarization):
    # Clear ownership
    word_a = Word(text="A", start=0.1, end=0.4)
    assert aligner._get_speaker_for_word(word_a, sample_diarization) == "SPEAKER_01"

    word_b = Word(text="B", start=2.0, end=2.5)
    assert aligner._get_speaker_for_word(word_b, sample_diarization) == "SPEAKER_02"

    # Overlapping speakers scenario
    # Create a complex diarization with overlap
    complex_diarization = DiarizationResult(
        segments=[
            SpeakerSegment(speaker="S1", start=0.0, end=1.0),
            SpeakerSegment(speaker="S2", start=0.5, end=1.5),
        ],
        num_speakers=2,
        speaker_labels=["S1", "S2"]
    )

    # Word 0.6-0.9 is mostly in overlap, but S1 ends at 1.0, S2 starts at 0.5.
    # Both cover it fully.
    # Wait, the logic is `get_dominant_speaker`.
    # Word 0.0-0.4 -> S1
    # Word 1.1-1.4 -> S2
    # Word 0.6-0.8 -> overlaps both equally. The implementation usually picks one.
    # Let's test clear dominance.

    # Word mostly S1 (0.2 - 0.6). S1 covers 0.2-0.6 (0.4). S2 covers 0.5-0.6 (0.1).
    word_mostly_s1 = Word(text="W", start=0.2, end=0.6)
    assert aligner._get_speaker_for_word(word_mostly_s1, complex_diarization) == "S1"

    # Word mostly S2 (0.9 - 1.3). S1 covers 0.9-1.0 (0.1). S2 covers 0.9-1.3 (0.4).
    word_mostly_s2 = Word(text="W", start=0.9, end=1.3)
    assert aligner._get_speaker_for_word(word_mostly_s2, complex_diarization) == "S2"

def test_attribute_words(aligner, sample_transcription, sample_diarization):
    attributed = aligner._attribute_words(sample_transcription, sample_diarization)

    assert len(attributed) == 5
    assert attributed[0].text == "Hello"
    assert attributed[0].speaker == "SPEAKER_01"
    assert attributed[2].text == "How" # First word of second segment
    assert attributed[2].speaker == "SPEAKER_02"

def test_attribute_words_filter_non_speech(aligner):
    # Test filtering of words that don't overlap with speech
    transcript = TranscriptionResult(
        text="Noise Speech",
        segments=[
            Segment(text="Noise Speech", start=0.0, end=2.0, words=[
                Word(text="Noise", start=0.0, end=0.5),   # No speech here
                Word(text="Speech", start=1.0, end=1.5),  # Speech here
            ])
        ],
        language="en", duration=2.0
    )
    diarization = DiarizationResult(
        segments=[SpeakerSegment(speaker="S1", start=0.9, end=1.6)],
        num_speakers=1, speaker_labels=["S1"]
    )

    # Enable filtering
    aligner.filter_non_speech = True
    attributed = aligner._attribute_words(transcript, diarization)

    assert len(attributed) == 1
    assert attributed[0].text == "Speech"

    # Disable filtering
    aligner.filter_non_speech = False
    attributed = aligner._attribute_words(transcript, diarization)
    assert len(attributed) == 2

def test_create_segments(aligner):
    words = [
        AttributedWord("Hi", 0.0, 0.5, "S1"),
        AttributedWord("there", 0.6, 1.0, "S1"),
        AttributedWord("Hello", 1.5, 2.0, "S2"), # Different speaker
        AttributedWord("back", 2.1, 2.5, "S1"),  # Back to S1
    ]

    segments = aligner._create_segments(words)

    assert len(segments) == 3
    assert segments[0].speaker == "S1"
    assert segments[0].text == "Hi there"
    assert segments[1].speaker == "S2"
    assert segments[1].text == "Hello"
    assert segments[2].speaker == "S1"
    assert segments[2].text == "back"

def test_merge_adjacent_segments(aligner):
    # Default gap threshold is 1.0s
    segments = [
        AttributedSegment("S1", "Part 1", 0.0, 1.0),
        AttributedSegment("S1", "Part 2", 1.5, 2.5), # Gap 0.5s -> should merge
        AttributedSegment("S1", "Part 3", 4.0, 5.0), # Gap 1.5s -> should NOT merge
        AttributedSegment("S2", "Interruption", 5.5, 6.0),
        AttributedSegment("S2", "Continuation", 6.2, 7.0), # Gap 0.2s -> should merge
    ]

    merged = aligner._merge_adjacent_segments(segments)

    assert len(merged) == 3

    # First merged segment
    assert merged[0].speaker == "S1"
    assert merged[0].text == "Part 1 Part 2"
    assert merged[0].start == 0.0
    assert merged[0].end == 2.5

    # Second segment (was separate due to gap)
    assert merged[1].speaker == "S1"
    assert merged[1].text == "Part 3"

    # Third merged segment (S2)
    assert merged[2].speaker == "S2"
    assert merged[2].text == "Interruption Continuation"

def test_is_hallucination(aligner):
    # Test known patterns
    assert aligner._is_hallucination("Thank you. Thank you.") is True
    assert aligner._is_hallucination("Thanks for watching") is True
    assert aligner._is_hallucination("Amara.org") is False # Not in list, unless pattern matches

    # Test repetition
    assert aligner._is_hallucination("Test test test test") is True
    # Code flags even 2-word repetitions if the segment is short (<=4 words)
    assert aligner._is_hallucination("Test test") is True


    # Test consecutive repetition
    assert aligner._is_hallucination("one two two two three") is True

    # Test pattern repetition
    assert aligner._is_hallucination("going home going home going home going home") is True

    # Test pure punctuation
    assert aligner._is_hallucination("... ? !") is True

    # Test normal speech
    assert aligner._is_hallucination("This is a normal sentence.") is False

def test_clean_hallucination(aligner):
    # Basic cleaning
    text = "Hello hello world"
    # Should keep one or two, but remove excessive.
    # Logic: allows max 2 consecutive same words.
    assert aligner._clean_hallucination(text) == "Hello hello world"

    text = "Hello hello hello hello world"
    # cleaned: Hello hello world (keeps first 2)
    assert aligner._clean_hallucination(text) == "Hello hello world"

    # If cleaning results in empty or still hallucination
    text = "Thank you Thank you Thank you"
    assert aligner._clean_hallucination(text) == ""

def test_filter_hallucinated_segments(aligner):
    segments = [
        AttributedSegment("S1", "Real speech here", 0.0, 1.0),
        AttributedSegment("S1", "Thank you thank you thank you", 1.0, 2.0),
        AttributedSegment("S2", "More real speech", 2.0, 3.0),
    ]

    filtered = aligner._filter_hallucinated_segments(segments)

    assert len(filtered) == 2
    assert filtered[0].text == "Real speech here"
    assert filtered[1].text == "More real speech"

def test_align_full_flow(aligner, sample_transcription, sample_diarization):
    result = aligner.align(sample_transcription, sample_diarization)

    assert isinstance(result, AlignedTranscript)
    assert len(result.segments) == 2
    assert result.segments[0].speaker == "SPEAKER_01"
    assert result.segments[0].text == "Hello world"
    assert result.segments[1].speaker == "SPEAKER_02"
    assert result.segments[1].text == "How are you"
    assert result.speakers == ["SPEAKER_01", "SPEAKER_02"]

def test_align_with_fallback(aligner, sample_transcription):
    # Pass None for diarization
    result = aligner.align_with_fallback(sample_transcription, None)

    assert len(result.segments) == 2 # Same as transcription segments
    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[1].speaker == "SPEAKER_00"
    assert result.speakers == ["SPEAKER_00"]

def test_edge_cases(aligner):
    # Empty inputs
    empty_trans = TranscriptionResult("", [], "en", 0.0)
    empty_diar = DiarizationResult([], 0, [])

    result = aligner.align(empty_trans, empty_diar)
    assert len(result.segments) == 0

    # Single word alignment
    single_trans = TranscriptionResult("Hi", [
        Segment("Hi", 0.0, 0.5, [Word("Hi", 0.0, 0.5)])
    ], "en", 0.5)
    single_diar = DiarizationResult([
        SpeakerSegment("S1", 0.0, 0.5)
    ], 1, ["S1"])

    result = aligner.align(single_trans, single_diar)
    assert len(result.segments) == 1
    assert result.segments[0].speaker == "S1"

    # Word completely in gap (with filtering OFF)
    aligner.filter_non_speech = False
    gap_trans = TranscriptionResult("Ghost", [
        Segment("Ghost", 10.0, 10.5, [Word("Ghost", 10.0, 10.5)])
    ], "en", 11.0)
    # Diarization nowhere near
    gap_diar = DiarizationResult([
        SpeakerSegment("S1", 0.0, 1.0)
    ], 1, ["S1"])

    # Should assign to S1 (dominant speaker fallback?) or UNKNOWN?
    # Logic: _get_dominant_speaker returns None if no overlap.
    # But _create_segments uses "UNKNOWN" if speaker is None.
    # Let's check _get_speaker_for_word logic...
    # It returns None if no overlap.

    result = aligner.align(gap_trans, gap_diar)
    assert result.segments[0].speaker == "UNKNOWN"

def test_aligner_init_params():
    custom = Aligner(
        min_word_duration=0.5,
        overlap_threshold=0.9,
        merge_same_speaker=False
    )
    assert custom.min_word_duration == 0.5
    assert custom.overlap_threshold == 0.9
    assert custom.merge_same_speaker is False

