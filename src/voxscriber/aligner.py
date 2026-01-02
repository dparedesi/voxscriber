"""
Alignment Module

Merges transcription results with speaker diarization to produce
speaker-attributed transcripts with word-level accuracy.

This is the critical "glue" that makes professional diarization work.
"""

from typing import List, Optional, Tuple
import re
from dataclasses import dataclass, field, asdict
from .transcriber import TranscriptionResult, Segment, Word
from .diarizer import DiarizationResult, SpeakerSegment


@dataclass
class AttributedWord:
    """A word with speaker attribution."""
    text: str
    start: float
    end: float
    speaker: Optional[str]
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AttributedSegment:
    """A segment of speech attributed to a speaker."""
    speaker: str
    text: str
    start: float
    end: float
    words: List[AttributedWord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class AlignedTranscript:
    """Complete aligned transcript with speaker attribution."""
    segments: List[AttributedSegment]
    speakers: List[str]
    duration: float
    language: str

    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "speakers": self.speakers,
            "duration": self.duration,
            "language": self.language,
        }

    def get_speaker_turns(self) -> List[Tuple[str, str, float, float]]:
        """Get list of (speaker, text, start, end) tuples."""
        return [
            (seg.speaker, seg.text, seg.start, seg.end)
            for seg in self.segments
        ]


class Aligner:
    """Aligns transcription with diarization results."""

    def __init__(
        self,
        min_word_duration: float = 0.01,
        overlap_threshold: float = 0.5,
        merge_same_speaker: bool = True,
        merge_gap_threshold: float = 1.0,
        filter_non_speech: bool = True,
        min_speech_overlap: float = 0.3,
        filter_hallucinations: bool = True,
        max_repetition_ratio: float = 0.5,
    ):
        """
        Initialize aligner.

        Args:
            min_word_duration: Minimum duration to consider a word valid
            overlap_threshold: Minimum overlap ratio to assign speaker
            merge_same_speaker: Merge consecutive segments from same speaker
            merge_gap_threshold: Max gap (seconds) to merge same-speaker segments
            filter_non_speech: Discard words that don't overlap with any diarization segment
                              (helps filter Whisper hallucinations during silence)
            min_speech_overlap: Minimum overlap ratio with speech to keep a word (0.0-1.0)
            filter_hallucinations: Filter out segments with excessive repetition
            max_repetition_ratio: Max ratio of repeated words before considering hallucination
        """
        self.min_word_duration = min_word_duration
        self.overlap_threshold = overlap_threshold
        self.merge_same_speaker = merge_same_speaker
        self.merge_gap_threshold = merge_gap_threshold
        self.filter_non_speech = filter_non_speech
        self.min_speech_overlap = min_speech_overlap
        self.filter_hallucinations = filter_hallucinations
        self.max_repetition_ratio = max_repetition_ratio

    def _word_overlaps_speech(
        self,
        word: Word,
        diarization: DiarizationResult,
    ) -> bool:
        """
        Check if a word overlaps with any detected speech segment.

        This uses diarization as VAD to filter hallucinations.
        """
        word_duration = word.end - word.start
        if word_duration <= 0:
            return False

        total_overlap = 0.0
        for seg in diarization.segments:
            overlap = seg.overlaps(word.start, word.end)
            total_overlap += overlap

        overlap_ratio = total_overlap / word_duration
        return overlap_ratio >= self.min_speech_overlap

    def _get_speaker_for_word(
        self,
        word: Word,
        diarization: DiarizationResult,
    ) -> Optional[str]:
        """
        Determine speaker for a word based on diarization.

        Uses overlap-based assignment for accuracy.
        """
        word_duration = word.end - word.start
        if word_duration < self.min_word_duration:
            # Very short word, use midpoint
            return diarization.get_speaker_at((word.start + word.end) / 2)

        # Find speaker with maximum overlap
        return diarization.get_dominant_speaker(word.start, word.end)

    def _attribute_words(
        self,
        transcription: TranscriptionResult,
        diarization: DiarizationResult,
    ) -> List[AttributedWord]:
        """Assign speakers to all words, filtering non-speech if enabled."""
        attributed = []

        for segment in transcription.segments:
            for word in segment.words:
                # Filter out words that don't overlap with detected speech
                # This removes Whisper hallucinations during silence
                if self.filter_non_speech:
                    if not self._word_overlaps_speech(word, diarization):
                        continue

                speaker = self._get_speaker_for_word(word, diarization)
                attributed.append(AttributedWord(
                    text=word.text,
                    start=word.start,
                    end=word.end,
                    speaker=speaker,
                    confidence=word.confidence,
                ))

        return attributed

    def _create_segments(
        self,
        attributed_words: List[AttributedWord],
    ) -> List[AttributedSegment]:
        """Group attributed words into speaker segments."""
        if not attributed_words:
            return []

        segments = []
        current_speaker = attributed_words[0].speaker
        current_words = [attributed_words[0]]

        for word in attributed_words[1:]:
            # Check if we should start a new segment
            should_split = False

            if word.speaker != current_speaker:
                should_split = True
            elif self.merge_same_speaker:
                # Check gap between words
                gap = word.start - current_words[-1].end
                if gap > self.merge_gap_threshold:
                    should_split = True

            if should_split:
                # Finalize current segment
                if current_words:
                    text = " ".join(w.text for w in current_words)
                    segments.append(AttributedSegment(
                        speaker=current_speaker or "UNKNOWN",
                        text=text,
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        words=current_words,
                    ))
                current_speaker = word.speaker
                current_words = [word]
            else:
                current_words.append(word)

        # Don't forget the last segment
        if current_words:
            text = " ".join(w.text for w in current_words)
            segments.append(AttributedSegment(
                speaker=current_speaker or "UNKNOWN",
                text=text,
                start=current_words[0].start,
                end=current_words[-1].end,
                words=current_words,
            ))

        return segments

    def _merge_adjacent_segments(
        self,
        segments: List[AttributedSegment],
    ) -> List[AttributedSegment]:
        """Merge adjacent segments from the same speaker."""
        if not segments or not self.merge_same_speaker:
            return segments

        merged = [segments[0]]

        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg.start - prev.end

            if seg.speaker == prev.speaker and gap <= self.merge_gap_threshold:
                # Merge with previous
                merged[-1] = AttributedSegment(
                    speaker=prev.speaker,
                    text=prev.text + " " + seg.text,
                    start=prev.start,
                    end=seg.end,
                    words=prev.words + seg.words,
                )
            else:
                merged.append(seg)

        return merged

    # Common Whisper hallucination patterns (case-insensitive)
    HALLUCINATION_PATTERNS = {
        # English patterns
        "thank you", "thanks for watching", "thanks for listening",
        "please subscribe", "like and subscribe", "see you next time",
        "bye bye", "goodbye", "good night",
        # Filler loops
        "um", "uh", "hmm", "ah", "home",
        # Foreign intrusions during English speech
        "bipolar", "geht", "webinar", "amen",
        # Spanish patterns
        "gracias", "adios", "hasta luego",
        # Nonsense/gibberish patterns
        "homebrainidad",
    }

    # Regex patterns for pure punctuation/noise segments
    NOISE_PATTERN = re.compile(r'^[\s\.\,\!\?\-\:\;\'\"]+$')

    def _is_hallucination(self, text: str) -> bool:
        """
        Detect if text is likely a Whisper hallucination.

        Hallucinations often manifest as:
        - Repeated words/phrases: "Thank you. Thank you. Thank you."
        - Filler loops: "um um um um", "uh uh uh"
        - Repetitive patterns: "Thanks for watching. Thanks for watching."
        - Foreign word intrusions: "geht geht geht"
        - Single word repeated: "Webinar Webinar Webinar"
        - Known YouTube/podcast outros: "Please subscribe", "Like and subscribe"
        - Pure punctuation: "!!!!!"
        """
        if not text or len(text) < 4:
            return False

        text_lower = text.lower().strip()

        # Check for pure punctuation/noise
        if self.NOISE_PATTERN.match(text):
            return True

        # Check if entire text is just known hallucination patterns repeated
        for pattern in self.HALLUCINATION_PATTERNS:
            # Check if text is mostly this pattern repeated
            pattern_stripped = text_lower.replace(pattern, "").strip()
            # Remove punctuation and spaces for comparison
            remaining = re.sub(r'[\s\.,!?]+', '', pattern_stripped)
            if len(remaining) < 3 and pattern in text_lower:
                # Text is mostly this pattern
                return True

        # Normalize text for analysis
        words = text_lower.split()
        if not words:
            return False

        # For very short texts (1-4 words), check if it's pure repetition
        if len(words) <= 4:
            unique_words = set(re.sub(r'[^\w]', '', w) for w in words if w)
            unique_words.discard('')
            # If all words are the same (or just punctuation variations)
            if len(unique_words) <= 1 and len(words) >= 2:
                return True

        if len(words) < 3:
            return False

        # Count word frequencies
        word_counts = {}
        for word in words:
            # Strip punctuation for comparison
            clean = re.sub(r'[^\w\s]', '', word)
            if clean and len(clean) > 1:  # Ignore single-char words
                word_counts[clean] = word_counts.get(clean, 0) + 1

        if not word_counts:
            return False

        # Check for excessive repetition of any word
        max_count = max(word_counts.values())
        total_words = len(words)

        # If any word appears more than max_repetition_ratio of the time, it's suspicious
        if max_count >= 3 and max_count / total_words > self.max_repetition_ratio:
            return True

        # Check for consecutive repetitions (3+ in a row is always suspicious)
        consecutive = 1
        prev_clean = ""
        for word in words:
            curr_clean = re.sub(r'[^\w]', '', word.lower())
            if curr_clean and curr_clean == prev_clean:
                consecutive += 1
                if consecutive >= 3:  # 3+ consecutive same words
                    return True
            else:
                consecutive = 1
            prev_clean = curr_clean

        # Check for repeated consecutive patterns (e.g., "thank you thank you thank you")
        for pattern_len in [2, 3]:
            if len(words) >= pattern_len * 3:
                patterns = {}
                for i in range(len(words) - pattern_len + 1):
                    pattern = " ".join(re.sub(r'[^\w]', '', w.lower()) for w in words[i:i + pattern_len])
                    patterns[pattern] = patterns.get(pattern, 0) + 1

                max_pattern_count = max(patterns.values()) if patterns else 0
                possible_patterns = len(words) - pattern_len + 1
                if max_pattern_count >= 3 and max_pattern_count / possible_patterns > 0.35:
                    return True

        return False

    def _clean_hallucination(self, text: str) -> str:
        """
        Clean hallucinated text by removing excessive repetitions.

        Returns cleaned text or empty string if entirely hallucination.
        """
        words = text.split()
        if len(words) < 4:
            return text

        # Remove consecutive duplicate words
        cleaned = [words[0]]
        consecutive_count = 1

        for word in words[1:]:
            # Normalize for comparison (ignore punctuation)
            prev_clean = re.sub(r'[^\w]', '', cleaned[-1].lower())
            curr_clean = re.sub(r'[^\w]', '', word.lower())

            if prev_clean == curr_clean:
                consecutive_count += 1
                # Allow max 2 consecutive same words (for natural speech like "sí sí")
                if consecutive_count <= 2:
                    cleaned.append(word)
            else:
                consecutive_count = 1
                cleaned.append(word)

        result = " ".join(cleaned)

        # If still looks like hallucination after cleaning, return empty
        if self._is_hallucination(result):
            return ""

        return result

    def _filter_hallucinated_segments(
        self,
        segments: List[AttributedSegment],
    ) -> List[AttributedSegment]:
        """Filter or clean segments that appear to be hallucinations."""
        if not self.filter_hallucinations:
            return segments

        filtered = []
        for seg in segments:
            if self._is_hallucination(seg.text):
                # Try to clean it
                cleaned_text = self._clean_hallucination(seg.text)
                if cleaned_text:
                    # Update segment with cleaned text
                    filtered.append(AttributedSegment(
                        speaker=seg.speaker,
                        text=cleaned_text,
                        start=seg.start,
                        end=seg.end,
                        words=seg.words,  # Keep original words for now
                    ))
                # else: discard entirely
            else:
                filtered.append(seg)

        return filtered

    def align(
        self,
        transcription: TranscriptionResult,
        diarization: DiarizationResult,
    ) -> AlignedTranscript:
        """
        Align transcription with diarization.

        Args:
            transcription: Result from Transcriber
            diarization: Result from Diarizer

        Returns:
            AlignedTranscript with speaker-attributed segments
        """
        # Step 1: Attribute speakers to each word
        attributed_words = self._attribute_words(transcription, diarization)

        # Step 2: Group into segments by speaker
        segments = self._create_segments(attributed_words)

        # Step 3: Merge adjacent same-speaker segments
        segments = self._merge_adjacent_segments(segments)

        # Step 4: Filter hallucinated segments (repetitive patterns)
        segments = self._filter_hallucinated_segments(segments)

        return AlignedTranscript(
            segments=segments,
            speakers=diarization.speaker_labels,
            duration=transcription.duration,
            language=transcription.language,
        )

    def align_with_fallback(
        self,
        transcription: TranscriptionResult,
        diarization: Optional[DiarizationResult],
    ) -> AlignedTranscript:
        """
        Align with fallback for missing diarization.

        If diarization is None, returns transcript with single speaker.
        """
        if diarization is None:
            # No diarization - treat as single speaker
            segments = []
            for seg in transcription.segments:
                words = [
                    AttributedWord(
                        text=w.text,
                        start=w.start,
                        end=w.end,
                        speaker="SPEAKER_00",
                        confidence=w.confidence,
                    )
                    for w in seg.words
                ]
                segments.append(AttributedSegment(
                    speaker="SPEAKER_00",
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    words=words,
                ))

            return AlignedTranscript(
                segments=segments,
                speakers=["SPEAKER_00"],
                duration=transcription.duration,
                language=transcription.language,
            )

        return self.align(transcription, diarization)
