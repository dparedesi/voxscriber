"""
Output Formatters Module

Formats aligned transcripts into various output formats:
- JSON (structured data)
- SRT (subtitles with speaker labels)
- TXT (readable text transcript)
- VTT (WebVTT subtitles)
"""

import json
from pathlib import Path
from typing import Optional

from .aligner import AlignedTranscript


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_timestamp_simple(seconds: float) -> str:
    """Format seconds as simple timestamp (MM:SS)."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


class OutputFormatter:
    """Formats aligned transcripts into various output formats."""

    def __init__(self, speaker_names: Optional[dict] = None):
        """
        Initialize formatter.

        Args:
            speaker_names: Optional mapping of speaker IDs to names
                          e.g., {"SPEAKER_00": "John", "SPEAKER_01": "Jane"}
        """
        self.speaker_names = speaker_names or {}

    def _get_speaker_name(self, speaker_id: str) -> str:
        """Get display name for speaker."""
        return self.speaker_names.get(speaker_id, speaker_id)

    def to_json(
        self,
        transcript: AlignedTranscript,
        include_words: bool = True,
        indent: int = 2,
    ) -> str:
        """
        Format as JSON.

        Args:
            transcript: Aligned transcript
            include_words: Include word-level detail
            indent: JSON indentation

        Returns:
            JSON string
        """
        data = transcript.to_dict()

        # Apply speaker names
        for seg in data["segments"]:
            seg["speaker"] = self._get_speaker_name(seg["speaker"])
            if not include_words:
                del seg["words"]

        data["speakers"] = [
            self._get_speaker_name(s) for s in data["speakers"]
        ]

        return json.dumps(data, indent=indent, ensure_ascii=False)

    def to_txt(
        self,
        transcript: AlignedTranscript,
        include_timestamps: bool = True,
        paragraph_gap: float = 2.0,
    ) -> str:
        """
        Format as readable text transcript.

        Args:
            transcript: Aligned transcript
            include_timestamps: Include timestamps
            paragraph_gap: Gap (seconds) that triggers new paragraph

        Returns:
            Formatted text
        """
        lines = []
        last_end = 0

        for seg in transcript.segments:
            # Add paragraph break for large gaps
            if seg.start - last_end > paragraph_gap and lines:
                lines.append("")

            speaker = self._get_speaker_name(seg.speaker)

            if include_timestamps:
                timestamp = format_timestamp_simple(seg.start)
                lines.append(f"[{timestamp}] {speaker}: {seg.text}")
            else:
                lines.append(f"{speaker}: {seg.text}")

            last_end = seg.end

        return "\n".join(lines)

    def to_srt(
        self,
        transcript: AlignedTranscript,
        max_chars_per_line: int = 42,
        include_speaker: bool = True,
    ) -> str:
        """
        Format as SRT subtitles.

        Args:
            transcript: Aligned transcript
            max_chars_per_line: Maximum characters per subtitle line
            include_speaker: Include speaker name in subtitle

        Returns:
            SRT formatted string
        """
        srt_entries = []
        index = 1

        for seg in transcript.segments:
            speaker = self._get_speaker_name(seg.speaker)
            text = seg.text

            # Add speaker prefix
            if include_speaker:
                text = f"[{speaker}] {text}"

            # Split long text into multiple lines
            if len(text) > max_chars_per_line:
                words = text.split()
                lines = []
                current_line = []
                current_len = 0

                for word in words:
                    if current_len + len(word) + 1 > max_chars_per_line and current_line:
                        lines.append(" ".join(current_line))
                        current_line = [word]
                        current_len = len(word)
                    else:
                        current_line.append(word)
                        current_len += len(word) + 1

                if current_line:
                    lines.append(" ".join(current_line))

                text = "\n".join(lines)

            start_ts = format_timestamp_srt(seg.start)
            end_ts = format_timestamp_srt(seg.end)

            srt_entries.append(f"{index}\n{start_ts} --> {end_ts}\n{text}\n")
            index += 1

        return "\n".join(srt_entries)

    def to_vtt(
        self,
        transcript: AlignedTranscript,
        include_speaker: bool = True,
    ) -> str:
        """
        Format as WebVTT subtitles.

        Args:
            transcript: Aligned transcript
            include_speaker: Include speaker name

        Returns:
            VTT formatted string
        """
        lines = ["WEBVTT", ""]

        for seg in transcript.segments:
            speaker = self._get_speaker_name(seg.speaker)
            text = seg.text

            if include_speaker:
                text = f"<v {speaker}>{text}"

            start_ts = format_timestamp_vtt(seg.start)
            end_ts = format_timestamp_vtt(seg.end)

            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    def to_md(
        self,
        transcript: AlignedTranscript,
        title: Optional[str] = None,
    ) -> str:
        """
        Format as clean Markdown transcript (no timestamps, bold speaker names).

        Args:
            transcript: Aligned transcript
            title: Optional title for the document

        Returns:
            Markdown formatted string
        """
        lines = []

        # Add title if provided
        if title:
            lines.append(f"# {title}")
            lines.append("")

        for seg in transcript.segments:
            speaker = self._get_speaker_name(seg.speaker)
            lines.append(f"**{speaker}:** {seg.text}")
            lines.append("")

        return "\n".join(lines)

    def save(
        self,
        transcript: AlignedTranscript,
        output_path: Path,
        format: Optional[str] = None,
        **kwargs,
    ) -> Path:
        """
        Save transcript to file.

        Args:
            transcript: Aligned transcript
            output_path: Output file path
            format: Format (json, txt, srt, vtt). Auto-detected from extension if None.
            **kwargs: Additional format-specific options

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)

        # Auto-detect format from extension
        if format is None:
            format = output_path.suffix.lower().lstrip(".")

        # Format content
        if format == "json":
            content = self.to_json(transcript, **kwargs)
        elif format == "txt":
            content = self.to_txt(transcript, **kwargs)
        elif format == "srt":
            content = self.to_srt(transcript, **kwargs)
        elif format == "vtt":
            content = self.to_vtt(transcript, **kwargs)
        elif format == "md":
            content = self.to_md(transcript, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Write file
        output_path.write_text(content, encoding="utf-8")
        return output_path


class TranscriptPrinter:
    """Pretty-prints transcripts to console."""

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
        self._colors = [
            "\033[94m",  # Blue
            "\033[92m",  # Green
            "\033[93m",  # Yellow
            "\033[95m",  # Magenta
            "\033[96m",  # Cyan
            "\033[91m",  # Red
        ]
        self._reset = "\033[0m"

    def _get_speaker_color(self, speaker: str, speakers: list) -> str:
        """Get color for speaker."""
        if not self.use_colors:
            return ""
        try:
            idx = speakers.index(speaker) % len(self._colors)
            return self._colors[idx]
        except ValueError:
            return ""

    def print(
        self,
        transcript: AlignedTranscript,
        speaker_names: Optional[dict] = None,
    ):
        """Print transcript with speaker colors."""
        speaker_names = speaker_names or {}

        for seg in transcript.segments:
            speaker_id = seg.speaker
            speaker = speaker_names.get(speaker_id, speaker_id)
            color = self._get_speaker_color(speaker_id, transcript.speakers)
            reset = self._reset if color else ""

            timestamp = format_timestamp_simple(seg.start)
            print(f"[{timestamp}] {color}{speaker}{reset}: {seg.text}")
